# SPDX-License-Identifier: Apache-2.0
"""CUDA-IPC memory reclaim on instance release (LMCache#4014).

The server imports each client's KV pool over CUDA IPC; when an instance is
released (unregister / reaper / close) those imported segments are only
returned to the driver by an ``empty_cache()`` + ``ipc_collect()`` pass run
AFTER every reference to the released entry is gone.

All tests drive the module through its public surface: the real constructor,
``register_kv_cache`` (with the module-level context factory stubbed),
``unregister_kv_cache`` / ``reap_stale_instances`` / ``close``, and
``context_entries_snapshot`` for reads. The stubbed boundaries are external
by nature: the GPU context factory, event IPC backend lookup, and the device
module (``torch_dev``).
"""

# Standard
# Standard Library
from concurrent.futures import ThreadPoolExecutor
from types import FrameType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
import sys
import threading
import time
import weakref

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.server import MPCacheServer
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as gpu_mod


class _FakeTorchDev:
    """Records the reclaim-call sequence; optionally omits ipc_collect."""

    empty_cache: MagicMock
    ipc_collect: MagicMock

    def __init__(self, with_ipc_collect: bool = True):
        self.calls: list[str] = []
        self.empty_cache = MagicMock(
            side_effect=lambda: self.calls.append("empty_cache")
        )
        if with_ipc_collect:
            self.ipc_collect = MagicMock(
                side_effect=lambda: self.calls.append("ipc_collect")
            )


def _module(monkeypatch) -> LMCacheDrivenTransferModule:
    """Construct the module through the real __init__ with stubbed deps."""
    monkeypatch.setattr(gpu_mod, "DeviceHostFuncDispatcher", MagicMock())
    return LMCacheDrivenTransferModule(MagicMock(name="ctx"))


def _register(
    module: LMCacheDrivenTransferModule,
    monkeypatch,
    instance_id: int,
    model: str = "m",
    age_s: float = 0.0,
) -> MagicMock:
    """Register an instance via the public API; return its cache context.

    ``age_s`` back-dates the registration (by stubbing the clock for the
    duration of the call) so reaper tests can create already-stale entries
    without touching module internals.

    Returns:
        The MagicMock standing in for the created cache context.
    """
    cache_context = MagicMock(name=f"cache_context-{instance_id}")
    cache_context.num_layers = 1
    event_backend = MagicMock(name=f"event_backend-{instance_id}")
    monkeypatch.setattr(gpu_mod, "create_cache_context", lambda *a, **kw: cache_context)
    monkeypatch.setattr(
        gpu_mod,
        "get_event_ipc_backend",
        lambda device: event_backend,
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    real_monotonic = time.monotonic
    if age_s:
        monkeypatch.setattr(gpu_mod.time, "monotonic", lambda: real_monotonic() - age_s)
    try:
        module.register_kv_cache(
            instance_id,
            kv_caches=MagicMock(name="kv_caches"),
            model_name=model,
            world_size=1,
            engine_type=MagicMock(name="engine_type"),
            layout_hints=MagicMock(name="layout_hints"),
            engine_group_infos=[],
        )
    finally:
        if age_s:
            monkeypatch.setattr(gpu_mod.time, "monotonic", real_monotonic)
    return cache_context


def test_unregister_reclaims_ipc_memory(monkeypatch) -> None:
    """Explicit unregister closes the context AND runs empty_cache +
    ipc_collect (in that order)."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx = _register(module, monkeypatch, 7)

    module.unregister_kv_cache(7)

    ctx.close.assert_called_once()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert module.context_entries_snapshot() == {}


def test_unregister_unknown_instance_does_not_reclaim(monkeypatch) -> None:
    """The warn path (already-reaped / never-registered id) must not touch
    the allocator — reclaim is tied to an actual release."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)

    module.unregister_kv_cache(404)

    assert dev.calls == []


def test_unregister_entry_refs_dead_before_ipc_collect(monkeypatch) -> None:
    """THE load-bearing ordering: ipc_collect only frees segments whose
    tensors are unreferenced, so the entry must be garbage by the time it
    fires. Verified with a weakref probed from inside the fake collector."""
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1)
    ref = weakref.ref(module.context_entries_snapshot()[1])

    seen: dict = {}
    dev = SimpleNamespace(
        empty_cache=lambda: None,
        ipc_collect=lambda: seen.setdefault("entry_alive", ref() is not None),
    )
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)

    module.unregister_kv_cache(1)

    assert seen == {"entry_alive": False}


def test_reaper_reclaims_once_per_batch(monkeypatch) -> None:
    """Reaping N stale instances closes each context but batches the
    allocator reclaim into ONE empty_cache + ipc_collect."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx_a = _register(module, monkeypatch, 1, model="a", age_s=1000.0)
    ctx_b = _register(module, monkeypatch, 2, model="b", age_s=1000.0)
    ctx_fresh = _register(module, monkeypatch, 3, model="c")

    reaped = module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert sorted(reaped) == [1, 2]
    ctx_a.close.assert_called_once()
    ctx_b.close.assert_called_once()
    ctx_fresh.close.assert_not_called()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert list(module.context_entries_snapshot()) == [3]


def test_reaper_noop_scan_does_not_reclaim(monkeypatch) -> None:
    """A scan that reaps nothing must not thrash the allocator."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1)

    reaped = module.reap_stale_instances(
        reap_timeout_s=3600.0, registration_grace_s=3600.0
    )

    assert reaped == []
    assert dev.calls == []


def test_reaper_entry_refs_dead_before_ipc_collect(monkeypatch) -> None:
    """Same ref-lifetime invariant on the reaper path."""
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1, age_s=1000.0)
    ref = weakref.ref(module.context_entries_snapshot()[1])

    seen: dict = {}
    dev = SimpleNamespace(
        empty_cache=lambda: None,
        ipc_collect=lambda: seen.setdefault("entry_alive", ref() is not None),
    )
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)

    module.reap_stale_instances(reap_timeout_s=60.0, registration_grace_s=60.0)

    assert seen == {"entry_alive": False}


def test_reclaim_degrades_without_ipc_collect(monkeypatch) -> None:
    """Device modules without ipc_collect (xpu / musa) must not raise —
    empty_cache still runs, the collect step is skipped."""
    dev = _FakeTorchDev(with_ipc_collect=False)
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    _register(module, monkeypatch, 9)

    module.unregister_kv_cache(9)

    assert dev.calls == ["empty_cache"]


def test_close_releases_all_and_reclaims_once(monkeypatch) -> None:
    """Server close() releases every remaining context and reclaims once."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    c1 = _register(module, monkeypatch, 1, model="a")
    c2 = _register(module, monkeypatch, 2, model="b")

    module.close()

    c1.close.assert_called_once()
    c2.close.assert_called_once()
    assert dev.calls == ["empty_cache", "ipc_collect"]
    assert module.context_entries_snapshot() == {}


def test_close_with_empty_registry_does_not_reclaim(monkeypatch) -> None:
    """close() on a server that never had clients skips the allocator."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)

    module.close()

    assert dev.calls == []


@pytest.mark.parametrize("operation", ["store", "retrieve"])
def test_returning_transfer_drops_entry_before_memory_collection(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A completed handler must not pin its entry while teardown collects IPC."""
    dev = MagicMock()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    cache_context = _register(module, monkeypatch, 1)
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    cache_context.kv_layer_groups_manager.num_kernel_groups = 1
    cache_context.calculate_num_blocks.return_value = 1
    cast(MagicMock, module.context).resolve_obj_keys.return_value = [[object()]]
    ref = weakref.ref(module.context_entries_snapshot()[1])
    entry_alive: list[bool] = []
    dev.ipc_collect.side_effect = lambda: entry_alive.append(ref() is not None)
    returning = threading.Event()
    resume = threading.Event()

    def pause_return(frame: FrameType, event: str, arg: object) -> None:
        # Pause the public handler after its body finishes, while its frame
        # still exists. This makes a retained local reference deterministic.
        if (
            event == "return"
            and frame.f_code.co_name == operation
            and frame.f_globals.get("__name__") == gpu_mod.__name__
        ):
            returning.set()
            assert resume.wait(5), "test did not release the returning handler"

    def transfer() -> None:
        sys.setprofile(pause_return)
        try:
            key = IPCCacheServerKey("m", 1, 0, (1,), 0, 1, "request")
            # An underfilled block list returns before any transfer is submitted.
            assert getattr(module, operation)(key, 1, [[]], b"evt")[1] is False
        finally:
            sys.setprofile(None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(transfer)
        try:
            assert returning.wait(5), "transfer did not reach its return"
            module.unregister_kv_cache(1)
            assert entry_alive == [False]
        finally:
            resume.set()
        pending.result(timeout=5)


@pytest.mark.parametrize(
    ("operation", "release"),
    [
        ("store", "unregister"),
        ("retrieve", "unregister"),
        ("store", "reap"),
        ("retrieve", "close"),
    ],
)
def test_release_waits_for_admitted_transfer(
    monkeypatch: pytest.MonkeyPatch, operation: str, release: str
) -> None:
    """An admitted handler survives draining; later requests submit no work."""
    monkeypatch.setattr(gpu_mod, "torch_dev", _FakeTorchDev())
    module = _module(monkeypatch)
    cache_context = _register(module, monkeypatch, 1)
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    context = cast(MagicMock, module.context)
    engine = MPCacheServer(module.context, [module])
    context.session_manager.get.return_value = None
    key = IPCCacheServerKey("m", 1, 0, (1,), 0, 1, "request")
    entered = threading.Event()
    resume = threading.Event()

    def resolve_keys(*args: object) -> None:
        entered.set()
        assert resume.wait(5), "test did not release the transfer"
        raise RuntimeError("transfer interrupted")

    context.resolve_obj_keys.side_effect = resolve_keys
    clock = [1000.0]
    monkeypatch.setattr(gpu_mod.time, "monotonic", lambda: clock[0])

    def release_context() -> None:
        if release == "unregister":
            module.unregister_kv_cache(1)
        elif release == "reap":
            assert module.reap_stale_instances(60.0, 60.0) == [1]
        else:
            module.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        transfer = executor.submit(getattr(module, operation), key, 1, [[1]], b"evt")
        assert entered.wait(5), "transfer did not reach key resolution"
        clock[0] += 1000.0
        cleanup = executor.submit(release_context)
        try:
            deadline = time.perf_counter() + 5
            while not module.context_entries_snapshot()[1].draining:
                assert time.perf_counter() < deadline, "release did not start"
                time.sleep(0.001)
            cache_context.close.assert_not_called()
            assert engine.cache_contexts == {}
            assert not cleanup.done()
            assert module.get_and_touch_context_entry(1) is None
            assert getattr(module, operation)(key, 1, [[1]], b"evt") == (b"", False)
            assert context.resolve_obj_keys.call_count == 1
            with pytest.raises(RuntimeError):
                _register(module, monkeypatch, 1)
        finally:
            resume.set()
        with pytest.raises(RuntimeError, match="transfer interrupted"):
            transfer.result(timeout=5)
        cleanup.result(timeout=5)

    cache_context.close.assert_called_once()
    assert module.context_entries_snapshot() == {}


@pytest.mark.parametrize("failure_at", ["context", "layout"])
def test_failed_batch_close_keeps_only_unreleased_entries_for_retry(
    monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    """Retry cannot admit a failed entry or restore an already closed one."""
    monkeypatch.setattr(gpu_mod, "torch_dev", _FakeTorchDev())
    module = _module(monkeypatch)
    engine = MPCacheServer(module.context, [module])
    first = _register(module, monkeypatch, 1, model="a")
    second = _register(module, monkeypatch, 2, model="b")
    if failure_at == "context":
        second.close.side_effect = RuntimeError("release failed")
    else:
        cast(MagicMock, module.context.layout_desc_registry.unregister).side_effect = [
            None,
            RuntimeError("release failed"),
            None,
        ]
    dispatcher = cast(MagicMock, gpu_mod.DeviceHostFuncDispatcher).return_value

    with pytest.raises(RuntimeError, match="release failed"):
        module.close()

    first.close.assert_called_once()
    assert set(module.context_entries_snapshot()) == {2}
    assert engine.cache_contexts == {}
    assert module.get_and_touch_context_entry(2) is None
    dispatcher.stop.assert_not_called()
    with pytest.raises(RuntimeError):
        _register(module, monkeypatch, 3)

    second.close.side_effect = None
    module.close()

    first.close.assert_called_once()
    assert second.close.call_count == (2 if failure_at == "context" else 1)
    assert module.context_entries_snapshot() == {}
    dispatcher.stop.assert_called_once()


def test_reaper_retries_draining_context_after_fresh_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A heartbeat cannot cancel cleanup already owed by a draining context."""
    monkeypatch.setattr(gpu_mod, "torch_dev", _FakeTorchDev())
    module = _module(monkeypatch)
    context = _register(module, monkeypatch, 1, age_s=120.0)
    context.close.side_effect = [RuntimeError("release failed"), None]
    _register(module, monkeypatch, 2)

    with pytest.raises(RuntimeError, match="release failed"):
        module.reap_stale_instances(60.0, 60.0)
    module.touch_instance(1)

    assert module.reap_stale_instances(60.0, 60.0) == [1]
    assert context.close.call_count == 2
    assert set(module.context_entries_snapshot()) == {2}


def test_close_waits_for_registration_being_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registration already admitted when close starts is also released."""
    monkeypatch.setattr(gpu_mod, "torch_dev", _FakeTorchDev())
    module = _module(monkeypatch)
    first = _register(module, monkeypatch, 1)
    second = MagicMock(num_layers=1)
    entered = threading.Event()
    resume = threading.Event()

    def create_context(*args: object, **kwargs: object) -> MagicMock:
        entered.set()
        assert resume.wait(5), "test did not release registration"
        return second

    monkeypatch.setattr(gpu_mod, "create_cache_context", create_context)
    with ThreadPoolExecutor(max_workers=2) as executor:
        registration = executor.submit(
            module.register_kv_cache,
            2,
            MagicMock(),
            "m",
            1,
            MagicMock(),
            MagicMock(),
            [],
        )
        assert entered.wait(5), "registration did not reach context construction"
        cleanup = executor.submit(module.close)
        try:
            deadline = time.perf_counter() + 5
            while module.get_and_touch_context_entry(1) is not None:
                assert time.perf_counter() < deadline, "close did not start"
                time.sleep(0.001)
            assert not cleanup.done()
            first.close.assert_not_called()
        finally:
            resume.set()
        registration.result(timeout=5)
        cleanup.result(timeout=5)

    first.close.assert_called_once()
    second.close.assert_called_once()
    assert module.context_entries_snapshot() == {}

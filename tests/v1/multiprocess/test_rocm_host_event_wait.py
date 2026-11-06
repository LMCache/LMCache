# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in ROCm host event wait of LMCache-driven transfers."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import argparse
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess import rocm_host_event_wait
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool, DeferredResponse
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
from lmcache.v1.multiprocess.rocm_host_event_wait import (
    ROCM_HOST_EVENT_WAIT_ENV,
    HostEventWaiter,
)
from tests.v1.multiprocess.test_event_ipc_handle_path import (
    _FakeEventBackend,
    _FakeStorageManager,
    _NoopDispatcher,
)

_THREAD_NAME = "lmcache-host-event-wait"


class _Backend(_FakeEventBackend):
    """Fake backend whose imported events complete when the test says so."""

    def __init__(self) -> None:
        super().__init__()
        self.done: set[bytes] = set()

    def query_event(self, event: object) -> bool:
        self.calls.append(("query", event))
        return cast(tuple, event)[1] in self.done


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def waiters():
    created: list[HostEventWaiter] = []

    def make(**kwargs: Any) -> HostEventWaiter:
        waiter = HostEventWaiter(**kwargs)
        created.append(waiter)
        return waiter

    yield make
    for waiter in created:
        waiter.stop()


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROCM_HOST_EVENT_WAIT_ENV, raising=False)


def _poller_threads() -> int:
    return sum(t.name == _THREAD_NAME for t in threading.enumerate())


def _build(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    hip: str | None,
    registry: bool = False,
) -> Any:
    """Build the module through ``create()`` with the option and platform.

    With ``registry`` the entry is registered in the module's real registry
    (``t.make_entry`` builds more); otherwise the lookup is patched to it.
    """
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(
        lmcache_driven_transfer, "DeviceHostFuncDispatcher", _NoopDispatcher
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "downsample_and_stage_block_ids", lambda c, b: b
    )
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "get_layout_desc",
        lambda cache_context, num_tokens, object_group_id: object(),
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "device", lambda d: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "stream", lambda s: nullcontext()
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "submit_callback_to_stream", lambda *args: None
    )
    monkeypatch.setattr(lmcache_driven_transfer, "fd_channel_path", lambda port: "")
    server_context = SimpleNamespace(
        chunk_size=1,
        null_block_id=0,
        storage_manager=_FakeStorageManager(),
        get_read_owners=lambda request_id: None,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: None,
            has_subscribers=lambda event_type: False,
        ),
        resolve_obj_keys=lambda key, group_ids: [[]],
        layout_desc_registry=SimpleNamespace(
            register=lambda *args, **kwargs: None,
            unregister=lambda *args, **kwargs: None,
        ),
    )
    build_ctx = SimpleNamespace(
        mp_config=SimpleNamespace(
            supported_transfer_mode="lmcache_driven",
            rocm_host_event_wait=enabled,
            port=0,
        ),
        engine_context=server_context,
    )
    module: Any = lmcache_driven_transfer.LMCacheDrivenTransferModule.create(
        cast(Any, build_ctx)
    )
    assert module is not None
    waiter = module._host_event_waiter

    def make_entry(backend: _Backend) -> Any:
        cache_context = SimpleNamespace(
            device=torch.device("cpu"),
            stream="transfer-stream",
            cupy_stream="cupy-stream",
            hold_imported_event=lambda event: 0,
            release_imported_event=lambda token: None,
            max_batch_size=1,
            kv_layer_groups_manager=SimpleNamespace(
                num_object_groups=1,
                num_kernel_groups=1,
                object_groups=[SimpleNamespace(kernel_group_indices=[0])],
                get_attn_desc=lambda: SimpleNamespace(
                    num_chunks_in_sw=[-1], group_kinds=()
                ),
            ),
            calculate_num_blocks=lambda chunk_size, group_idx: 1,
            transfer_object_group=lambda *args, **kwargs: None,
            num_layers=1,
            close=lambda: None,
        )
        return lmcache_driven_transfer.ContextEntry(
            cache_context=cast(Any, cache_context),
            model_name="model",
            world_size=1,
            event_backend=cast(
                Any, waiter.wrap(backend) if waiter is not None else backend
            ),
        )

    backend = _Backend()
    entry = make_entry(backend)
    if registry:
        module._cache_contexts[1] = entry
    else:
        monkeypatch.setattr(module, "get_and_touch_context_entry", lambda iid: entry)
    return SimpleNamespace(module=module, backend=backend, make_entry=make_entry)


def _key(request_id: str = "request") -> Any:
    return SimpleNamespace(request_id=request_id, cache_salt="", worker_id=0)


def _kinds(backend: _Backend) -> list[str]:
    return [call[0] for call in backend.calls]


# ---------------------------------------------------------------- option


def test_option_defaults_off_and_flag_enables_it() -> None:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    assert MPServerConfig.rocm_host_event_wait is False
    assert (
        parse_args_to_mp_server_config(parser.parse_args([])).rocm_host_event_wait
        is False
    )
    assert parse_args_to_mp_server_config(
        parser.parse_args(["--rocm-host-event-wait"])
    ).rocm_host_event_wait


@pytest.mark.parametrize("hip", [None, "7.0"])
def test_off_by_default_creates_no_waiter_and_keeps_device_wait(
    monkeypatch: pytest.MonkeyPatch, hip: str | None
) -> None:
    """Default on HIP and non-HIP: no waiter, no thread, upstream call order."""
    threads = _poller_threads()
    t = _build(monkeypatch, enabled=False, hip=hip)
    assert t.module._host_event_waiter is None
    assert _poller_threads() == threads
    assert "rocm_host_event_wait" not in t.module.report_status()

    assert t.module.store(_key(), 1, [[]], b"s") == (b"completion-handle", True)
    assert t.module.retrieve(_key(), 1, [[]], b"r") == (b"completion-handle", False)
    one = ["create", "import", "wait", "record", "export"]
    assert _kinds(t.backend) == one + one
    waited = [call[1] for call in t.backend.calls if call[0] == "wait"]
    assert waited == [("remote", b"s"), ("remote", b"r")]


def test_env_enables_it_on_hip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROCM_HOST_EVENT_WAIT_ENV, "1")
    t = _build(monkeypatch, enabled=False, hip="7.0")
    assert isinstance(t.module._host_event_waiter, HostEventWaiter)
    t.module._host_event_waiter.stop()


def test_on_without_hip_warns_and_keeps_device_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads = _poller_threads()
    logger = MagicMock()
    monkeypatch.setattr(rocm_host_event_wait, "logger", logger)
    t = _build(monkeypatch, enabled=True, hip=None)
    assert t.module._host_event_waiter is None
    assert _poller_threads() == threads
    logger.warning.assert_called_once()
    assert "only needed on ROCm" in logger.warning.call_args.args[0]
    assert t.module.store(_key(), 1, [[]], b"s") == (b"completion-handle", True)
    assert _kinds(t.backend) == ["create", "import", "wait", "record", "export"]


# ---------------------------------------------------------------- host path


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch):
    t = _build(monkeypatch, enabled=True, hip="7.0")
    yield t
    t.module._host_event_waiter.stop()


def test_completed_event_runs_inline_without_device_wait(host: Any) -> None:
    host.backend.done = {b"s", b"r"}
    assert host.module.store(_key(), 1, [[]], b"s") == (b"completion-handle", True)
    assert host.module.retrieve(_key(), 1, [[]], b"r") == (b"completion-handle", False)
    one = ["import", "query", "create", "record", "export"]
    assert _kinds(host.backend) == one + one
    stats = host.module.report_status()["rocm_host_event_wait"]
    assert stats == {
        "inline": 2,
        "deferred": 0,
        "timeout": 0,
        "shutdown": 0,
        "stale": 0,
        "device_waits": 0,
        "pending": 0,
    }


def test_pending_event_defers_until_it_completes(host: Any) -> None:
    response = host.module.store(_key(), 1, [[]], b"s")
    assert isinstance(response, DeferredResponse)
    time.sleep(0.01)
    assert not response.done()
    assert "create" not in _kinds(host.backend)
    host.backend.done.add(b"s")
    assert response.result(timeout=5) == (b"completion-handle", True)
    assert "wait" not in _kinds(host.backend)
    assert _kinds(host.backend).count("import") == 1
    stats = host.module._host_event_waiter.stats()
    assert (stats["deferred"], stats["device_waits"], stats["pending"]) == (1, 0, 0)


def test_transfers_keep_arrival_order_per_instance(host: Any) -> None:
    first = host.module.store(_key("a"), 1, [[]], b"a")
    host.backend.done.add(b"b")
    second = host.module.retrieve(_key("b"), 1, [[]], b"b")
    assert isinstance(second, DeferredResponse)
    time.sleep(0.01)
    assert not second.done()
    host.backend.done.add(b"a")
    assert first.result(timeout=5) == (b"completion-handle", True)
    assert second.result(timeout=5) == (b"completion-handle", False)
    created = [i for i, call in enumerate(host.backend.calls) if call[0] == "create"]
    assert len(created) == 2


def test_timeout_fails_store_and_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = _build(monkeypatch, enabled=True, hip="7.0")
    t.module._host_event_waiter.stop()
    clock = _Clock()
    waiter = HostEventWaiter(timeout_s=1.0, clock=clock)
    t.module._host_event_waiter = waiter
    try:
        store = t.module.store(_key("s"), 1, [[]], b"s")
        retrieve = t.module.retrieve(_key("r"), 1, [[]], b"r")
        clock.now = 2.0
        assert store.result(timeout=5) == (b"", False)
        assert retrieve.result(timeout=5) == (b"", False)
    finally:
        waiter.stop()
    assert "create" not in _kinds(t.backend)
    assert waiter.stats()["timeout"] == 2


def test_stop_fails_queued_transfers_and_later_ones(host: Any) -> None:
    waiter = host.module._host_event_waiter
    queued = host.module.retrieve(_key("q"), 1, [[]], b"q")
    waiter.stop()
    assert queued.result(timeout=5) == (b"", False)
    assert not waiter._thread.is_alive()
    assert host.module.store(_key("late"), 1, [[]], b"late") == (b"", False)
    assert "create" not in _kinds(host.backend)
    assert waiter.stats()["shutdown"] == 2


def test_unregistered_instance_is_rejected_inline(
    host: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host.module, "get_and_touch_context_entry", lambda iid: None)
    assert host.module.store(_key(), 1, [[]], b"s") == (b"", False)
    assert host.module.retrieve(_key(), 1, [[]], b"r") == (b"", False)
    assert host.backend.calls == []


def test_query_error_fails_the_transfer(
    host: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(event: object) -> bool:
        raise RuntimeError("query failed")

    monkeypatch.setattr(host.backend, "query_event", broken)
    response = host.module.store(_key(), 1, [[]], b"s")
    assert response.result(timeout=5) == (b"", False)
    assert "create" not in _kinds(host.backend)


def test_no_inline_run_while_the_poller_runs_the_same_instance(waiters: Any) -> None:
    waiter = waiters()
    backend = _Backend()
    entry = SimpleNamespace(
        event_backend=waiter.wrap(backend), cache_context=SimpleNamespace(device="d")
    )
    started = threading.Event()
    order: list[str] = []

    def slow() -> str:
        started.set()
        time.sleep(0.05)
        order.append("a")
        return "a"

    def fast() -> str:
        order.append("b")
        return "b"

    first = waiter.submit_after_event(7, entry, b"a", slow, lambda: None)
    backend.done.update({b"a", b"b"})
    assert started.wait(timeout=5)
    # Queue empty but the instance is busy: B must wait for A.
    second = waiter.submit_after_event(7, entry, b"b", fast, lambda: None)
    other = waiter.submit_after_event(8, entry, b"b", lambda: "c", lambda: None)
    assert isinstance(second, DeferredResponse)
    assert other == "c"
    assert (first.result(timeout=5), second.result(timeout=5)) == ("a", "b")
    assert order == ["a", "b"]
    assert waiter.stats()["inline"] == 1


# ---------------------------------------------------------------- lifecycle


def _until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.001)


def _entry(waiter: HostEventWaiter, backend: _Backend) -> Any:
    return SimpleNamespace(
        event_backend=waiter.wrap(backend), cache_context=SimpleNamespace(device="d")
    )


@pytest.mark.parametrize("ready_after_stop", [False, True])
def test_stop_during_inline_probe_fails_the_transfer(
    waiters: Any, ready_after_stop: bool
) -> None:
    """stop() while the inline probe runs: no stranded response, no late run."""
    waiter = waiters()
    entered, release = threading.Event(), threading.Event()

    class _Blocking(_Backend):
        def query_event(self, event: object) -> bool:
            entered.set()
            assert release.wait(timeout=5)
            return ready_after_stop

    ran: list[int] = []
    result: dict[str, Any] = {}

    def submit() -> None:
        result["value"] = waiter.submit_after_event(
            1,
            _entry(waiter, _Blocking()),
            b"e",
            lambda: ran.append(1),
            lambda: "failed",
        )

    submitter = threading.Thread(target=submit)
    submitter.start()
    assert entered.wait(timeout=5)
    stopper = threading.Thread(target=waiter.stop)
    stopper.start()
    _until(lambda: waiter._stopped)
    stopper.join(timeout=0.05)
    assert stopper.is_alive()  # stop() waits for the running probe
    release.set()
    submitter.join(timeout=5)
    stopper.join(timeout=5)
    assert not stopper.is_alive() and not waiter._thread.is_alive()
    assert result["value"] == "failed"
    assert ran == []
    assert waiter.stats()["shutdown"] == 1


@pytest.mark.parametrize("drop", ["unregister", "reap"])
@pytest.mark.parametrize("op", ["store", "retrieve"])
def test_reregistration_while_waiting_drops_the_transfer(
    monkeypatch: pytest.MonkeyPatch, op: str, drop: str
) -> None:
    """A waiting transfer never runs against a replacement registration."""
    t = _build(monkeypatch, enabled=True, hip="7.0", registry=True)
    module = t.module
    monkeypatch.setattr(lmcache_driven_transfer.torch_dev, "empty_cache", lambda: None)
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev, "ipc_collect", lambda: None, raising=False
    )
    replacement = _Backend()
    monkeypatch.setattr(
        module,
        "_create_cache_context",
        lambda *args, **kwargs: t.make_entry(replacement).cache_context,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer, "get_event_ipc_backend", lambda device: replacement
    )
    try:
        response = getattr(module, op)(_key("old"), 1, [[]], b"old")
        assert isinstance(response, DeferredResponse)
        if drop == "unregister":
            module.unregister_kv_cache(1)
        else:
            module._cache_contexts[1].last_seen = -1e9
            assert module.reap_stale_instances(1.0, 1.0) == [1]
        module.register_kv_cache(1, [], "model", 1, None, None, [])
        t.backend.done.add(b"old")
        assert response.result(timeout=5) == (b"", False)
        assert _kinds(replacement) == ["check"]
        assert "create" not in _kinds(t.backend)
        assert module._host_event_waiter.stats()["stale"] == 1
    finally:
        module._host_event_waiter.stop()


def test_waiting_transfer_runs_on_the_receiving_affinity_worker(
    waiters: Any,
) -> None:
    """The poller only polls; the transfer runs on the worker that received it,
    in order with that worker's other tasks (e.g. CacheBlend handlers)."""
    waiter = waiters()
    backend = _Backend()
    pool = AffinityThreadPool(max_workers=1, thread_name_prefix="aff")
    threads: list[str] = []

    def run() -> str:
        threads.append(threading.current_thread().name)
        return "ran"

    try:
        future = pool.submit(
            lambda: waiter.submit_after_event(
                1, _entry(waiter, backend), b"e", run, lambda: "failed"
            ),
            affinity_key=0,
        )
        pool.submit(
            lambda: threads.append("other@" + threading.current_thread().name),
            affinity_key=0,
        ).result(timeout=5)
        assert not future.done()
        backend.done.add(b"e")
        assert future.result(timeout=5) == "ran"
        assert threads == ["other@aff-0", "aff-0"]
    finally:
        pool.shutdown()


def test_stop_fails_a_handed_over_transfer_that_has_not_started(
    waiters: Any,
) -> None:
    """A transfer handed to a busy (or exited) worker is failed by stop()."""
    waiter = waiters()
    backend = _Backend()
    pool = AffinityThreadPool(max_workers=1)
    gate = threading.Event()
    ran: list[int] = []
    try:
        future = pool.submit(
            lambda: waiter.submit_after_event(
                1,
                _entry(waiter, backend),
                b"e",
                lambda: ran.append(1),
                lambda: "failed",
            ),
            affinity_key=0,
        )
        blocker = pool.submit(gate.wait, 5, affinity_key=0)
        _until(lambda: waiter.stats()["deferred"] == 1)
        backend.done.add(b"e")
        _until(lambda: bool(waiter._posted))
        waiter.stop()
        assert future.result(timeout=5) == "failed"
        gate.set()
        blocker.result(timeout=5)
        pool.submit(lambda: None, affinity_key=0).result(timeout=5)
        assert ran == []
        assert waiter.stats()["shutdown"] == 1
    finally:
        gate.set()
        pool.shutdown()


# ---------------------------------------------------------------- affinity pool


def test_affinity_pool_completes_future_from_deferred_response() -> None:
    pool = AffinityThreadPool(max_workers=1)
    deferred = DeferredResponse()
    failing = DeferredResponse()
    try:
        first = pool.submit(lambda: deferred, affinity_key=0)
        second = pool.submit(lambda: failing, affinity_key=0)
        third = pool.submit(lambda: "next", affinity_key=0)
        # The worker is not blocked by the deferred responses.
        assert third.result(timeout=5) == "next"
        assert not first.done()
        deferred.set_result(("handle", True))
        failing.set_exception(RuntimeError("boom"))
        assert first.result(timeout=5) == ("handle", True)
        with pytest.raises(RuntimeError, match="boom"):
            second.result(timeout=5)
    finally:
        pool.shutdown()

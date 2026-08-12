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
from types import SimpleNamespace

# Third Party
import pytest
from unittest.mock import MagicMock
import time
import weakref

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
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


class _StoppedPeriodicThread:
    """Stub for create_periodic_thread — never starts the background loop.

    The periodic IPC collector thread fires _ipc_collect_cycle every 60 s
    in the background.  Without this stub, background calls to
    empty_cache/ipc_collect race with the test assertions on dev.calls.
    Tests that need the collector call module._ipc_collect_cycle() directly.
    """

    name = "lmcache-ipc-collector-stub"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _module(monkeypatch) -> LMCacheDrivenTransferModule:
    """Construct the module through the real __init__ with stubbed deps."""
    monkeypatch.setattr(gpu_mod, "DeviceHostFuncDispatcher", MagicMock())
    monkeypatch.setattr(gpu_mod, "create_periodic_thread", lambda **kw: _StoppedPeriodicThread())
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


# =============================================================================
# Store/retrieve through @_pins_context_entry decorator
# =============================================================================
# Regression guard for the bug where the decorator never passed the pinned
# entry to the method body, causing `entry = _pinned_entry` (None) to crash
# with AttributeError at `entry.cache_context`.  These tests drive the real
# decorated store/retrieve through the public API and prove the body receives
# a non-None pinned entry by reaching a sentinel deep in the body.


class _SentinelError(Exception):
    """Raised by a stubbed ctx method to prove the body reached that line."""


def _module_with_sentinel_at_resolve(monkeypatch):
    """Build a module whose ``ctx.resolve_obj_keys`` raises ``_SentinelError``.

    Both store and retrieve call ``self._ctx.resolve_obj_keys(key, ...)`` right
    after ``cache_context = entry.cache_context``.  If ``_pinned_entry`` were
    None, the body would crash with ``AttributeError`` *before* reaching
    resolve_obj_keys.  So a ``_SentinelError`` from resolve_obj_keys proves the
    pinned entry was passed and its cache_context was accessed successfully.
    """
    module = _module(monkeypatch)
    module._ctx.resolve_obj_keys.side_effect = _SentinelError
    return module


def test_store_receives_pinned_entry_not_none(monkeypatch) -> None:
    """store() body must get the pinned entry, not None (bug4 regression)."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module_with_sentinel_at_resolve(monkeypatch)
    _register(module, monkeypatch, 1)

    raised = False
    try:
        module.store(
            key=MagicMock(name="key"),
            instance_id=1,
            gpu_block_ids=[[0]],
            event_ipc_handle=b"\x00",
        )
    except _SentinelError:
        raised = True
    except AttributeError as e:
        pytest.fail(
            f"store got _pinned_entry=None (bug4 regression): {e}"
        )
    assert raised, "store body never reached resolve_obj_keys"
    # checkin must have run (pin released)
    assert module.context_entries_snapshot()[1].in_use == 0


def test_retrieve_receives_pinned_entry_not_none(monkeypatch) -> None:
    """retrieve() body must get the pinned entry, not None (bug4 regression)."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module_with_sentinel_at_resolve(monkeypatch)
    _register(module, monkeypatch, 2)

    raised = False
    try:
        module.retrieve(
            key=MagicMock(name="key"),
            instance_id=2,
            gpu_block_ids=[[0]],
            event_ipc_handle=b"\x00",
        )
    except _SentinelError:
        raised = True
    except AttributeError as e:
        pytest.fail(
            f"retrieve got _pinned_entry=None (bug4 regression): {e}"
        )
    assert raised, "retrieve body never reached resolve_obj_keys"
    assert module.context_entries_snapshot()[2].in_use == 0


def test_store_raises_value_error_for_unknown_instance(monkeypatch) -> None:
    """store() on an unregistered instance must raise ValueError, not crash."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)

    with pytest.raises(ValueError, match="instance ID 999"):
        module.store(
            key=MagicMock(name="key"),
            instance_id=999,
            gpu_block_ids=[[0]],
            event_ipc_handle=b"\x00",
        )


# =============================================================================
# Dead-entry safety net: busy entry retirement + deferred release
# =============================================================================


def test_busy_retire_enters_dead_entries(monkeypatch) -> None:
    """A checked-out entry retired by unregister must enter _dead_entries."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1)

    # Pin the entry (simulate an in-flight transfer)
    entry = module.checkout_context_entry(1)
    assert entry is not None
    assert entry.in_use == 1

    # Retire while busy — should defer, not release
    module.unregister_kv_cache(1)

    assert entry in module._dead_entries, "busy entry should be in _dead_entries"
    assert entry.dead is True
    # Context not yet closed (transfer may still be using it)
    entry.cache_context.close.assert_not_called()
    assert module.context_entries_snapshot() == {}

    # Clean up: checkin to release
    module.checkin_context_entry(entry)
    entry.cache_context.close.assert_called_once()


def test_dead_entry_checkin_releases_exactly_once(monkeypatch) -> None:
    """After retire + checkin, the context is closed exactly once."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx = _register(module, monkeypatch, 1)

    entry = module.checkout_context_entry(1)
    module.unregister_kv_cache(1)

    assert entry in module._dead_entries

    module.checkin_context_entry(entry)

    ctx.close.assert_called_once()
    assert entry not in module._dead_entries
    assert entry.released is True
    assert dev.calls == ["empty_cache", "ipc_collect"]


def test_dead_entry_timeout_force_releases(monkeypatch) -> None:
    """_ipc_collect_cycle force-releases entries past _CHECKIN_TIMEOUT_S."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx = _register(module, monkeypatch, 1)

    entry = module.checkout_context_entry(1)
    module.unregister_kv_cache(1)
    assert entry in module._dead_entries

    # Backdate the dead-entry timestamp to simulate timeout
    module._dead_entries[entry] = time.monotonic() - 999.0

    summary = module._ipc_collect_cycle()

    ctx.close.assert_called_once()
    assert entry not in module._dead_entries
    assert entry.released is True
    assert "force_released=1" in summary.message
    assert dev.calls == ["empty_cache", "ipc_collect"]


def test_ipc_collect_noop_when_no_dead_entries(monkeypatch) -> None:
    """_ipc_collect_cycle with no dead entries just runs global collect."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    _register(module, monkeypatch, 1)

    summary = module._ipc_collect_cycle()

    assert "force_released=0" in summary.message
    assert dev.calls == ["empty_cache", "ipc_collect"]


# =============================================================================
# Double-release race: checkin vs _ipc_collect_cycle
# =============================================================================


def test_no_double_release_when_checkin_wins(monkeypatch) -> None:
    """If checkin releases first, the collector must not release again."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx = _register(module, monkeypatch, 1)

    entry = module.checkout_context_entry(1)
    module.unregister_kv_cache(1)
    # Backdate so the collector *would* fire if the entry were still tracked
    module._dead_entries[entry] = time.monotonic() - 999.0

    # checkin wins the race: releases and pops from _dead_entries
    module.checkin_context_entry(entry)
    assert ctx.close.call_count == 1
    assert entry not in module._dead_entries
    assert entry.released is True

    # Now the collector runs — should find nothing to release
    module._ipc_collect_cycle()
    assert ctx.close.call_count == 1, "collector must not double-release"


def test_no_double_release_when_collector_wins(monkeypatch) -> None:
    """If the collector releases first, a late checkin must not release again."""
    dev = _FakeTorchDev()
    monkeypatch.setattr(gpu_mod, "torch_dev", dev)
    module = _module(monkeypatch)
    ctx = _register(module, monkeypatch, 1)

    entry = module.checkout_context_entry(1)
    module.unregister_kv_cache(1)
    module._dead_entries[entry] = time.monotonic() - 999.0

    # Collector wins the race
    module._ipc_collect_cycle()
    assert ctx.close.call_count == 1
    assert entry.released is True
    assert entry not in module._dead_entries

    # Late checkin: must detect released=True and NOT release again
    module.checkin_context_entry(entry)
    assert ctx.close.call_count == 1, "late checkin must not double-release"
    assert entry.in_use == 0

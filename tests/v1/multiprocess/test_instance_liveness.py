# SPDX-License-Identifier: Apache-2.0
"""Tests for worker liveness tracking + reaping in MPCacheEngine.

Covers the new `_liveness` table, `_reap_worker` CAS semantics, the
``ping(instance_id)`` handler, and the reaper-stop-on-close ordering.
The full engine is heavy to spin up (storage manager + GPU), so we
construct via ``__new__`` and wire only the attributes the unit under
test reaches.
"""

# Standard
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.protocols.controller import PING_SENTINEL_INSTANCE_ID
from lmcache.v1.multiprocess.server import (
    MPCacheEngine,
    _InstanceLiveness,
    _InstanceReaperThread,
)


def _bare_engine(reap_after: float = 30.0) -> MPCacheEngine:
    """Construct an engine with only the liveness machinery wired.

    Skips storage manager / event bus / observability so each test stays
    a focused unit test of the liveness path.
    """
    engine = MPCacheEngine.__new__(MPCacheEngine)
    engine.gpu_contexts = {}
    engine.gpu_context_meta = {}
    engine.chunk_size = 256  # consumed by `register_kv_cache` ctor path
    engine.reap_after_seconds = reap_after
    engine._liveness = {}
    engine._liveness_lock = threading.Lock()
    return engine


# =============================================================================
# register / unregister / refresh
# =============================================================================


def test_register_overwrite_unregisters_prior_context(monkeypatch):
    """If `register_kv_cache` is called twice with the same `instance_id`,
    the first context must be cleaned up before the new one is installed.

    This is defensive — random 63-bit ids make collisions astronomically
    unlikely — but the cleanup is the safety net that prevents leaked
    CUDA IPC handles in the (impossible-in-practice) collision case.

    Calls the real `engine.register_kv_cache` (with `GPUCacheContext`
    construction and `torch.cuda.empty_cache` stubbed out) so the test
    fails if anyone removes the defensive `unregister_kv_cache` call.
    """
    engine = _bare_engine()
    iid = 0xDEADBEEF

    # Stub out the GPU-touching pieces. `register_kv_cache` builds a
    # `GPUCacheContext` and then — when the defensive branch fires —
    # calls `unregister_kv_cache`, which itself calls
    # `torch.cuda.empty_cache`. We pin a MagicMock on `empty_cache` so
    # we can assert the defensive branch ran (a plain dict-overwrite
    # would NOT call empty_cache, so the call is uniquely produced by
    # the `unregister_kv_cache(instance_id)` line under test).
    fake_ctx = MagicMock()
    fake_ctx.num_layers = 7

    def fake_gpu_context_ctor(*_args, **_kwargs):
        return fake_ctx

    monkeypatch.setattr(
        "lmcache.v1.multiprocess.server.GPUCacheContext",
        fake_gpu_context_ctor,
    )
    empty_cache_mock = MagicMock()
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.server.torch.cuda.empty_cache",
        empty_cache_mock,
    )

    # Seed prior incarnation.
    old_ctx = MagicMock()
    engine.gpu_contexts[iid] = old_ctx
    engine.gpu_context_meta[iid] = ("model", 1)
    engine._liveness[iid] = _InstanceLiveness(
        last_seen=time.monotonic() - 10.0,
        registered_at=time.monotonic() - 10.0,
    )

    # Real call. `KVCache`, `EngineType`, and `LayoutHints` are payload
    # types that GPUCacheContext consumes — we passed a stub so it's
    # never accessed. None placeholders are fine.
    engine.register_kv_cache(
        instance_id=iid,
        kv_caches=None,  # type: ignore[arg-type]
        model_name="model",
        world_size=1,
        engine_type=None,  # type: ignore[arg-type]
        layout_hints=None,  # type: ignore[arg-type]
    )

    # The new context replaced the old one.
    assert engine.gpu_contexts[iid] is fake_ctx
    assert engine.gpu_contexts[iid] is not old_ctx
    # Liveness was overwritten with a fresh entry.
    assert iid in engine._liveness
    fresh_now = time.monotonic()
    assert fresh_now - engine._liveness[iid].last_seen < 1.0
    # **Load-bearing assertion**: `empty_cache` is called only via
    # `unregister_kv_cache`, and only when the entry was present in
    # `gpu_contexts`. Seeing it called proves the defensive branch in
    # `register_kv_cache` ran. Removing that branch would leave a
    # plain dict-overwrite, and this assertion would fail.
    empty_cache_mock.assert_called_once()


def test_unregister_pops_liveness_and_idempotent():
    """`unregister_kv_cache` must drop the `_liveness` entry and be
    callable twice without raising (idempotent on missing entries).

    Calls the real method. With `gpu_contexts` empty the `if … in` guard
    short-circuits before any CUDA call, so we don't need to stub torch.
    """
    engine = _bare_engine()
    iid = 1234567

    # Pre-populate the liveness entry; leave gpu_contexts empty so the
    # CUDA path inside `unregister_kv_cache` is skipped (the warning log
    # branch is fine for our purposes).
    engine._liveness[iid] = _InstanceLiveness(
        last_seen=time.monotonic(), registered_at=time.monotonic()
    )

    engine.unregister_kv_cache(iid)
    assert iid not in engine._liveness

    # Second unregister: must not raise.
    engine.unregister_kv_cache(iid)
    assert iid not in engine._liveness


def test_refresh_liveness_updates_last_seen():
    engine = _bare_engine()
    iid = 42
    t0 = time.monotonic() - 100.0
    engine._liveness[iid] = _InstanceLiveness(last_seen=t0, registered_at=t0)

    engine._refresh_liveness(iid)
    assert engine._liveness[iid].last_seen > t0
    assert engine._liveness[iid].registered_at == t0  # not touched


def test_refresh_liveness_silent_on_unknown():
    """Unknown id: no exception, no insert. Aligns with the design's
    'the existing assert in store/retrieve will fire next' policy."""
    engine = _bare_engine()
    engine._refresh_liveness(99999)
    assert 99999 not in engine._liveness


# =============================================================================
# ping
# =============================================================================


def test_ping_sentinel_returns_true_without_tracking():
    engine = _bare_engine()
    assert engine.ping(PING_SENTINEL_INSTANCE_ID) is True
    assert PING_SENTINEL_INSTANCE_ID not in engine._liveness


def test_ping_known_returns_true_and_refreshes():
    engine = _bare_engine()
    iid = 7777
    t0 = time.monotonic() - 50.0
    engine._liveness[iid] = _InstanceLiveness(last_seen=t0, registered_at=t0)

    assert engine.ping(iid) is True
    assert engine._liveness[iid].last_seen > t0


def test_ping_unknown_returns_false_no_insert():
    """Terminal signal to the adapter — must NOT auto-resurrect the entry."""
    engine = _bare_engine()
    assert engine.ping(11111) is False
    assert 11111 not in engine._liveness


# =============================================================================
# reaper / _reap_worker
# =============================================================================


def test_reap_worker_evicts_stale_entry():
    engine = _bare_engine(reap_after=30.0)
    iid = 555
    # Stale: last seen well past the deadline.
    far_past = time.monotonic() - 100.0
    engine._liveness[iid] = _InstanceLiveness(
        last_seen=far_past, registered_at=far_past
    )
    engine.gpu_contexts[iid] = MagicMock()
    engine.gpu_context_meta[iid] = ("m", 1)

    # Substitute a fake unregister to avoid touching CUDA.
    unregister_calls = []

    def fake_unregister(instance_id):
        unregister_calls.append(instance_id)
        engine.gpu_contexts.pop(instance_id, None)
        engine.gpu_context_meta.pop(instance_id, None)

    engine.unregister_kv_cache = fake_unregister  # type: ignore[method-assign]
    engine._reap_worker(iid, time.monotonic())

    assert iid not in engine._liveness
    assert iid not in engine.gpu_contexts
    assert unregister_calls == [iid]


def test_reap_worker_skips_recently_refreshed_entry():
    """CAS race: PING refreshed `last_seen` between reaper snapshot and
    the moment `_reap_worker` re-checks. Must skip without unregistering."""
    engine = _bare_engine(reap_after=30.0)
    iid = 888
    snapshot_now = time.monotonic()
    # Stamp `last_seen` AFTER the snapshot — simulates a PING that
    # arrived in the gap.
    engine._liveness[iid] = _InstanceLiveness(
        last_seen=snapshot_now - 1.0,
        registered_at=snapshot_now - 1.0,
    )

    unregister_calls = []
    engine.unregister_kv_cache = (  # type: ignore[method-assign]
        lambda i: unregister_calls.append(i)
    )

    engine._reap_worker(iid, snapshot_now)
    assert iid in engine._liveness
    assert unregister_calls == []


def test_reap_worker_skips_already_evicted_entry():
    engine = _bare_engine(reap_after=30.0)
    unregister_calls = []
    engine.unregister_kv_cache = (  # type: ignore[method-assign]
        lambda i: unregister_calls.append(i)
    )

    # Entry never existed — reap is a no-op.
    engine._reap_worker(404, time.monotonic())
    assert unregister_calls == []


def test_reaper_thread_snapshots_only_stale_entries():
    """End-to-end sanity for the reaper's `_execute` snapshot pass."""
    engine = _bare_engine(reap_after=30.0)
    now = time.monotonic()
    engine._liveness[1] = _InstanceLiveness(
        last_seen=now - 100.0, registered_at=now - 100.0
    )  # stale
    engine._liveness[2] = _InstanceLiveness(
        last_seen=now - 1.0, registered_at=now - 1.0
    )  # fresh

    reaped = []
    engine.unregister_kv_cache = (  # type: ignore[method-assign]
        lambda i: reaped.append(i)
    )

    reaper = _InstanceReaperThread(engine=engine, reaper_interval_seconds=10.0)
    summary = reaper._execute()

    assert summary.success is True
    assert reaped == [1]
    assert 1 not in engine._liveness
    assert 2 in engine._liveness


def test_reaper_does_not_hold_liveness_lock_during_cleanup():
    """The reaper releases `_liveness_lock` between snapshot and cleanup
    so concurrent PINGs don't block on GPU work in `unregister_kv_cache`.

    We assert this by detecting that a thread can acquire `_liveness_lock`
    while `unregister_kv_cache` is executing.
    """
    engine = _bare_engine(reap_after=30.0)
    far_past = time.monotonic() - 100.0
    engine._liveness[1] = _InstanceLiveness(last_seen=far_past, registered_at=far_past)

    held_during_unregister = threading.Event()
    proceed_unregister = threading.Event()

    def slow_unregister(_iid):
        # Try to grab `_liveness_lock` from this thread (the reaper
        # body) — if reaper held the lock, we'd deadlock. Instead we
        # spawn a probe to take it while we're "in" unregister.
        def probe():
            with engine._liveness_lock:
                held_during_unregister.set()

        t = threading.Thread(target=probe)
        t.start()
        # Give the probe a chance to acquire the lock concurrently.
        proceed_unregister.wait(timeout=2.0)
        t.join(timeout=2.0)

    engine.unregister_kv_cache = slow_unregister  # type: ignore[method-assign]
    proceed_unregister.set()  # let unregister return immediately

    reaper = _InstanceReaperThread(engine=engine, reaper_interval_seconds=10.0)
    reaper._execute()

    assert held_during_unregister.is_set(), (
        "Probe should have acquired _liveness_lock while reaper was in "
        "the unregister callback — reaper must not hold it across cleanup."
    )


# =============================================================================
# close ordering
# =============================================================================


def test_close_stops_reaper_first():
    """`MPCacheEngine.close()` must stop the reaper before the storage
    manager closes, so no reap can fire against a half-closed engine."""
    engine = _bare_engine()
    # Wire the bits `close()` touches.
    storage_manager = MagicMock()
    storage_manager.close.return_value = None
    engine.storage_manager = storage_manager
    engine._reaper = _InstanceReaperThread(engine=engine, reaper_interval_seconds=10.0)
    engine._reaper.start()

    # Record the order of `_reaper.stop()` vs `storage_manager.close()`.
    call_order: list[str] = []
    real_stop = engine._reaper.stop

    def tracked_stop(*args, **kwargs):
        call_order.append("reaper_stop")
        return real_stop(*args, **kwargs)

    engine._reaper.stop = tracked_stop  # type: ignore[method-assign]

    def tracked_close():
        call_order.append("storage_close")

    storage_manager.close = tracked_close

    engine.close()

    assert call_order == ["reaper_stop", "storage_close"]
    assert not engine._reaper.is_running
    assert engine._liveness == {}


# =============================================================================
# Sentinel constant is shared, not re-defined
# =============================================================================


def test_sentinel_is_shared_constant():
    """Both server and adapter import the wire-protocol sentinel from the
    same source of truth (controller.py). This test guards against future
    drift where someone hard-codes 0 in either side."""
    # First Party
    from lmcache.integration.vllm import vllm_multi_process_adapter
    from lmcache.v1.multiprocess.protocols import controller

    assert (
        controller.PING_SENTINEL_INSTANCE_ID
        == vllm_multi_process_adapter.PING_SENTINEL_INSTANCE_ID
        == PING_SENTINEL_INSTANCE_ID
        == 0
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

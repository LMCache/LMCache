# SPDX-License-Identifier: Apache-2.0

"""Tests for MaruL1Manager (fake maru runtime, no CXL required)."""

# Standard
from unittest.mock import MagicMock
import time

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_protocol import L1ManagerInterface
    from lmcache.v1.distributed.maru_l1_manager import (
        MaruL1Manager,
        _PendingRead,
        object_key_to_string,
    )
    from lmcache.v1.mp_observability.event import EventType

    # Local
    from .maru_fakes import RecordingListener, make_maru_manager
    from .test_l1_protocol import _interface_methods, _params, _unwrap
except ImportError:
    pytest.skip("maru manager deps unavailable", allow_module_level=True)

_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([4, 8])], dtypes=[torch.float16])


def _key(idx: int, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=idx.to_bytes(4, "big"),
        model_name="test-model",
        kv_rank=0xABCD,
        cache_salt=salt,
    )


def _seed(handler, key: ObjectKey, rid: int = 7, pid: int = 3, size: int = 64):
    """Register a key in the fake directory (as if another instance stored it)."""
    handler.store_map[object_key_to_string(key)] = (rid, pid, size)


def _store(manager, keys: list[ObjectKey]):
    """Drive the full write path: reserve + finish."""
    res = manager.reserve_write(keys, [False] * len(keys), _LAYOUT, mode="new")
    assert all(err == L1Error.SUCCESS for err, _ in res.values())
    fin = manager.finish_write(keys)
    assert all(err == L1Error.SUCCESS for err in fin.values())


# =========================================================================
# interface conformance (binds MaruL1Manager to the shared Protocol)
# =========================================================================


def test_conforms_to_l1_manager_interface():
    assert issubclass(MaruL1Manager, L1ManagerInterface)


def test_signatures_match_interface():
    """Each Protocol method's call shape matches MaruL1Manager's."""
    for name in _interface_methods():
        proto = _params(getattr(L1ManagerInterface, name))
        impl = _params(_unwrap(getattr(MaruL1Manager, name)))
        assert proto == impl, name


# =========================================================================
# reserve_read / unsafe_read / finish_read
# =========================================================================


def test_reserve_read_hit_pins_and_stages():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)

    res = manager.reserve_read([k])
    err, obj = res[k]
    assert err == L1Error.SUCCESS and obj is not None
    assert handler.pins[object_key_to_string(k)] == 1
    assert manager.unsafe_read([k])[k] == (L1Error.SUCCESS, obj)


def test_reserve_read_miss_leaves_no_pin():
    manager, handler, _ = make_maru_manager()
    res = manager.reserve_read([_key(1)])
    assert res[_key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert not handler.pins


def test_reserve_read_is_per_key_independent():
    """A miss in the middle must not shadow later hits (unlike prefix-stop)."""
    manager, handler, _ = make_maru_manager()
    k1, k2, k3 = _key(1), _key(2), _key(3)
    _seed(handler, k1, pid=1)
    _seed(handler, k3, pid=3)

    res = manager.reserve_read([k1, k2, k3])
    assert res[k1][0] == L1Error.SUCCESS
    assert res[k2] == (L1Error.KEY_NOT_EXIST, None)
    assert res[k3][0] == L1Error.SUCCESS
    assert handler.pins[object_key_to_string(k3)] == 1


def test_extra_count_takes_and_releases_n_pins():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)
    _seed(handler, k)

    manager.reserve_read([k], extra_count=2)
    assert handler.pins[ks] == 3

    # Three independent finish_read calls balance the three pins.
    for _ in range(3):
        assert manager.finish_read([k])[k] == L1Error.SUCCESS
    assert handler.pins[ks] == 0
    assert manager.unsafe_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)


def test_overlapping_reserve_accumulates_refcount():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)
    _seed(handler, k)

    _, obj1 = manager.reserve_read([k])[k]
    _, obj2 = manager.reserve_read([k])[k]
    assert obj1 is obj2  # one staged object per key
    assert handler.pins[ks] == 2

    manager.finish_read([k])
    assert manager.unsafe_read([k])[k][0] == L1Error.SUCCESS  # still staged
    manager.finish_read([k])
    assert handler.pins[ks] == 0


def test_finish_read_unstaged_does_not_unpin():
    manager, handler, _ = make_maru_manager()
    assert manager.finish_read([_key(1)])[_key(1)] == L1Error.KEY_NOT_EXIST
    assert not handler.unpin_log


def test_finish_read_never_over_releases():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])  # one pin

    assert manager.finish_read([k], extra_count=5)[k] == L1Error.SUCCESS
    assert len(handler.unpin_log) == 1  # released min(6, refcount=1)
    assert handler.pins[object_key_to_string(k)] == 0


def test_temporary_read_absorbing_overlapping_pins_releases_them():
    """A peer-registered key read while temporary-staged must not leak pins.

    A temporary promote stages ``k`` locally with no server pin. If a peer has
    registered ``k`` in the directory, a later ``reserve_read`` pins the shared
    copy and folds those pins into the temporary entry. Draining the reads must
    release every absorbed pin -- regression: the temporary release path used to
    reclaim the page without unpinning, leaking the pin permanently (the page
    became un-evictable on MaruServer).
    """
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)

    # Temporary promote: staged in _pending_read, no directory pin.
    manager.reserve_write([k], [True], _LAYOUT, mode="new")
    manager.finish_write_and_reserve_read([k])
    assert handler.pins.get(ks, 0) == 0  # temporary -> no server pin

    # A peer registers k; an overlapping reserve_read pins the shared copy and
    # absorbs that pin onto the temporary entry.
    _seed(handler, k)
    manager.reserve_read([k])
    assert handler.pins[ks] == 1

    # Drain both holds (temporary hold + absorbed read): every pin released.
    manager.finish_read([k])
    manager.finish_read([k])
    assert handler.pins[ks] == 0  # was 1 (leaked) before the fix
    assert ks not in manager._pending_read


def test_reserve_read_excludes_mid_write_key_no_double_staging():
    """A key mid-write on this instance stays KEY_NOT_READABLE even if a peer
    registered it -- it must not be pinned or read-staged.

    A local write reserves ``k`` (staged in _pending_write); before it finishes,
    a peer registers ``k`` in the shared directory. A reserve_read that slips in
    must refuse ``k``, not serve the peer copy: staging it would leave ``k`` in
    both _pending_write and _pending_read (double staging) and strand the write
    -- its promote would then return KEY_IN_WRONG_STATE and never pop
    _pending_write, orphaning the page until the write-TTL sweeper.
    """
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)

    # Local write in flight: k staged for write, not yet registered by us.
    manager.reserve_write([k], [True], _LAYOUT, mode="new")
    # A peer registers the same key in the shared directory mid-write.
    _seed(handler, k)

    # A read racing the in-flight write is refused, not served from the peer.
    res = manager.reserve_read([k])
    assert res[k] == (L1Error.KEY_NOT_READABLE, None)
    assert handler.pins.get(ks, 0) == 0  # no pin taken on the peer copy
    assert k not in manager._pending_read  # no double staging
    assert k in manager._pending_write  # write still in flight

    # The write still completes cleanly: promote pops _pending_write and stages
    # the read (no orphan, no KEY_IN_WRONG_STATE).
    fin = manager.finish_write_and_reserve_read([k])
    assert fin[k][0] == L1Error.SUCCESS
    assert k not in manager._pending_write


# =========================================================================
# reserve_write / finish_write
# =========================================================================


def test_write_path_registers_key():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _store(manager, [k])
    assert object_key_to_string(k) in handler.store_map
    # Now readable through the directory.
    assert manager.reserve_read([k])[k][0] == L1Error.SUCCESS


def test_reserve_write_rejects_staged_and_registered_keys():
    manager, handler, _ = make_maru_manager()
    staged, registered, fresh = _key(1), _key(2), _key(3)
    manager.reserve_write([staged], [False], _LAYOUT, mode="new")
    _seed(handler, registered)

    res = manager.reserve_write(
        [staged, registered, fresh], [False] * 3, _LAYOUT, mode="new"
    )
    assert res[staged] == (L1Error.KEY_NOT_WRITABLE, None)  # locally staged
    assert res[registered] == (L1Error.KEY_NOT_WRITABLE, None)  # cross-instance dedup
    assert res[fresh][0] == L1Error.SUCCESS


def test_reserve_write_oom_marks_whole_batch():
    manager, _, adapter = make_maru_manager()
    adapter.oom = True
    res = manager.reserve_write([_key(1), _key(2)], [False] * 2, _LAYOUT, mode="new")
    assert all(v == (L1Error.OUT_OF_MEMORY, None) for v in res.values())


def test_reserve_write_rejects_non_new_mode():
    manager, _, _ = make_maru_manager()
    with pytest.raises(ValueError):
        manager.reserve_write([_key(1)], [False], _LAYOUT, mode="update")


def test_finish_write_unstaged_key():
    manager, _, _ = make_maru_manager()
    assert manager.finish_write([_key(1)])[_key(1)] == L1Error.KEY_NOT_EXIST


def test_finish_write_store_failure_reclaims_page():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    handler.fail_store_keys.add(object_key_to_string(k))

    res = manager.reserve_write([k], [False], _LAYOUT, mode="new")
    addr = res[k][1].metadata.address
    assert manager.finish_write([k])[k] == L1Error.KEY_IN_WRONG_STATE
    assert addr in adapter.freed  # aborted back to the owner free list


# =========================================================================
# delete / clear / misc
# =========================================================================


def test_delete_locally_staged_key_is_locked():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])

    assert manager.delete([k])[k] == L1Error.KEY_IS_LOCKED
    assert object_key_to_string(k) in handler.store_map  # server untouched


def test_delete_remotely_pinned_key_is_locked():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.pins[object_key_to_string(k)] = 1  # pinned by another instance

    assert manager.delete([k])[k] == L1Error.KEY_IS_LOCKED


def test_delete_absent_and_present_keys():
    manager, handler, _ = make_maru_manager()
    present, absent = _key(1), _key(2)
    _seed(handler, present)

    res = manager.delete([present, absent])
    assert res[present] == L1Error.SUCCESS
    assert res[absent] == L1Error.KEY_NOT_EXIST
    assert object_key_to_string(present) not in handler.store_map


def test_clear_non_force_preserves_staging():
    manager, handler, adapter = make_maru_manager()
    rk, wk = _key(1), _key(2)
    _seed(handler, rk)
    manager.reserve_read([rk])
    manager.reserve_write([wk], [False], _LAYOUT, mode="new")

    manager.clear()  # force=False keeps locked (staged) entries
    assert handler.pins[object_key_to_string(rk)] == 1
    assert not adapter.freed
    assert manager.unsafe_read([rk])[rk][0] == L1Error.SUCCESS


def test_clear_force_balances_pins_and_reclaims_writes():
    manager, handler, adapter = make_maru_manager()
    rk, wk = _key(1), _key(2)
    _seed(handler, rk)
    manager.reserve_read([rk], extra_count=1)  # two pins
    manager.reserve_write([wk], [False], _LAYOUT, mode="new")

    manager.clear(force=True)
    assert handler.pins[object_key_to_string(rk)] == 0
    assert len(adapter.freed) == 1  # staged write page aborted
    assert manager.unsafe_read([rk])[rk] == (L1Error.KEY_NOT_EXIST, None)


# =========================================================================
# failure paths (pin balance + page-reclaim safety under RPC failures)
# =========================================================================


def test_reserve_read_pin_rpc_failure_is_a_miss():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.fail_pin = True
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)
    assert not handler.pins


def test_reserve_read_retrieve_rpc_failure_rolls_back_pins():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.fail_retrieve = True
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)
    assert handler.pins[object_key_to_string(k)] == 0


def test_reserve_read_all_none_retrieve_rolls_back_pins():
    """Real transport failure: batch_retrieve returns [None]*len, no raise."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.retrieve_none = True
    assert manager.reserve_read([k], extra_count=1)[k] == (L1Error.KEY_NOT_EXIST, None)
    assert handler.pins[object_key_to_string(k)] == 0  # both pins rolled back


def test_reserve_read_pool_miss_rolls_back_pins():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    adapter.resolve_none = True
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)
    assert handler.pins[object_key_to_string(k)] == 0


def test_reserve_read_mid_write_key_not_readable():
    manager, _, _ = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert manager.reserve_read([k])[k] == (L1Error.KEY_NOT_READABLE, None)
    assert manager.unsafe_read([k])[k] == (L1Error.KEY_NOT_READABLE, None)


def test_finish_write_handle_failure_reclaims_all_pages():
    manager, _, adapter = make_maru_manager()
    keys = [_key(1), _key(2)]
    adapter.fail_handle = True
    manager.reserve_write(keys, [False] * 2, _LAYOUT, mode="new")
    res = manager.finish_write(keys)
    assert all(err == L1Error.KEY_IN_WRONG_STATE for err in res.values())
    assert len(adapter.freed) == 2  # never reached the server: reclaimed


def test_finish_write_store_rpc_failure_never_recycles():
    """Unknown server state: pages must leak rather than be reused."""
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    handler.fail_store = True
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert manager.finish_write([k])[k] == L1Error.KEY_IN_WRONG_STATE
    assert not adapter.freed


def test_finish_write_dup_skip_is_success_without_abort():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    _seed(handler, k)  # another instance registered it meanwhile
    assert manager.finish_write([k])[k] == L1Error.SUCCESS  # dup-skip
    assert not adapter.freed  # no local double-free


def test_reserve_write_exists_rpc_failure_proceeds():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    handler.fail_exists = True
    res = manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert res[k][0] == L1Error.SUCCESS  # allocated; dup-skip at store time


def test_delete_rpc_failure_reports_locked():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    handler.fail_delete = True
    assert manager.delete([k])[k] == L1Error.KEY_IS_LOCKED  # retryable


def test_get_memory_usage_before_init_reports_pool_size():
    manager, _, _ = make_maru_manager()
    manager._allocator._cxl_adapter = None  # back to pre-init state
    used, total = manager.get_memory_usage()
    assert (used, total) == (0, 1 << 20)


def test_get_memory_usage_watermark_tracks_auto_expand():
    """total anchors to device fill when auto_expand is on, to the owned pool
    when off -- so a hard-capped pool evicts before it is exhausted instead of
    OOMing while the device still has (unusable) free space."""
    chunk = 64
    own_pool = 16 * chunk  # FakeMaruHandler.get_stats total_pool_size
    free = 100 * chunk

    # auto_expand on (default): total = owned pool + device free.
    mgr_on, handler_on, _ = make_maru_manager(chunk_size=chunk)
    handler_on.cxl_free = free
    _seed(handler_on, _key(1))  # one allocated page
    assert mgr_on.get_memory_usage() == (chunk, own_pool + free)

    # auto_expand off: device free is ignored; total is the owned pool alone.
    mgr_off, handler_off, _ = make_maru_manager(chunk_size=chunk, auto_expand=False)
    handler_off.cxl_free = free
    _seed(handler_off, _key(1))
    assert mgr_off.get_memory_usage() == (chunk, own_pool)


def test_is_key_evictable_tracks_local_staging():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    assert manager.is_key_evictable(k)
    manager.reserve_read([k])
    assert not manager.is_key_evictable(k)
    manager.finish_read([k])
    assert manager.is_key_evictable(k)


# =========================================================================
# finish_write_and_reserve_read (C5): L2->L1 promote
# =========================================================================


def test_temporary_promote_stages_read_without_registering():
    """Default prefetch: private staging -- no batch_store, no server pin."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    page = manager.reserve_write([k], [True], _LAYOUT, mode="new")[k][1]

    err, obj = manager.finish_write_and_reserve_read([k])[k]
    assert err == L1Error.SUCCESS and obj is page
    assert object_key_to_string(k) not in handler.store_map  # never registered
    assert not handler.pins  # temporary reads hold no pin
    assert manager.unsafe_read([k])[k] == (L1Error.SUCCESS, page)


def test_temporary_promote_page_reclaimed_after_read():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    page = manager.reserve_write([k], [True], _LAYOUT, mode="new")[k][1]
    manager.finish_write_and_reserve_read([k])

    assert manager.finish_read([k])[k] == L1Error.SUCCESS
    assert page.metadata.address in adapter.freed  # local page reclaimed
    assert not handler.unpin_log  # nothing was pinned
    assert manager.unsafe_read([k])[k] == (L1Error.KEY_NOT_EXIST, None)


def test_retained_promote_registers_and_pins():
    """retain policy: batch_store + authoritative re-resolve with pins."""
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")

    assert manager.finish_write_and_reserve_read([k])[k][0] == L1Error.SUCCESS
    assert ks in handler.store_map  # registered in the shared directory
    assert handler.pins[ks] == 1  # read hold pinned

    assert manager.finish_read([k])[k] == L1Error.SUCCESS
    assert handler.pins[ks] == 0  # unpinned, not freed
    assert not adapter.freed
    assert ks in handler.store_map  # retained page survives the read


def test_retained_promote_extra_count_pins_n():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")

    manager.finish_write_and_reserve_read([k], extra_count=2)
    assert handler.pins[object_key_to_string(k)] == 3  # 1 + extra_count


def test_retained_promote_dup_skip_resolves_winner():
    """A peer registered the key first: re-resolve pins the winning page."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    _seed(handler, k, rid=9, pid=5)  # peer registered it meanwhile

    assert manager.finish_write_and_reserve_read([k])[k][0] == L1Error.SUCCESS
    assert handler.pins[object_key_to_string(k)] == 1


def test_promote_unstaged_key_is_not_exist():
    manager, _, _ = make_maru_manager()
    assert manager.finish_write_and_reserve_read([_key(1)])[_key(1)] == (
        L1Error.KEY_NOT_EXIST,
        None,
    )


def test_promote_already_read_staged_is_wrong_state():
    """The both-staged race the read-staged guard defends against."""
    manager, _, adapter = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    rpage = adapter.batched_allocate(_LAYOUT.shapes, _LAYOUT.dtypes, 1)[0]
    manager._pending_read[k] = _PendingRead(mem_obj=rpage, refcount=1)

    assert manager.finish_write_and_reserve_read([k])[k] == (
        L1Error.KEY_IN_WRONG_STATE,
        None,
    )
    assert k in manager._pending_write  # guard returned before popping


def test_retained_promote_store_failure_is_wrong_state():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    handler.fail_store_keys.add(object_key_to_string(k))
    manager.reserve_write([k], [False], _LAYOUT, mode="new")

    assert manager.finish_write_and_reserve_read([k])[k] == (
        L1Error.KEY_IN_WRONG_STATE,
        None,
    )
    assert k not in manager._pending_write  # write staging drained


def test_promote_fires_promote_event_not_write_finished():
    """Anti-#2744: promote must never fire write_finished (would re-store L2)."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.finish_write_and_reserve_read([k])
    assert rec.kinds("finish_write_and_reserve_read") == [[k]]
    assert rec.kinds("write_finished") == []


def test_temporary_promote_fires_promote_event():
    manager, _, _ = make_maru_manager()
    k = _key(1)
    manager.reserve_write([k], [True], _LAYOUT, mode="new")
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.finish_write_and_reserve_read([k])
    assert rec.kinds("finish_write_and_reserve_read") == [[k]]
    assert rec.kinds("write_finished") == []


# =========================================================================
# TTL sweeper (C6): reclaim orphan write pages / read pins
# =========================================================================


def test_sweeper_thread_starts_and_stops():
    manager, _, _ = make_maru_manager()
    assert manager._sweeper.is_alive()
    manager.close()
    manager._sweeper.join(timeout=2)
    assert not manager._sweeper.is_alive()


def test_sweep_reclaims_expired_write():
    manager, _, adapter = make_maru_manager()
    k = _key(1)
    page = manager.reserve_write([k], [False], _LAYOUT, mode="new")[k][1]
    manager._pending_write[k].deadline = time.monotonic() - 1  # force expiry

    manager._sweep_once()
    assert k not in manager._pending_write
    assert page.metadata.address in adapter.freed  # returned to the owner


def test_sweep_reclaims_expired_read_unpins():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    ks = object_key_to_string(k)
    _seed(handler, k)
    manager.reserve_read([k], extra_count=1)  # two pins
    manager._pending_read[k].deadline = time.monotonic() - 1

    manager._sweep_once()
    assert k not in manager._pending_read
    assert handler.pins[ks] == 0  # both pins released
    assert not adapter.freed  # registered page is not freed


def test_sweep_reclaims_expired_temporary_read():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    page = manager.reserve_write([k], [True], _LAYOUT, mode="new")[k][1]
    manager.finish_write_and_reserve_read([k])  # temporary read staged
    manager._pending_read[k].deadline = time.monotonic() - 1

    manager._sweep_once()
    assert k not in manager._pending_read
    assert page.metadata.address in adapter.freed  # private page reclaimed
    assert not handler.unpin_log  # temporary reads hold no pin


def test_sweep_leaves_live_staging():
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])  # deadline ~ now + read_ttl (300s)

    manager._sweep_once()
    assert k in manager._pending_read  # not expired
    assert handler.pins[object_key_to_string(k)] == 1
    assert not adapter.freed


def test_overlapping_reserve_refreshes_read_deadline():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])
    manager._pending_read[k].deadline = time.monotonic() - 1  # go stale

    manager.reserve_read([k])  # overlap -> refresh
    assert manager._pending_read[k].deadline > time.monotonic()


def test_finish_read_after_sweep_is_not_exist():
    """A late finish after a sweep behaves like a stock TTL expiry."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])
    manager._pending_read[k].deadline = time.monotonic() - 1
    manager._sweep_once()

    assert manager.finish_read([k])[k] == L1Error.KEY_NOT_EXIST


# =========================================================================
# event_bus observability parity (C11a): mirrors stock L1Manager publishes
# =========================================================================


def _keys_for(bus, event_type):
    """Return the ``keys`` metadata of every publish of ``event_type``."""
    return [
        c.args[0].metadata["keys"]
        for c in bus.publish.call_args_list
        if c.args[0].event_type == event_type
    ]


def test_event_bus_write_read_delete_lifecycle():
    manager, handler, _ = make_maru_manager()
    manager._event_bus = MagicMock()
    k = _key(1)

    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    manager.finish_write([k])
    manager.reserve_read([k])
    manager.finish_read([k])
    manager.delete([k])

    assert _keys_for(manager._event_bus, EventType.L1_WRITE_RESERVED) == [[k]]
    assert _keys_for(manager._event_bus, EventType.L1_WRITE_FINISHED) == [[k]]
    assert _keys_for(manager._event_bus, EventType.L1_READ_RESERVED) == [[k]]
    assert _keys_for(manager._event_bus, EventType.L1_READ_FINISHED) == [[k]]
    assert [k] in _keys_for(manager._event_bus, EventType.L1_KEYS_EVICTED)  # delete


def test_event_bus_promote_publishes_promote_event_not_write_finished():
    manager, _, _ = make_maru_manager()
    manager._event_bus = MagicMock()
    k = _key(1)
    manager.reserve_write([k], [False], _LAYOUT, mode="new")

    manager.finish_write_and_reserve_read([k])
    assert _keys_for(
        manager._event_bus, EventType.L1_WRITE_FINISHED_AND_READ_RESERVED
    ) == [[k]]
    # anti re-store at the event-bus level too.
    assert _keys_for(manager._event_bus, EventType.L1_WRITE_FINISHED) == []


def test_event_bus_temporary_finish_read_publishes_evicted():
    manager, _, _ = make_maru_manager()
    manager._event_bus = MagicMock()
    k = _key(1)
    manager.reserve_write([k], [True], _LAYOUT, mode="new")
    manager.finish_write_and_reserve_read([k])  # temporary read staged

    manager.finish_read([k])
    assert _keys_for(manager._event_bus, EventType.L1_READ_FINISHED) == [[k]]
    assert [k] in _keys_for(manager._event_bus, EventType.L1_KEYS_EVICTED)


def test_event_bus_touch_keys_does_not_publish():
    manager, _, _ = make_maru_manager()
    manager._event_bus = MagicMock()

    manager.touch_keys([_key(1)])
    manager._event_bus.publish.assert_not_called()


def test_report_status_keys():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])

    status = manager.report_status()
    assert status["backend"] == "maru"
    assert status["read_locked_count"] == 1
    assert status["is_healthy"] is True
    assert status["memory_total_bytes"] > 0


# =========================================================================
# listener firing (C4): feeds the eviction LRU + store controller
# =========================================================================


def test_reserve_read_fires_only_for_hits():
    """A miss must not be reported as a reserved-read hold."""
    manager, handler, _ = make_maru_manager()
    hit, miss = _key(1), _key(2)
    _seed(handler, hit)
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.reserve_read([hit, miss])
    assert rec.kinds("reserved_read") == [[hit]]


def test_reserve_and_finish_write_fire_with_registered_keys():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    assert rec.kinds("reserved_write") == [[k]]
    manager.finish_write([k])
    assert rec.kinds("write_finished") == [[k]]


def test_finish_write_store_failure_is_not_fired():
    """A page that never registered must not notify write-finished."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    handler.fail_store_keys.add(object_key_to_string(k))
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.reserve_write([k], [False], _LAYOUT, mode="new")
    manager.finish_write([k])
    assert rec.kinds("write_finished") == [[]]  # fired once, empty


def test_finish_read_normal_fires_read_finished_without_delete():
    """A directory-backed read unpins; it is never a manager delete."""
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    manager.reserve_read([k])
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.finish_read([k])
    assert rec.kinds("read_finished") == [[k]]
    assert all(ks == [] for ks in rec.kinds("deleted_by_manager"))


def test_finish_read_temporary_frees_page_and_fires_delete():
    """A temporary read (local staging) reclaims its page at refcount zero.

    The promote path that creates temporary reads lands in C5; here the entry
    is staged white-box to exercise C4's refcount-zero reclaim branch.
    """
    manager, handler, adapter = make_maru_manager()
    k = _key(1)
    page = adapter.batched_allocate(_LAYOUT.shapes, _LAYOUT.dtypes, 1)[0]
    manager._pending_read[k] = _PendingRead(mem_obj=page, refcount=1, is_temporary=True)
    rec = RecordingListener()
    manager.register_listener(rec)

    assert manager.finish_read([k])[k] == L1Error.SUCCESS
    # Temporary pages hold no server pin -- reclaimed, not unpinned.
    assert page.metadata.address in adapter.freed
    assert not handler.unpin_log
    assert rec.kinds("read_finished") == [[k]]
    assert rec.kinds("deleted_by_manager") == [[k]]
    assert k not in manager._pending_read


def test_delete_fires_only_for_removed_keys():
    """Locked / absent keys are not reported as manager deletes."""
    manager, handler, _ = make_maru_manager()
    removed, absent, locked = _key(1), _key(2), _key(3)
    _seed(handler, removed)
    _seed(handler, locked)
    handler.pins[object_key_to_string(locked)] = 1  # pinned elsewhere
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.delete([removed, absent, locked])
    assert rec.kinds("deleted_by_manager") == [[removed]]


def test_touch_keys_fires_accessed():
    manager, _, _ = make_maru_manager()
    rec = RecordingListener()
    manager.register_listener(rec)

    keys = [_key(1), _key(2)]
    manager.touch_keys(keys)
    assert rec.kinds("accessed") == [keys]


def test_clear_force_fires_deleted_for_dropped_staging():
    manager, handler, _ = make_maru_manager()
    rk, wk = _key(1), _key(2)
    _seed(handler, rk)
    manager.reserve_read([rk])
    manager.reserve_write([wk], [False], _LAYOUT, mode="new")
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.clear(force=True)
    fired = rec.kinds("deleted_by_manager")
    assert len(fired) == 1
    assert set(fired[0]) == {rk, wk}


def test_clear_non_force_fires_nothing():
    manager, handler, _ = make_maru_manager()
    rk = _key(1)
    _seed(handler, rk)
    manager.reserve_read([rk])
    rec = RecordingListener()
    manager.register_listener(rec)

    manager.clear()  # keeps locked staging -> no drops
    assert rec.events == []

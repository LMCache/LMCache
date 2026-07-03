# SPDX-License-Identifier: Apache-2.0

"""Tests for MaruL1Manager (fake maru runtime, no CXL required)."""

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
        object_key_to_string,
    )

    # Local
    from .maru_fakes import make_maru_manager
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


def test_is_key_evictable_tracks_local_staging():
    manager, handler, _ = make_maru_manager()
    k = _key(1)
    _seed(handler, k)
    assert manager.is_key_evictable(k)
    manager.reserve_read([k])
    assert not manager.is_key_evictable(k)
    manager.finish_read([k])
    assert manager.is_key_evictable(k)


def test_promote_not_implemented_yet():
    manager, _, _ = make_maru_manager()
    with pytest.raises(NotImplementedError):
        manager.finish_write_and_reserve_read([_key(1)])


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

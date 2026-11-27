# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for PR #1: L1 named read leases & untimed active reads."""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
from lmcache.v1.distributed.storage_manager import StorageManager


def _make_key(idx: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=idx.to_bytes(8, "big"),
        model_name="test_model",
        kv_rank=0,
    )


def _make_l1(read_ttl: int = 5) -> DramL1Manager:
    cfg = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1024 * 1024, use_lazy=False, shm_name=""
        ),
        read_ttl_seconds=read_ttl,
        write_ttl_seconds=5,
    )
    return DramL1Manager(cfg)


def _seed(l1: DramL1Manager, keys: list[ObjectKey], temp: bool = False) -> None:
    layout = MemoryLayoutDesc(shapes=[torch.Size([16])], dtypes=[torch.uint8])
    res = l1.reserve_write(keys, [temp] * len(keys), layout, "seed")
    assert all(err == L1Error.SUCCESS for err, _ in res.values())
    l1.finish_write(keys, "seed")


def test_named_leases_isolate_expired_and_fresh_owners() -> None:
    """Expired/released owner A cannot steal fresh owner B's named lease."""
    l1 = _make_l1(read_ttl=1)
    try:
        k = _make_key(1)
        _seed(l1, [k])

        assert (
            l1.reserve_read([k], read_locks=1, read_owner="owner-A")[k][0]
            == L1Error.SUCCESS
        )
        time.sleep(1.1)
        # Owner A's lease expired; owner B takes a fresh lease.
        assert (
            l1.reserve_read([k], read_locks=1, read_owner="owner-B")[k][0]
            == L1Error.SUCCESS
        )
        assert not l1.is_key_evictable(k)

        # Owner A fails to acquire and releasing owner A never disturbs owner B.
        res_a, h_a = l1.acquire_read([k], read_owner="owner-A", consumer_id=0)
        assert res_a[k][0] == L1Error.KEY_NOT_READABLE
        assert h_a == []
        assert (
            l1.finish_read([k], read_owner="owner-A")[k] == L1Error.KEY_IN_WRONG_STATE
        )
        assert not l1.is_key_evictable(k)

        # Owner B still acquires cleanly.
        res_b, h_b = l1.acquire_read([k], read_owner="owner-B", consumer_id=0)
        assert res_b[k][0] == L1Error.SUCCESS
        assert len(h_b) == 1
        l1.finish_active_reads(h_b)
        assert l1.is_key_evictable(k)
    finally:
        l1.close()


def test_active_read_outlives_ttl_and_blocks_eviction_delete_clear_close() -> None:
    """Active read pins buffer past read_ttl and blocks force delete/clear/close."""
    l1 = _make_l1(read_ttl=1)
    try:
        k = _make_key(2)
        _seed(l1, [k], temp=True)
        l1.reserve_read([k], read_locks=1, read_owner="owner-1")
        res, handles = l1.acquire_read([k], read_owner="owner-1", consumer_id=0)
        assert res[k][0] == L1Error.SUCCESS
        assert len(handles) == 1

        # Wait past read TTL: still pinned!
        time.sleep(1.1)
        assert not l1.is_key_evictable(k)
        assert l1.delete([k], force=False)[k] == L1Error.KEY_IS_LOCKED
        assert l1.delete([k], force=True)[k] == L1Error.KEY_IS_LOCKED
        l1.clear(force=True)
        assert l1.get_object_state(k) is not None
        with pytest.raises(RuntimeError, match="active read"):
            l1.close()

        # Releasing active read unpins and frees temporary object.
        l1.finish_active_reads(handles)
        assert l1.get_object_state(k) is None
    finally:
        l1.close()


def test_acquire_read_is_atomic_and_deduplicates_keys_and_consumers() -> None:
    """Partial miss rolls back cleanly; duplicates collapse; consumer claimed once."""
    l1 = _make_l1(read_ttl=5)
    try:
        k1, k2 = _make_key(10), _make_key(11)
        _seed(l1, [k1])

        l1.reserve_read([k1], read_locks=2, read_owner="owner-tp")
        # k2 missing -> k1's lease MUST NOT be decremented!
        res_miss, h_miss = l1.acquire_read(
            [k1, k2], read_owner="owner-tp", consumer_id=0
        )
        assert res_miss[k2][0] == L1Error.KEY_NOT_EXIST
        assert h_miss == []
        assert l1.get_object_state(k1).read_leases["owner-tp"].remaining == 2  # type: ignore[union-attr]

        # Duplicate key in one call collapses cleanly to 1 handle.
        res0, h0 = l1.acquire_read([k1, k1], read_owner="owner-tp", consumer_id=0)
        assert res0[k1][0] == L1Error.SUCCESS
        assert len(h0) == 1
        # Same consumer_id=0 cannot claim a second share.
        res_dup, h_dup = l1.acquire_read([k1], read_owner="owner-tp", consumer_id=0)
        assert res_dup[k1][0] == L1Error.KEY_NOT_READABLE
        assert h_dup == []
        # Second worker consumer_id=1 claims the remaining share.
        res1, h1 = l1.acquire_read([k1], read_owner="owner-tp", consumer_id=1)
        assert res1[k1][0] == L1Error.SUCCESS
        assert len(h1) == 1
        assert "owner-tp" not in l1.get_object_state(k1).read_leases  # type: ignore[union-attr]

        l1.finish_active_reads(h0)
        assert not l1.is_key_evictable(k1)
        l1.finish_active_reads(h1)
        assert l1.is_key_evictable(k1)
    finally:
        l1.close()


def test_storage_manager_prefetch_and_acquire_read_end_to_end() -> None:
    """StorageManager prefetch -> acquire_read -> finish_active_reads."""
    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1024 * 1024, use_lazy=False, shm_name=""
            ),
            read_ttl_seconds=5,
            write_ttl_seconds=5,
        ),
        eviction_config=EvictionConfig("noop"),
    )
    sm = StorageManager(cfg)
    try:
        l1 = sm.l1_managers()[0]
        layout = MemoryLayoutDesc(shapes=[torch.Size([16])], dtypes=[torch.uint8])
        k1, k2 = _make_key(20), _make_key(21)
        res = l1.reserve_write([k1, k2], [False, False], layout, "w")
        for _, obj in res.values():
            assert obj is not None
            obj.tensor.fill_(42)
        l1.finish_write([k1, k2], "w")

        spec = PrefetchTaskSpec(
            key_groups=[
                GroupedObjectKeys(
                    object_group_id=0,
                    keys=[k1, k2],
                    layout_desc=layout,
                    num_kv_readers=1,
                )
            ],
            allowed_l2_memory_tags=frozenset(),
            read_owner="req-1:lkp-1",
        )
        handle = sm.submit_prefetch_task(spec, external_request_id="req-1")
        assert sm.wait_prefetch_status(handle, timeout=2.0) is True
        pf_res = sm.query_prefetch_status(handle)
        assert pf_res is not None
        assert pf_res.l1_owners == {k1: l1.l1_manager_id, k2: l1.l1_manager_id}

        # Wrong owner fails cleanly without disturbing req-1:lkp-1.
        objs_bad, comp_bad = sm.acquire_read(
            [k1, k2], read_owner="other", consumer_id=0, l1_owners=pf_res.l1_owners
        )
        assert objs_bad is None and comp_bad is None

        # Correct owner promotes both keys into active handles.
        objs, comp = sm.acquire_read(
            [k1, k2],
            read_owner="req-1:lkp-1",
            consumer_id=0,
            l1_owners=pf_res.l1_owners,
        )
        assert objs is not None and len(objs) == 2
        assert comp is not None and not comp.is_empty
        assert all(not l1.is_key_evictable(k) for k in (k1, k2))

        sm.finish_active_reads(comp)
        assert all(l1.is_key_evictable(k) for k in (k1, k2))
    finally:
        sm.close()

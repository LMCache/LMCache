# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat
from lmcache.v1.memory_pool import MemoryPool, PoolRequest, lease


def _make_pool():
    allocator = AdHocMemoryAllocator(device="cpu")
    pool = MemoryPool(allocator)
    return pool, allocator


def _sample_request(tag: str = "test") -> PoolRequest:
    return PoolRequest(
        shape=torch.Size([1, 1, 1, 1]),
        dtype=torch.float32,
        fmt=MemoryFormat.KV_2LTD,
        tag=tag,
    )


def test_borrow_release_updates_stats_and_parent():
    pool, allocator = _make_pool()
    req = _sample_request()
    memory_obj = pool.borrow(req)

    stats = pool.stats()
    assert stats["live_leases"] == 1
    assert stats["borrowed_bytes"] == memory_obj.meta.phy_size
    assert memory_obj.parent_allocator is pool

    pool.release(memory_obj)

    stats = pool.stats()
    assert stats["live_leases"] == 0
    assert stats["borrowed_bytes"] == 0
    assert memory_obj.parent_allocator is allocator


def test_lease_scope_releases_on_exception():
    pool, _ = _make_pool()
    req = _sample_request(tag="lease_exception")

    with pytest.raises(RuntimeError):
        with lease(pool, req):
            raise RuntimeError("boom")

    stats = pool.stats()
    assert stats["live_leases"] == 0
    assert stats["borrowed_bytes"] == 0


def test_ref_count_down_triggers_pool_release():
    pool, _ = _make_pool()
    req = _sample_request(tag="refcount")

    memory_obj = pool.borrow(req)
    assert pool.stats()["live_leases"] == 1

    memory_obj.ref_count_down()

    stats = pool.stats()
    assert stats["live_leases"] == 0
    assert stats["borrowed_bytes"] == 0


class _BackendStub:
    def __init__(self, allocator):
        self.allocator = allocator
        self.calls = []

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        self.calls.append((shape, dtype, fmt, eviction, busy_loop))
        return self.allocator.allocate(shape, dtype, fmt)


def test_backend_fallback_preserves_allocator_type():
    allocator = AdHocMemoryAllocator(device="cpu")
    backend = _BackendStub(allocator)
    pool = MemoryPool(allocator, backend=backend)
    req = _sample_request(tag="backend_fallback")

    memory_obj = pool.borrow(req)
    assert memory_obj is not None
    assert backend.calls, "expected backend.allocate to be used in fallback path"

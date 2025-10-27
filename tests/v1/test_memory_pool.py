# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1 import memory_pool as memory_pool_module
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat
from lmcache.v1.memory_pool import MemoryPool, PoolRequest
from lmcache.v1.storage_backend.local_cpu_backend import (
    LocalCPUBackend,
    LocalCPUDisabledError,
)
from tests.v1.utils import dumb_metadata


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
        with pool.lease(req):
            raise RuntimeError("boom")

    stats = pool.stats()
    assert stats["live_leases"] == 0
    assert stats["borrowed_bytes"] == 0


def test_lease_logs_exception(monkeypatch):
    pool, _ = _make_pool()
    req = _sample_request(tag="lease_log")

    captured: dict[str, str] = {}

    def fake_exception(msg: str, *args: object, **kwargs: object) -> None:
        captured["message"] = msg % args if args else msg

    monkeypatch.setattr(memory_pool_module.logger, "exception", fake_exception)

    with pytest.raises(RuntimeError):
        with pool.lease(req):
            raise RuntimeError("boom")

    assert "MemoryPool lease raised exception" in captured.get("message", "")


def test_ref_count_down_triggers_pool_release():
    pool, _ = _make_pool()
    req = _sample_request(tag="refcount")

    memory_obj = pool.borrow(req)
    assert pool.stats()["live_leases"] == 1

    memory_obj.ref_count_down()

    stats = pool.stats()
    assert stats["live_leases"] == 0
    assert stats["borrowed_bytes"] == 0


class _BorrowHook:
    def __init__(self, allocator):
        self.allocator = allocator
        self.calls = []

    def __call__(self, req: PoolRequest):
        shape = req.resolve_shape()
        assert req.dtype is not None
        self.calls.append((shape, req.dtype, req.fmt, req.eviction, req.busy_loop))
        return self.allocator.allocate(shape, req.dtype, req.fmt)


def test_backend_fallback_preserves_allocator_type():
    allocator = AdHocMemoryAllocator(device="cpu")
    hook = _BorrowHook(allocator)
    pool = MemoryPool(allocator, borrow_hook=hook)
    req = _sample_request(tag="backend_fallback")

    memory_obj = pool.borrow(req)
    assert memory_obj is not None
    assert hook.calls, "expected borrow hook to be invoked"


def test_local_cpu_backend_zero_size_disables():
    config = LMCacheEngineConfig.from_defaults()
    config.max_local_cpu_size = 0.0
    config.local_cpu = False
    config.enable_pd = True
    metadata = dumb_metadata()

    with pytest.raises(LocalCPUDisabledError):
        LocalCPUBackend(config, metadata, dst_device="cpu")

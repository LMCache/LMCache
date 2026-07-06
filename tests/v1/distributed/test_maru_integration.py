# SPDX-License-Identifier: Apache-2.0
"""C11b: maru control integration through the stock tiering controllers.

A real StorageManager + StoreController/PrefetchController/EvictionController
drive MaruL1Manager, whose CXL pool + MaruServer directory are the in-memory
fakes from the manager-level tests, plus a mock L2 adapter. These assert maru's
*control* integration (register / read-reserve / evict-delete under the real
controllers). L1<->L2 byte movement is stock controller logic (identical for
any L1 backend) and is covered by the stock StorageManager tests, so it is not
re-asserted here (the fakes back pages with no real memory).
"""

# Standard
import time

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        MaruL1Config,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
    from lmcache.v1.distributed.maru_l1_manager import (
        MaruL1Manager,
        object_key_to_string,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager

    # Local
    from .maru_fakes import FakeCxlAdapter, FakeMaruHandler
except ImportError:
    pytest.skip("maru integration deps unavailable", allow_module_level=True)

_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([4, 8])], dtypes=[torch.float16])


def _key(idx: int) -> ObjectKey:
    return ObjectKey(chunk_hash=idx.to_bytes(4, "big"), model_name="m", kv_rank=0)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _maru_sm_with_fakes(chunk_size: int = 64, trigger_watermark: float = 0.8):
    """A real StorageManager whose maru L1 tier is backed by the fakes."""
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=0,
                use_lazy=False,
                maru_config=MaruL1Config(
                    server_url="maru://localhost:5555",
                    pool_size_bytes=1 << 20,
                    instance_id="t",
                ),
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(
            eviction_policy="LRU", trigger_watermark=trigger_watermark
        ),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=1.0)]
        ),
    )
    sm = StorageManager(config)
    assert isinstance(sm._l1_manager, MaruL1Manager)  # harness selected maru
    handler = FakeMaruHandler(chunk_size)
    adapter = FakeCxlAdapter(chunk_size)
    alloc = sm._l1_manager._allocator
    alloc._handler = handler
    alloc._cxl_adapter = adapter
    alloc._single_token_size = 16
    return sm, handler, adapter


def test_store_registers_in_maru_directory():
    """reserve_write -> finish_write through the full stack registers in maru."""
    sm, handler, _ = _maru_sm_with_fakes()
    try:
        k = _key(1)
        res = sm.reserve_write([k], _LAYOUT, mode="new")
        assert res[k] is not None
        sm.finish_write([k])
        assert object_key_to_string(k) in handler.store_map
    finally:
        sm.close()


def test_prefetch_hits_l1_resident_keys():
    """Prefetch of directory-resident keys is a full L1 hit (maru reserve_read)."""
    sm, _, _ = _maru_sm_with_fakes()
    try:
        keys = [_key(i) for i in range(3)]
        sm.reserve_write(keys, _LAYOUT, mode="new")
        sm.finish_write(keys)

        handle = sm.submit_prefetch_task(keys, _LAYOUT)
        assert _wait(lambda: sm.query_prefetch_status(handle) is not None)
        assert sm.query_prefetch_status(handle).count_leading_ones() == len(keys)
    finally:
        sm.close()


def test_eviction_deletes_from_maru_directory():
    """Watermark eviction drives MaruL1Manager.delete on the shared directory."""
    # Low watermark: the fake pool is 16 pages, so a handful of stored keys
    # crosses it and the eviction controller must reclaim some.
    sm, handler, _ = _maru_sm_with_fakes(trigger_watermark=0.1)
    try:
        keys = [_key(i) for i in range(6)]
        sm.reserve_write(keys, _LAYOUT, mode="new")
        sm.finish_write(keys)
        assert len(handler.store_map) == 6

        # The eviction controller runs on its own thread; it deletes evictable
        # keys from the maru directory until usage falls under the watermark.
        assert _wait(lambda: len(handler.store_map) < 6)
    finally:
        sm.close()

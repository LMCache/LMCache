# SPDX-License-Identifier: Apache-2.0
"""A streaming reservation is excluded from hot-cache capacity and eviction."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
from lmcache.v1.multiprocess.streaming import FSStreamingReader
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator


@pytest.mark.parametrize("l1_bytes", [8192, 12288, 16384])
def test_streaming_reservation_leaves_one_aligned_hot_allocation(
    monkeypatch: pytest.MonkeyPatch, l1_bytes: int
) -> None:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    owner = DramL1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=l1_bytes, use_lazy=False, align_bytes=8192, shm_name=""
            )
        )
    )
    try:
        if l1_bytes < 16384:
            with pytest.raises(ValueError, match="leave space for hot L1"):
                owner.reserve_streaming_arena(4096)
            assert owner.get_memory_usage() == (0, l1_bytes)
        else:
            arena = owner.reserve_streaming_arena(4096)
            try:
                assert owner.get_memory_usage() == (0, 8192)
            finally:
                owner.release_streaming_arena(arena)
            assert owner.get_memory_usage() == (0, l1_bytes)
    finally:
        owner.close()


def test_reserved_window_never_becomes_hot_capacity(monkeypatch):
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    owner = DramL1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=16384, use_lazy=False, align_bytes=4096, shm_name=""
            )
        )
    )
    reader = FSStreamingReader(MagicMock(), 8192, shared_owner=owner)
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    keys = [
        ObjectKey(chunk_hash=bytes([i]) * 32, model_name="m", kv_rank=0)
        for i in (1, 2, 3)
    ]
    try:
        initial = owner.reserve_write(keys[:2], [False, False], layout, "hot")
        assert all(initial[k][0] == L1Error.SUCCESS for k in keys[:2])
        for key in keys[:2]:
            initial[key][1].raw_tensor.fill_(key.chunk_hash[0])
        owner.finish_write(keys[:2], tag="hot")
        before = owner.get_memory_usage()
        w = reader.pool.acquire(layout)
        w.memory_obj.raw_tensor.fill_(42)
        reader.pool.release(w)
        assert owner.get_memory_usage() == before == (8192, 8192)
        assert (
            owner.reserve_write(keys[2:], [False], layout, "hot")[keys[2]][0]
            == L1Error.OUT_OF_MEMORY
        )
        for key in keys[:2]:
            e, obj = owner.reserve_read([key])[key]
            assert e == L1Error.SUCCESS
            assert torch.all(obj.raw_tensor == key.chunk_hash[0])
            owner.finish_read([key])
    finally:
        reader.close()
        owner.close()


def test_reserved_bytes_do_not_trigger_hot_lru_watermark(monkeypatch):
    # Standard
    import time

    # First Party
    from lmcache.v1.distributed.config import EvictionConfig
    from lmcache.v1.distributed.storage_controllers.eviction_controller import (
        L1EvictionController,
    )

    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda n, *a, **kw: torch.empty(n, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **kw: None)
    owner = DramL1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=16384, use_lazy=False, align_bytes=4096, shm_name=""
            )
        )
    )
    reader = FSStreamingReader(MagicMock(), 8192, shared_owner=owner)
    controller = L1EvictionController(
        owner,
        EvictionConfig(
            eviction_policy="LRU", trigger_watermark=0.6, eviction_ratio=1.0
        ),
    )
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    key = ObjectKey(chunk_hash=b"hot", model_name="m", kv_rank=0)
    try:
        assert (
            owner.reserve_write([key], [False], layout, "hot")[key][0]
            == L1Error.SUCCESS
        )
        owner.finish_write([key], tag="hot")
        assert owner.get_memory_usage() == (4096, 8192)
        controller.start()
        time.sleep(1.2)
        assert owner.reserve_read([key])[key][0] == L1Error.SUCCESS
        owner.finish_read([key])
    finally:
        controller.stop()
        reader.close()
        owner.close()

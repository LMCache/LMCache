# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
import asyncio
import os
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)


def _has_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture
def loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, name="test-loop", daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_put_get_roundtrip(memory_allocator, loop_in_thread):
    """Test basic put/get roundtrip with RustRawBlockBackend."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            key = CacheEngineKey("test_model", 1, 0, 12345, torch.bfloat16)
            allocator = AdHocMemoryAllocator(device="cpu")
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            assert obj.tensor is not None
            obj.tensor.fill_(7)
            expected = bytes(obj.byte_array)

            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            assert isinstance(futs[0], Future)
            futs[0].result(timeout=10)

            out = backend.get_blocking(key)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_eviction_lru(memory_allocator, loop_in_thread):
    """Test LRU eviction when capacity is exceeded."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_evict",
        )
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.capacity_bytes": 2 * 4 * 1024 * 1024,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.slot_bytes": 4 * 1024 * 1024,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            alloc = AdHocMemoryAllocator(device="cpu")

            k1 = CacheEngineKey("test_model", 1, 0, 1, torch.bfloat16)
            k2 = CacheEngineKey("test_model", 1, 0, 2, torch.bfloat16)
            k3 = CacheEngineKey("test_model", 1, 0, 3, torch.bfloat16)

            o1 = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            o2 = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            o3 = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert o1 and o2 and o3
            assert (
                o1.tensor is not None
                and o2.tensor is not None
                and o3.tensor is not None
            )
            o1.tensor.fill_(1)
            o2.tensor.fill_(2)
            o3.tensor.fill_(3)

            f1 = backend.batched_submit_put_task([k1], [o1])[0]
            f2 = backend.batched_submit_put_task([k2], [o2])[0]
            f1.result(timeout=10)
            f2.result(timeout=10)

            # Touch k1 so k2 becomes LRU
            assert backend.get_blocking(k1) is not None

            f3 = backend.batched_submit_put_task([k3], [o3])[0]
            f3.result(timeout=10)

            # k2 should be evicted
            assert backend.contains(k2) is False
            assert backend.get_blocking(k2) is None
            assert backend.get_blocking(k1) is not None
            assert backend.get_blocking(k3) is not None
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_manifest_roundtrip(memory_allocator, loop_in_thread):
    """Test manifest persistence and restoration across backend restarts."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        manifest_path = os.path.join(td, "manifest.json")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        base_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_manifest",
        )
        base_cfg.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.manifest_path": manifest_path,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
        )

        local_cpu = LocalCPUBackend(
            config=base_cfg,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend1 = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        alloc = AdHocMemoryAllocator(device="cpu")
        k1 = CacheEngineKey("test_model", 1, 0, 111, torch.bfloat16)
        o1 = alloc.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert o1 and o1.tensor is not None
        o1.tensor.fill_(9)
        expected = bytes(o1.byte_array)
        try:
            fut = backend1.batched_submit_put_task([k1], [o1])[0]
            fut.result(timeout=10)
        finally:
            backend1.close()

        # New backend instance should restore index and retrieve
        backend2 = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        try:
            assert backend2.contains(k1)
            out = backend2.get_blocking(k1)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend2.close()

# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
from unittest.mock import MagicMock, patch
import asyncio
import os
import struct
import tempfile
import threading
import time

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
    _DEFAULT_META_MAGIC,
    _DEFAULT_META_VERSION,
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


def _run_batched_get_prefix_stop(
    memory_allocator,
    loop_in_thread,
    *,
    async_get: bool,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_batched_get",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
            allocator = AdHocMemoryAllocator(device="cpu")
            key1 = CacheEngineKey("test_model", 1, 0, 1001, torch.bfloat16)
            key_miss = CacheEngineKey("test_model", 1, 0, 1002, torch.bfloat16)
            key3 = CacheEngineKey("test_model", 1, 0, 1003, torch.bfloat16)

            obj1 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            obj3 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj1 is not None and obj1.tensor is not None
            assert obj3 is not None and obj3.tensor is not None
            obj1.tensor.fill_(1)
            obj3.tensor.fill_(3)
            expected1 = bytes(obj1.byte_array)
            expected3 = bytes(obj3.byte_array)

            for key, obj in ((key1, obj1), (key3, obj3)):
                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

            if async_get:
                future = asyncio.run_coroutine_threadsafe(
                    backend.batched_get_non_blocking(
                        "lookup-rawblock", [key1, key_miss, key3]
                    ),
                    loop_in_thread,
                )
                async_results = future.result(timeout=10)
                assert len(async_results) == 1
                assert bytes(async_results[0].byte_array) == expected1
                async_results[0].ref_count_down()
            else:
                blocking_results = backend.batched_get_blocking([key1, key_miss, key3])
                assert len(blocking_results) == 3
                assert blocking_results[0] is not None
                assert bytes(blocking_results[0].byte_array) == expected1
                assert blocking_results[1] is None
                assert blocking_results[2] is None
                blocking_results[0].ref_count_down()

            out3 = backend.get_blocking(key3)
            assert out3 is not None
            assert bytes(out3.byte_array) == expected3
            out3.ref_count_down()
        finally:
            backend.close()


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
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
def test_rust_raw_block_backend_batched_get_blocking_prefix_stop(
    memory_allocator, loop_in_thread
):
    """Batched blocking get should stop at the first miss and preserve order."""
    _run_batched_get_prefix_stop(memory_allocator, loop_in_thread, async_get=False)


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_non_blocking_prefix_stop(
    memory_allocator, loop_in_thread
):
    """Async batched get should return only the successful prefix."""
    _run_batched_get_prefix_stop(memory_allocator, loop_in_thread, async_get=True)


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_resets_inflight_on_rawdev_error(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_rawdev_error",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
            allocator = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 3001, torch.bfloat16)
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(31)
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()

            with patch.object(backend, "_rawdev", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    backend.get_blocking(key)
            assert backend._inflight_io_count == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_handles_allocator_exhaustion(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_allocator_exhaustion",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
            allocator = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 3002, torch.bfloat16)
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(32)
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()

            with patch.object(local_cpu, "allocate", return_value=None):
                assert backend.get_blocking(key) is None
                assert backend.batched_get_blocking([key]) == [None]
            assert backend._inflight_io_count == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_releases_allocation_on_read_error(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_release_failed_get",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
            allocator = AdHocMemoryAllocator(device="cpu")
            key = CacheEngineKey("test_model", 1, 0, 3003, torch.bfloat16)
            obj = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None and obj.tensor is not None
            obj.tensor.fill_(33)
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()

            leaked_obj = local_cpu.allocate(
                torch.Size([2, 16, 8, 128]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
            assert leaked_obj is not None
            assert leaked_obj.get_ref_count() == 1

            raw_dev = MagicMock()
            raw_dev.pread_into.side_effect = OSError("read failed")
            with patch.object(local_cpu, "allocate", return_value=leaked_obj):
                with patch.object(backend, "_rawdev", return_value=raw_dev):
                    with pytest.raises(OSError, match="read failed"):
                        backend.get_blocking(key)

            assert leaked_obj.get_ref_count() == 0
            assert backend._inflight_io_count == 0
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_releases_loaded_prefix_on_read_error(
    memory_allocator, loop_in_thread
):
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_release_prefix_get",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
            allocator = AdHocMemoryAllocator(device="cpu")
            key1 = CacheEngineKey("test_model", 1, 0, 3004, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 3005, torch.bfloat16)
            for key, fill in ((key1, 34), (key2, 35)):
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None and obj.tensor is not None
                obj.tensor.fill_(fill)
                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

            loaded_obj = local_cpu.allocate(
                torch.Size([2, 16, 8, 128]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
            failed_obj = local_cpu.allocate(
                torch.Size([2, 16, 8, 128]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
            assert loaded_obj is not None
            assert failed_obj is not None
            assert loaded_obj.get_ref_count() == 1
            assert failed_obj.get_ref_count() == 1

            raw_dev = MagicMock()
            raw_dev.pread_into.side_effect = [None, OSError("read failed")]
            with patch.object(
                local_cpu, "allocate", side_effect=[loaded_obj, failed_obj]
            ):
                with patch.object(backend, "_rawdev", return_value=raw_dev):
                    with pytest.raises(OSError, match="read failed"):
                        backend.batched_get_blocking([key1, key2])

            assert loaded_obj.get_ref_count() == 0
            assert failed_obj.get_ref_count() == 0
            assert backend._inflight_io_count == 0
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
            "rust_raw_block.capacity_bytes": 3 * 4 * 1024 * 1024,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.slot_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
def test_rust_raw_block_backend_device_checkpoint_roundtrip(
    memory_allocator, loop_in_thread
):
    """Test on-device metadata checkpoint persistence across backend restarts."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
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
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_data_offsets_start_after_metadata(
    memory_allocator, loop_in_thread
):
    """Slot allocations must begin after reserved metadata region."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_offsets",
        )
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 8 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
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
            key = CacheEngineKey("test_model", 1, 0, 777, torch.bfloat16)
            alloc = AdHocMemoryAllocator(device="cpu")
            obj = alloc.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj is not None
            futs = backend.batched_submit_put_task([key], [obj])
            assert futs is not None
            futs[0].result(timeout=10)

            with backend._lock:
                entry = backend._index.get(key)
                assert entry is not None
                assert entry.offset >= 8 * 1024 * 1024
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_ignores_torn_newer_checkpoint(
    memory_allocator, loop_in_thread
):
    """
    If a newer checkpoint copy is torn, loader falls back to the older valid copy.
    """
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        base_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_torn_checkpoint",
        )
        meta_total = 4 * 1024 * 1024
        align = 4096
        base_cfg.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": align,
            "rust_raw_block.header_bytes": align,
            "rust_raw_block.meta_total_bytes": meta_total,
            "rust_raw_block.meta_enable_periodic": False,
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
        key = CacheEngineKey("test_model", 1, 0, 888, torch.bfloat16)
        obj = alloc.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert obj is not None and obj.tensor is not None
        obj.tensor.fill_(11)
        expected = bytes(obj.byte_array)
        try:
            fut = backend1.batched_submit_put_task([key], [obj])[0]
            fut.result(timeout=10)
        finally:
            torn_offset = backend1._meta_container_offsets()[1]
            backend1.close()

        # Corrupt the newer checkpoint copy with invalid CRC.
        # Header format: <8sIQQI (magic, version, seq, payload_len, crc).
        header = struct.pack(
            "<8sIQQI", _DEFAULT_META_MAGIC, _DEFAULT_META_VERSION, 9999, 2, 0
        )
        padded_header = header + bytes(align - len(header))
        with open(dev_path, "r+b") as f:
            f.seek(torn_offset + align)
            f.write(b"{}")
            f.seek(torn_offset)
            f.write(padded_header)

        backend2 = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        try:
            assert backend2.contains(key)
            out = backend2.get_blocking(key)
            assert out is not None
            assert bytes(out.byte_array) == expected
        finally:
            backend2.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_skips_invalid_checkpoint_entries(
    memory_allocator, loop_in_thread
):
    """Checkpoint restore should reject invalid offset/size metadata entries."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        base_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_invalid_checkpoint",
        )
        base_cfg.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.meta_verify_on_load": False,
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
        backend = RustRawBlockBackend(
            config=base_cfg,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        try:
            entries = {}
            for chunk_hash, (offset, size) in {
                1: (backend._data_base_offset - backend.slot_bytes, 1024),
                2: (backend._data_base_offset + 1, 1024),
                3: (
                    backend._data_base_offset,
                    backend.slot_bytes - backend.header_bytes + 1,
                ),
            }.items():
                key = CacheEngineKey("test_model", 1, 0, chunk_hash, torch.bfloat16)
                entries[key.to_string()] = {
                    "offset": offset,
                    "size": size,
                    "shape": [2, 16, 8, 128],
                    "dtype": "bfloat16",
                    "fmt": MemoryFormat.KV_T2D.name,
                    "cached_positions": None,
                }

            applied = backend._apply_loaded_state(
                {
                    "version": 1,
                    "device_path": dev_path,
                    "capacity_bytes": backend.capacity_bytes,
                    "block_align": backend.block_align,
                    "header_bytes": backend.header_bytes,
                    "slot_bytes": backend.slot_bytes,
                    "meta_total_bytes": backend.meta_total_bytes,
                    "meta_magic": backend.meta_magic_text,
                    "meta_version": backend.meta_version,
                    "data_base_offset": backend._data_base_offset,
                    "next_slot": 0,
                    "free_slots": [],
                    "lru_keys": [],
                    "entries": entries,
                }
            )
            assert applied is True
            assert backend._index == {}
        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp4_initialization(memory_allocator, loop_in_thread):
    """Test TP=4 initialization with per-TP device paths."""
    TP = 4
    with tempfile.TemporaryDirectory() as td:
        device_paths = [os.path.join(td, f"device{i}.bin") for i in range(TP)]
        for p in device_paths:
            with open(p, "wb") as f:
                f.truncate(256 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp4_init",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {
                str(i): device_paths[i] for i in range(TP)
            },
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        backends = []
        try:
            for i in range(TP):
                metadata = LMCacheMetadata(
                    model_name="test-model",
                    world_size=TP,
                    local_world_size=TP,
                    worker_id=i,
                    local_worker_id=i,
                    kv_dtype=torch.bfloat16,
                    kv_shape=(32, 2, 256, 32, 128),
                )
                local_cpu = LocalCPUBackend(
                    config=config,
                    metadata=metadata,
                    dst_device="cpu",
                    memory_allocator=memory_allocator,
                )
                be = RustRawBlockBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=loop_in_thread,
                    dst_device="cpu",
                )
                backends.append(be)
                assert be.device_path == device_paths[i]
            assert len({b.device_path for b in backends}) == TP
        finally:
            for backend in backends:
                backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp4_initialization_accepts_integer_yaml_keys(
    memory_allocator, loop_in_thread
):
    """Accept integer per-TP YAML keys in addition to quoted string keys."""
    TP = 4
    with tempfile.TemporaryDirectory() as td:
        device_paths = [os.path.join(td, f"device{i}.bin") for i in range(TP)]
        for p in device_paths:
            with open(p, "wb") as f:
                f.truncate(256 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp4_init_int_keys",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {
                i: device_paths[i] for i in range(TP)
            },
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        backends = []
        try:
            for i in range(TP):
                metadata = LMCacheMetadata(
                    model_name="test-model",
                    world_size=TP,
                    local_world_size=TP,
                    worker_id=i,
                    local_worker_id=i,
                    kv_dtype=torch.bfloat16,
                    kv_shape=(32, 2, 256, 32, 128),
                )
                local_cpu = LocalCPUBackend(
                    config=config,
                    metadata=metadata,
                    dst_device="cpu",
                    memory_allocator=memory_allocator,
                )
                be = RustRawBlockBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=loop_in_thread,
                    dst_device="cpu",
                )
                backends.append(be)
                assert be.device_path == device_paths[i]
        finally:
            for backend in backends:
                backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp4_comprehensive_io(memory_allocator, loop_in_thread):
    """Comprehensive TP=4 I/O test covering roundtrip, multiple ops, and isolation."""
    TP = 4
    with tempfile.TemporaryDirectory() as td:
        device_paths = [os.path.join(td, f"device{i}.bin") for i in range(TP)]
        for p in device_paths:
            with open(p, "wb") as f:
                f.truncate(512 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp4_io",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {
                str(i): device_paths[i] for i in range(TP)
            },
            "rust_raw_block.capacity_bytes": 0,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        backends = []
        try:
            for i in range(TP):
                metadata = LMCacheMetadata(
                    model_name="test-model",
                    world_size=TP,
                    local_world_size=TP,
                    worker_id=i,
                    local_worker_id=i,
                    kv_dtype=torch.bfloat16,
                    kv_shape=(32, 2, 256, 32, 128),
                )
                local_cpu = LocalCPUBackend(
                    config=config,
                    metadata=metadata,
                    dst_device="cpu",
                    memory_allocator=memory_allocator,
                )
                be = RustRawBlockBackend(
                    config=config,
                    metadata=metadata,
                    local_cpu_backend=local_cpu,
                    loop=loop_in_thread,
                    dst_device="cpu",
                )
                backends.append(be)

            allocator = AdHocMemoryAllocator(device="cpu")
            for tp in range(TP):
                backend = backends[tp]
                key = CacheEngineKey("test-model", TP, tp, 1000 + tp, torch.bfloat16)
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None
                obj.tensor.fill_(tp * 10)
                expected = bytes(obj.byte_array)

                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

                out = backend.get_blocking(key)
                assert out is not None
                assert bytes(out.byte_array) == expected
                out.ref_count_down()

                for other in range(TP):
                    if other == tp:
                        continue
                    other_key = CacheEngineKey(
                        "test-model", TP, other, 1000 + other, torch.bfloat16
                    )
                    assert backend.get_blocking(other_key) is None

            all_keys = []
            for tp in range(TP):
                keys = []
                for i in range(TP - 1):
                    allocator = AdHocMemoryAllocator(device="cpu")
                    obj = allocator.allocate(
                        [torch.Size([2, 16, 8, 128])],
                        [torch.bfloat16],
                        fmt=MemoryFormat.KV_T2D,
                    )
                    assert obj is not None
                    assert obj.tensor is not None
                    obj.tensor.fill_(tp * 100 + i)
                    key = CacheEngineKey(
                        "test-model", TP, tp, tp * 100 + i, torch.bfloat16
                    )
                    keys.append(key)
                    futs = backends[tp].batched_submit_put_task([key], [obj])
                    assert futs is not None
                    futs[0].result(timeout=10)
                    obj.ref_count_down()
                all_keys.append(keys)

            for tp in range(TP):
                for key in all_keys[tp]:
                    out = backends[tp].get_blocking(key)
                    assert out is not None
                    assert out.tensor is not None
                    out.ref_count_down()

            for tp in range(TP):
                for other in range(TP):
                    if other == tp:
                        continue
                    for key in all_keys[other]:
                        assert backends[tp].get_blocking(key) is None
        finally:
            for backend in backends:
                backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_tp_paths_must_be_unique(
    memory_allocator, loop_in_thread
):
    """Reject TP config when multiple ranks point to the same partition."""
    with tempfile.TemporaryDirectory() as td:
        shared_dev = os.path.join(td, "shared.bin")
        with open(shared_dev, "wb") as f:
            f.truncate(512 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_tp_dupe_paths",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.per_tp_device_paths": {"0": shared_dev, "1": shared_dev},
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }
        metadata = LMCacheMetadata(
            model_name="test-model",
            world_size=2,
            local_world_size=2,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(32, 2, 256, 32, 128),
        )
        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        with pytest.raises(ValueError, match="Duplicate device path configured"):
            RustRawBlockBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu,
                loop=loop_in_thread,
                dst_device="cpu",
            )


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_warns_on_cross_rank_metadata_load(
    memory_allocator, loop_in_thread
):
    """Warn when loading metadata whose first entry belongs to another worker."""
    with tempfile.TemporaryDirectory() as td:
        device0 = os.path.join(td, "device0.bin")
        device1 = os.path.join(td, "device1.bin")
        for p in [device0, device1]:
            with open(p, "wb") as f:
                f.truncate(512 * 1024 * 1024)

        base_extra = {
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
        }

        config_tp0 = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_cross_rank_warn_tp0",
        )
        config_tp0.storage_plugins = []
        config_tp0.extra_config = {
            **base_extra,
            "rust_raw_block.per_tp_device_paths": {"0": device0, "1": device1},
        }
        metadata_tp0 = LMCacheMetadata(
            model_name="test-model",
            world_size=2,
            local_world_size=2,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(32, 2, 256, 32, 128),
        )
        local_cpu_tp0 = LocalCPUBackend(
            config=config_tp0,
            metadata=metadata_tp0,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )
        backend_tp0 = RustRawBlockBackend(
            config=config_tp0,
            metadata=metadata_tp0,
            local_cpu_backend=local_cpu_tp0,
            loop=loop_in_thread,
            dst_device="cpu",
        )
        key_tp0 = CacheEngineKey("test-model", 2, 0, 31337, torch.bfloat16)
        allocator = AdHocMemoryAllocator(device="cpu")
        obj = allocator.allocate(
            [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
        )
        assert obj is not None
        assert obj.tensor is not None
        obj.tensor.fill_(9)
        try:
            futs = backend_tp0.batched_submit_put_task([key_tp0], [obj])
            assert futs is not None
            futs[0].result(timeout=10)
            obj.ref_count_down()
        finally:
            backend_tp0.close()

        config_tp1_mis = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_backend_cross_rank_warn_tp1",
        )
        config_tp1_mis.storage_plugins = []
        config_tp1_mis.extra_config = {
            **base_extra,
            "rust_raw_block.per_tp_device_paths": {"1": device0},
        }
        metadata_tp1 = LMCacheMetadata(
            model_name="test-model",
            world_size=2,
            local_world_size=2,
            worker_id=1,
            local_worker_id=1,
            kv_dtype=torch.bfloat16,
            kv_shape=(32, 2, 256, 32, 128),
        )
        local_cpu_tp1 = LocalCPUBackend(
            config=config_tp1_mis,
            metadata=metadata_tp1,
            dst_device="cpu",
            memory_allocator=memory_allocator,
        )

        with patch(
            "lmcache.v1.storage_backend.plugins.rust_raw_block_backend.logger.warning"
        ) as mock_warning:
            backend_tp1 = RustRawBlockBackend(
                config=config_tp1_mis,
                metadata=metadata_tp1,
                local_cpu_backend=local_cpu_tp1,
                loop=loop_in_thread,
                dst_device="cpu",
            )
        try:
            matched = False
            for call in mock_warning.call_args_list:
                call_args = call.args
                if not call_args:
                    continue
                fmt = call_args[0]
                if "loaded metadata may belong to another worker" not in str(fmt):
                    continue
                assert int(call_args[2]) == 1
                assert int(call_args[3]) == 0
                matched = True
                break
            assert matched, "Expected cross-rank metadata warning was not emitted"
        finally:
            backend_tp1.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_uring_put_get_roundtrip(
    memory_allocator, loop_in_thread
):
    """Test batched write with io_uring and verify data integrity on read.

    Writes 128 items asynchronously, then reads them back and verifies
    the data matches what was written.
    """
    NUM_OPS = 128

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(1024 * 1024 * 1024)  # 1G

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=1,
            lmcache_instance_id="test_rust_raw_block_uring",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.use_uring": True,  # Enable io_uring
            "rust_raw_block.use_odirect": True,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
            chunk_size=256,
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        try:
            allocator = local_cpu.memory_allocator

            # Create keys and memory objects with unique data patterns
            keys = []
            objs = []
            expected_data = []

            for i in range(NUM_OPS):
                key = CacheEngineKey("test_model", 1, 0, i, torch.bfloat16)
                keys.append(key)

                # Each object gets a unique fill value
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None

                # Use unique pattern: (i + 1) as fill value
                obj.tensor.fill_(float(i + 1))
                expected_data.append(bytes(obj.byte_array))
                objs.append(obj)

            # Submit all writes using io_uring batched write
            futs = backend.batched_submit_put_task(keys, objs)
            assert futs is not None
            futs[0].result(timeout=10)

            # Read back and verify data integrity
            for i, key in enumerate(keys):
                out = backend.get_blocking(key)
                assert out is not None, f"Failed to read key {i}"
                actual_data = bytes(out.byte_array)
                assert actual_data == expected_data[i], (
                    f"Data mismatch for key {i}: "
                    f"expected first bytes {expected_data[i][:16]}, "
                    f"got {actual_data[:16]}"
                )

        finally:
            backend.close()


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_close_with_inflight(memory_allocator, loop_in_thread):
    """Test that closing the backend while writes are queued / inflight
    is handled gracefully (uring)."""
    NUM_OPS = 128

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(1024 * 1024 * 1024)  # 1G

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=1,
            lmcache_instance_id="test_rust_raw_block_uring",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.use_uring": True,
            "rust_raw_block.use_odirect": True,
        }
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 256, 8, 128),
            chunk_size=256,
        )

        local_cpu = LocalCPUBackend(
            config=config,
            metadata=metadata,
            dst_device="cpu",
        )
        backend = RustRawBlockBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu,
            loop=loop_in_thread,
            dst_device="cpu",
        )

        # Dummy raw that simulates few operations.
        class DummyRaw:
            def __init__(self):
                self._inner = backend._rawdev()

            def batched_write(self, offsets, buffers, total_lens):
                return self._inner.batched_write(offsets, buffers, total_lens)

            def read_uring(self, offset, data, payload_len, total_len):
                return self._inner.read_uring(offset, data, payload_len, total_len)

            def wait_iouring(self, batch_id):
                self._inner.wait_iouring(batch_id)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def close(self):
                self._inner.close()

        backend._raw = DummyRaw()

        try:
            allocator = local_cpu.memory_allocator

            # Create keys and memory objects with unique data patterns
            keys = []
            objs = []
            expected_data = []
            offsets = []
            buffers = []
            total_lens = []

            for i in range(NUM_OPS):
                key = CacheEngineKey("test_model", 1, 0, i, torch.bfloat16)
                keys.append(key)

                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None

                obj.tensor.fill_(float(i + 1))
                expected_data.append(bytes(obj.byte_array))
                objs.append(obj)

                offset = i * len(obj.byte_array)
                total_len = len(obj.byte_array)
                buf = obj.byte_array
                offsets.append(offset)
                buffers.append(buf)
                total_lens.append(total_len)
            backend._raw.batched_write(offsets, buffers, total_lens)
            time.sleep(0.001)
            # close while writes may still be in progress.
            backend.close()

            backend = RustRawBlockBackend(
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu,
                loop=loop_in_thread,
                dst_device="cpu",
            )
            # Read back and verify data integrity
            for i, key in enumerate(keys):
                obj = allocator.allocate(
                    [torch.Size([2, 16, 8, 128])],
                    [torch.bfloat16],
                    fmt=MemoryFormat.KV_T2D,
                )
                assert obj is not None
                assert obj.tensor is not None

                offset = i * len(obj.byte_array)
                total_len = len(obj.byte_array)
                buf = obj.byte_array
                backend._raw.read_uring(offset, buf, total_len, total_len)
                actual_data = bytes(obj.byte_array)
                assert actual_data == expected_data[i], (
                    f"Data mismatch for key {i}: "
                    f"expected first bytes {expected_data[i][:16]}, "
                    f"got {actual_data[:16]}"
                )

        finally:
            try:
                backend.close()
            except Exception:
                pass


@pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)
def test_rust_raw_block_backend_batched_get_with_uring(
    memory_allocator, loop_in_thread
):
    """Test batched_get_blocking and batched_get_non_blocking with io_uring enabled."""
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(64 * 1024 * 1024)

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            lmcache_instance_id="test_rust_raw_block_batched_get_uring",
        )
        config.storage_plugins = []
        config.extra_config = {
            "rust_raw_block.device_path": dev_path,
            "rust_raw_block.block_align": 4096,
            "rust_raw_block.header_bytes": 4096,
            "rust_raw_block.meta_total_bytes": 4 * 1024 * 1024,
            "rust_raw_block.meta_enable_periodic": False,
            "rust_raw_block.use_uring": True,  # Enable io_uring
            "rust_raw_block.use_odirect": True,
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
            allocator = local_cpu.memory_allocator
            key1 = CacheEngineKey("test_model", 1, 0, 4001, torch.bfloat16)
            key2 = CacheEngineKey("test_model", 1, 0, 4002, torch.bfloat16)
            key_miss = CacheEngineKey("test_model", 1, 0, 4003, torch.bfloat16)
            key3 = CacheEngineKey("test_model", 1, 0, 4004, torch.bfloat16)

            obj1 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            obj2 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            obj3 = allocator.allocate(
                [torch.Size([2, 16, 8, 128])], [torch.bfloat16], fmt=MemoryFormat.KV_T2D
            )
            assert obj1 is not None and obj1.tensor is not None
            assert obj2 is not None and obj2.tensor is not None
            assert obj3 is not None and obj3.tensor is not None
            obj1.tensor.fill_(41)
            obj2.tensor.fill_(42)
            obj3.tensor.fill_(43)
            expected1 = bytes(obj1.byte_array)
            expected2 = bytes(obj2.byte_array)

            # Put keys 1, 2, and 3
            for key, obj in ((key1, obj1), (key2, obj2), (key3, obj3)):
                futs = backend.batched_submit_put_task([key], [obj])
                assert futs is not None
                futs[0].result(timeout=10)
                obj.ref_count_down()

            # Test batched_get_blocking with uring. It should stop at first miss
            blocking_results = backend.batched_get_blocking(
                [key1, key2, key_miss, key3]
            )
            assert len(blocking_results) == 4
            assert blocking_results[0] is not None
            assert bytes(blocking_results[0].byte_array) == expected1
            assert blocking_results[1] is not None
            assert bytes(blocking_results[1].byte_array) == expected2
            assert blocking_results[2] is None  # Miss
            assert blocking_results[3] is None  # After miss
            blocking_results[0].ref_count_down()
            blocking_results[1].ref_count_down()

            # Test batched_get_non_blocking with uring.
            # It should return only successful prefix
            future = asyncio.run_coroutine_threadsafe(
                backend.batched_get_non_blocking(
                    "lookup-uring", [key1, key2, key_miss, key3]
                ),
                loop_in_thread,
            )
            async_results = future.result(timeout=10)
            assert len(async_results) == 2  # Only key1 and key2
            assert bytes(async_results[0].byte_array) == expected1
            assert bytes(async_results[1].byte_array) == expected2
            async_results[0].ref_count_down()
            async_results[1].ref_count_down()

        finally:
            backend.close()

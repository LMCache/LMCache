# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
import asyncio
import os
import struct
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

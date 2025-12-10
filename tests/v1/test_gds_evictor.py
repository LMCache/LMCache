# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import safetensors
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import CuFileMemoryAllocator, MemoryFormat
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.gds_backend import pack_metadata, unpack_metadata


def dumb_metadata(fmt="vllm", kv_shape=(32, 2, 256, 8, 128)):
    """Create a dummy metadata object for testing."""
    return LMCacheEngineMetadata("test_model", 1, 0, fmt, torch.bfloat16, kv_shape)


def test_gds_backend_metadata():
    # This is a sanity check that packing and unpacking works. We can add
    # more tensor types to be sure.
    for [tensor, expected_nbytes] in [(torch.randn(3, 10), 120)]:
        r = pack_metadata(tensor, fmt=MemoryFormat.KV_2LTD, version="test")
        size, dtype, nbytes, fmt, meta = unpack_metadata(r)
        assert size == tensor.size()
        assert dtype == tensor.dtype
        assert expected_nbytes == nbytes
        assert fmt == MemoryFormat.KV_2LTD
        assert meta["version"] == "test"

        # Make sure that safetensors can load this
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "test.safetensors")
            with open(temp_file_path, "wb") as f:
                f.write(r)
                f.write(b" " * nbytes)

            with safetensors.safe_open(temp_file_path, framework="pt") as f:
                tensor = f.get_tensor("kvcache")
                assert size == tensor.size()
                assert dtype == tensor.dtype
                assert expected_nbytes == nbytes


def test_gds_backend_eviction_lru():
    """
    Test LRU eviction for GDS backend.
    Insert 3 blocks, cache can only hold 2. The least recently used should be evicted.
    """
    BASE_DIR = Path(__file__).parent
    GDS_DIR = "/tmp/gds/test-cache-evict"
    BACKEND_NAME = "GdsBackend"
    BLOCK_SHAPE = [2048, 2048]  # ~4MB per block (uint8)

    try:
        os.makedirs(GDS_DIR, exist_ok=True)
        config_gds = LMCacheEngineConfig.from_file(BASE_DIR / "data/gds.yaml")
        # Set max_gds_size to hold ~2 blocks (~8MB)
        # NOTE: max_gds_size is in GB, so 8MB = 8 / 1024 GB
        config_gds.max_gds_size = 8 / 1024  # 8MB in GB
        # Disable cufile to avoid hardware-specific issues
        config_gds.extra_config = {"use_cufile": False}

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gds,
            dumb_metadata(),
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gds, None, None),
        )
        gds_backend = backends[BACKEND_NAME]

        # Create 3 keys
        keys = []
        for i in range(3):
            key = CacheEngineKey(
                fmt="vllm",
                model_name=f"evict-test-model-{i}",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=torch.uint8,
            )
            keys.append(key)

        # Store block 1 and block 2
        for i in range(2):
            memory_obj = gds_backend.memory_allocator.allocate(
                BLOCK_SHAPE, dtype=torch.uint8
            )
            print(f"[TEST DEBUG] Allocated memory for key {i}: physical_size={memory_obj.get_physical_size()}")
            future = gds_backend.submit_put_task(keys[i], memory_obj)
            future.result()
            print(f"[TEST DEBUG] After PUT {i}: current_cache_size={gds_backend.current_cache_size}, max_cache_size={gds_backend.max_cache_size}, hot_cache_keys={[k.chunk_hash for k in gds_backend.hot_cache.keys()]}")

        # Access block 1 to make it recently used
        _ = gds_backend.get_blocking(keys[0])
        print(f"[TEST DEBUG] After get_blocking keys[0]: hot_cache_keys={[k.chunk_hash for k in gds_backend.hot_cache.keys()]}")

        # Store block 3 -> block 2 should be evicted (LRU)
        memory_obj = gds_backend.memory_allocator.allocate(
            BLOCK_SHAPE, dtype=torch.uint8
        )
        print(f"[TEST DEBUG] Allocated memory for key 2: physical_size={memory_obj.get_physical_size()}")
        future = gds_backend.submit_put_task(keys[2], memory_obj)
        future.result()
        print(f"[TEST DEBUG] After PUT 2: current_cache_size={gds_backend.current_cache_size}, max_cache_size={gds_backend.max_cache_size}, hot_cache_keys={[k.chunk_hash for k in gds_backend.hot_cache.keys()]}")

        # Verify: block 1 should still exist (recently accessed)
        assert gds_backend.contains(keys[0], False), "Key 0 should remain in cache!"

        # Verify: block 2 should be evicted (LRU)
        assert not gds_backend.contains(keys[1], False), "Key 1 should be evicted!"

        # Verify: block 3 should exist (just inserted)
        assert gds_backend.contains(keys[2], False), "Key 2 should remain in cache!"

    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        if os.path.exists(GDS_DIR):
            shutil.rmtree(GDS_DIR)


def test_gds_backend_eviction_fifo():
    """
    Test FIFO eviction for GDS backend.
    Insert 3 blocks, cache can only hold 2. The first inserted should be evicted.
    """
    BASE_DIR = Path(__file__).parent
    GDS_DIR = "/tmp/gds/test-cache-evict-fifo"
    BACKEND_NAME = "GdsBackend"
    BLOCK_SHAPE = [2048, 2048]

    try:
        os.makedirs(GDS_DIR, exist_ok=True)
        config_gds = LMCacheEngineConfig.from_file(BASE_DIR / "data/gds.yaml")
        # NOTE: max_gds_size is in GB, so 8MB = 8 / 1024 GB
        config_gds.max_gds_size = 8 / 1024  # 8MB in GB
        # Disable cufile to avoid hardware-specific issues
        config_gds.extra_config = {"use_cufile": False}
        # Set cache_policy to FIFO if supported
        if hasattr(config_gds, 'cache_policy'):
            config_gds.cache_policy = "fifo"

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gds,
            dumb_metadata(),
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gds, None, None),
        )
        gds_backend = backends[BACKEND_NAME]

        keys = []
        for i in range(3):
            key = CacheEngineKey(
                fmt="vllm",
                model_name=f"fifo-test-model-{i}",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=torch.uint8,
            )
            keys.append(key)

        # Store all 3 blocks
        for i in range(3):
            memory_obj = gds_backend.memory_allocator.allocate(
                BLOCK_SHAPE, dtype=torch.uint8
            )
            future = gds_backend.submit_put_task(keys[i], memory_obj)
            future.result()

        # FIFO: first inserted (key 0) should be evicted
        assert not gds_backend.contains(keys[0], False), "FIFO: Key 0 should be evicted!"
        assert gds_backend.contains(keys[1], False), "FIFO: Key 1 should remain!"
        assert gds_backend.contains(keys[2], False), "FIFO: Key 2 should remain!"

    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        if os.path.exists(GDS_DIR):
            shutil.rmtree(GDS_DIR)


def test_gds_backend_no_eviction():
    """
    Test that no eviction happens when cache size is large enough.
    """
    BASE_DIR = Path(__file__).parent
    GDS_DIR = "/tmp/gds/test-cache-no-evict"
    BACKEND_NAME = "GdsBackend"
    BLOCK_SHAPE = [2048, 2048]

    try:
        os.makedirs(GDS_DIR, exist_ok=True)
        config_gds = LMCacheEngineConfig.from_file(BASE_DIR / "data/gds.yaml")
        # Large cache size to hold all blocks
        # NOTE: max_gds_size is in GB, so 100MB = 100 / 1024 GB
        config_gds.max_gds_size = 100 / 1024  # 100MB in GB
        # Disable cufile to avoid hardware-specific issues
        config_gds.extra_config = {"use_cufile": False}

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gds,
            dumb_metadata(),
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gds, None, None),
        )
        gds_backend = backends[BACKEND_NAME]

        keys = []
        for i in range(5):
            key = CacheEngineKey(
                fmt="vllm",
                model_name=f"no-evict-model-{i}",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=torch.uint8,
            )
            keys.append(key)
            memory_obj = gds_backend.memory_allocator.allocate(
                BLOCK_SHAPE, dtype=torch.uint8
            )
            future = gds_backend.submit_put_task(key, memory_obj)
            future.result()

        # All keys should remain in cache
        for i, key in enumerate(keys):
            assert gds_backend.contains(key, False), f"Key {i} should remain in cache!"

    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        if os.path.exists(GDS_DIR):
            shutil.rmtree(GDS_DIR)

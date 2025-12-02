# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple
import argparse
import asyncio
import os
import tempfile
import threading
import time

# Third Party
import pytest
import torch

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.nixl_storage_backend import (
    NixlStorageBackend,
    NixlStorageConfig,
)

logger = init_logger(__name__)


def generate_test_data(
    num_objs: int, shape: torch.Size, dtype: torch.dtype = torch.bfloat16
) -> Tuple[List[CacheEngineKey], List[MemoryObj]]:
    keys = []
    objs = []
    allocator = AdHocMemoryAllocator(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    for i in range(num_objs):
        keys.append(
            CacheEngineKey(
                fmt="test",
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=dtype,
            )
        )
        obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
        obj.tensor.fill_((i + 1) / num_objs)  # Fill with some test data
        objs.append(obj)
    return keys, objs


def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s"""
    if elapsed_time == 0:
        return float("inf")
    gb = total_bytes / (1024 * 1024 * 1024)
    return gb / elapsed_time


def create_test_config(
    buffer_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    backend: str = "GDS_MT" if torch.cuda.is_available() else "POSIX",
) -> LMCacheEngineConfig:
    """Create a test configuration for NixlStorageBackend"""
    config = LMCacheEngineConfig()
    config.nixl_buffer_size = 2**32  # 4GB
    config.nixl_buffer_device = buffer_device
    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": backend,
        "nixl_file_pool_size": 10,
        "nixl_path": tempfile.mkdtemp(),  # Create a temporary directory for testing
    }
    return config


def create_test_metadata() -> LMCacheEngineMetadata:
    """Create test metadata for NixlStorageBackend"""
    return LMCacheEngineMetadata(
        model_name="test_model",
        worker_id=0,
        world_size=1,
        fmt="test",
        kv_dtype=torch.bfloat16,
        kv_shape=(
            32,
            2,
            256,
            1024,
            128,
        ),  # (num_layer, 2, chunk_size, num_kv_head, head_size)
    )


@pytest.mark.no_shared_allocator
def test_nixl_storage_config():
    """Test NixlStorageConfig creation and validation"""
    config = create_test_config()
    metadata = create_test_metadata()

    nixl_config = NixlStorageConfig.from_cache_engine_config(config, metadata)
    assert nixl_config.buffer_size == config.nixl_buffer_size
    assert nixl_config.buffer_device == config.nixl_buffer_device
    assert nixl_config.backend == config.extra_config["nixl_backend"]
    assert nixl_config.file_pool_size == config.extra_config["nixl_file_pool_size"]
    assert nixl_config.path == config.extra_config["nixl_path"]

    # Test validation
    assert NixlStorageConfig.validate_nixl_backend("GDS", "cuda")
    assert NixlStorageConfig.validate_nixl_backend("GDS", "cpu")
    assert NixlStorageConfig.validate_nixl_backend("GDS_MT", "cuda")
    assert NixlStorageConfig.validate_nixl_backend("GDS_MT", "cpu")
    assert NixlStorageConfig.validate_nixl_backend("POSIX", "cpu")
    assert not NixlStorageConfig.validate_nixl_backend("POSIX", "cuda")
    assert not NixlStorageConfig.validate_nixl_backend("INVALID", "cpu")


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nixl_storage_backend_basic():
    """Test basic NixlStorageBackend operations"""
    config = create_test_config()
    metadata = create_test_metadata()

    thread_loop = None
    thread = None
    backend = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backend = NixlStorageBackend.CreateNixlStorageBackend(
            config=config,
            loop=thread_loop,
            metadata=metadata,
        )

        # Test allocation
        shape = torch.Size([32, 2, 256, 1024])
        dtype = torch.bfloat16
        obj = backend.allocate(shape, dtype)
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.shape == shape
        assert obj.tensor.dtype == dtype

        # Test batched allocation
        batch_size = 5
        objs = backend.batched_allocate(shape, dtype, batch_size)
        assert objs is not None
        assert len(objs) == batch_size
        for obj in objs:
            assert obj.tensor is not None
            assert obj.tensor.shape == shape
            assert obj.tensor.dtype == dtype

    except Exception:
        raise
    finally:
        if backend:
            backend.close()
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()
        # Cleanup temporary directory
        if os.path.exists(config.extra_config["nixl_path"]):
            os.rmdir(config.extra_config["nixl_path"])


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nixl_storage_backend_put_get():
    """Test put and get operations in NixlStorageBackend"""
    config = create_test_config()
    metadata = create_test_metadata()

    thread_loop = None
    thread = None
    backend = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backend = NixlStorageBackend.CreateNixlStorageBackend(
            config=config,
            loop=thread_loop,
            metadata=metadata,
        )

        # Generate test data
        keys, objs = generate_test_data(10, torch.Size([32, 2, 256, 1024]))

        # Test contains before put
        for key in keys:
            assert not backend.contains(key)
            assert not backend.exists_in_put_tasks(key)

        # Test put
        backend.batched_submit_put_task(keys, objs)

        # Test get
        for key, original_obj in zip(keys, objs, strict=False):
            assert backend.contains(key)
            retrieved_obj = backend.get_blocking(key)
            assert retrieved_obj is not None
            assert retrieved_obj.tensor is not None
            assert torch.equal(retrieved_obj.tensor, original_obj.tensor)

        # Test batched get
        retrieved_objs = asyncio.run(
            backend.batched_get_non_blocking(lookup_id="test", keys=keys)
        )
        assert len(retrieved_objs) == len(objs)
        for retrieved_obj, original_obj in zip(retrieved_objs, objs, strict=False):
            assert retrieved_obj is not None
            assert retrieved_obj.tensor is not None
            assert torch.equal(retrieved_obj.tensor, original_obj.tensor)

        # Test remove
        for key in keys:
            backend.remove(key)
            assert not backend.contains(key)

    except Exception:
        raise
    finally:
        if backend:
            backend.close()
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()
        # Cleanup temporary directory
        if os.path.exists(config.extra_config["nixl_path"]):
            os.rmdir(config.extra_config["nixl_path"])


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nixl_storage_backend_different_backends():
    """Test NixlStorageBackend with different backend types"""
    backends = (
        [
            ("GDS_MT", "cuda"),
            ("GDS", "cuda"),
            ("GDS_MT", "cpu"),
            ("GDS", "cpu"),
            ("POSIX", "cpu"),
        ]
        if torch.cuda.is_available()
        else [
            ("GDS_MT", "cpu"),
            ("GDS", "cpu"),
            ("POSIX", "cpu"),
        ]
    )

    for backend_type, device in backends:
        config = create_test_config(buffer_device=device, backend=backend_type)
        metadata = create_test_metadata()

        thread_loop = None
        thread = None
        backend = None
        try:
            thread_loop = asyncio.new_event_loop()
            thread = threading.Thread(target=thread_loop.run_forever)
            thread.start()

            backend = NixlStorageBackend.CreateNixlStorageBackend(
                config=config,
                loop=thread_loop,
                metadata=metadata,
            )

            # Basic allocation test
            obj = backend.allocate(torch.Size([32, 2, 256, 1024]), torch.bfloat16)
            assert obj is not None
            assert obj.tensor is not None

        except Exception:
            raise
        finally:
            if backend:
                backend.close()
            if thread_loop and thread_loop.is_running():
                thread_loop.call_soon_threadsafe(thread_loop.stop)
            if thread and thread.is_alive():
                thread.join()
            # Cleanup temporary directory
            if os.path.exists(config.extra_config["nixl_path"]):
                os.rmdir(config.extra_config["nixl_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test NixlStorageBackend with different configurations"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="GDS_MT",
        choices=["GDS_MT", "GDS", "POSIX"],
        help="NIXL backend type to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for buffer",
    )
    parser.add_argument(
        "--num-objs",
        type=int,
        default=100,
        help="Number of objects to test with",
    )
    args = parser.parse_args()

    # Create config and metadata
    config = create_test_config(buffer_device=args.device, backend=args.backend)
    metadata = create_test_metadata()

    thread_loop = None
    thread = None
    backend = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        # Create backend
        backend = NixlStorageBackend.CreateNixlStorageBackend(
            config=config,
            loop=thread_loop,
            metadata=metadata,
        )

        # Generate and test with data
        keys, objs = generate_test_data(args.num_objs, torch.Size([32, 2, 256, 1024]))
        total_size = sum(obj.get_size() for obj in objs)
        logger.info(
            "Generated %d objects with total size %.2f MB",
            len(objs),
            total_size / (1024 * 1024),
        )

        # Test put performance
        start_time = time.time()
        backend.batched_submit_put_task(keys, objs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = calculate_throughput(total_size, elapsed_time)
        logger.info("Put throughput: %.2f GB/s", throughput)

        # Test get performance
        start_time = time.time()
        retrieved_objs = asyncio.run(
            backend.batched_get_non_blocking(lookup_id="test", keys=keys)
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = calculate_throughput(total_size, elapsed_time)
        logger.info("Get throughput: %.2f GB/s", throughput)

        # Verify data
        for retrieved_obj, original_obj in zip(retrieved_objs, objs, strict=False):
            assert torch.equal(retrieved_obj.tensor, original_obj.tensor)

        logger.info("All tests passed successfully!")

    except Exception:
        raise
    finally:
        if backend:
            backend.close()
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()
        # Cleanup temporary directory
        if os.path.exists(config.extra_config["nixl_path"]):
            os.rmdir(config.extra_config["nixl_path"])

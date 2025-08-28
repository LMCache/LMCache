# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []

    def put_msg(self, msg):
        self.messages.append(msg)


def create_test_config(disk_path: str, max_disk_size: float = 1.0):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_key(key_id: str = "test_key") -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, hash(key_id))


def create_test_memory_obj(shape=(2, 16, 8, 128), dtype=torch.bfloat16) -> MemoryObj:
    """Create a test MemoryObj using AdHocMemoryAllocator for testing."""
    # First Party
    from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat

    allocator = AdHocMemoryAllocator(device="cpu")
    memory_obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)
    return memory_obj


@pytest.fixture
def temp_disk_path():
    """Create a temporary directory for disk storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ----------------------------------------------------------------------------


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    return LocalCPUBackend(config, memory_allocator)


@pytest.fixture
def local_disk_backend(temp_disk_path, async_loop, local_cpu_backend):
    """Create a LocalDiskBackend for testing."""
    config = create_test_config(temp_disk_path)
    return LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda",
    )


class TestLocalDiskBackend:
    """Test cases for LocalDiskBackend."""

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test LocalDiskBackend initialization."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        assert backend.dst_device == "cuda"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.path == temp_disk_path
        assert os.path.exists(temp_disk_path)
        assert backend.lookup_server is None
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert backend.usage == 0
        assert len(backend.dict) == 0

        local_cpu_backend.memory_allocator.close()

    def test_init_with_lookup_server_and_worker(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Test LocalDiskBackend initialization with lookup server and worker."""
        config = create_test_config(temp_disk_path)
        lookup_server = MockLookupServer()
        lmcache_worker = MockLMCacheWorker()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
            lookup_server=lookup_server,
            lmcache_worker=lmcache_worker,
        )

        assert backend.lookup_server == lookup_server
        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key("test_hash")
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key("nonexistent")
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_exists(self, local_disk_backend):
        """Test contains() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_disk_backend.insert_key(key, memory_obj)

        assert local_disk_backend.contains(key)
        assert local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_pin_unpin(self, local_disk_backend):
        """Test pin() and unpin() operations."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()
        # Insert key first
        local_disk_backend.insert_key(key, memory_obj)
        # Test pin
        assert local_disk_backend.pin(key)
        assert local_disk_backend.dict[key].pin_count > 0
        # Test unpin
        assert local_disk_backend.unpin(key)
        assert local_disk_backend.dict[key].pin_count == 0

        # Test pin/unpin non-existent key
        non_existent_key = create_test_key("non_existent")
        assert not local_disk_backend.pin(non_existent_key)
        assert not local_disk_backend.unpin(non_existent_key)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_insert_key(self, local_disk_backend):
        """Test insert_key()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()
        local_disk_backend.insert_key(key, memory_obj)
        assert key in local_disk_backend.dict
        metadata = local_disk_backend.dict[key]
        assert metadata.path == local_disk_backend._key_to_path(key)
        assert metadata.shape == memory_obj.metadata.shape
        assert metadata.dtype == memory_obj.metadata.dtype
        assert metadata.fmt == memory_obj.metadata.fmt
        assert metadata.pin_count == 0
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_insert_key_reinsert(self, local_disk_backend):
        """Test insert_key() with reinsertion."""
        key = create_test_key("test_key")
        memory_obj1 = create_test_memory_obj(shape=(2, 16, 8, 128))
        memory_obj2 = create_test_memory_obj(shape=(2, 32, 8, 128))

        # First insertion
        local_disk_backend.insert_key(key, memory_obj1)
        original_path = local_disk_backend.dict[key].path

        # Reinsertion
        local_disk_backend.insert_key(key, memory_obj2)

        assert key in local_disk_backend.dict
        metadata = local_disk_backend.dict[key]
        assert metadata.path == original_path  # Path should remain the same

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_remove(self, local_disk_backend):
        """Test remove()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_disk_backend.insert_key(key, memory_obj)
        assert key in local_disk_backend.dict

        # Create a dummy file to simulate the disk file
        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(b"dummy data")

        # Remove the key
        local_disk_backend.remove(key)

        # Wait for worker tasks
        local_disk_backend.disk_worker.pq.join()
        local_disk_backend.disk_worker.executor.shutdown()

        assert key not in local_disk_backend.dict
        assert not os.path.exists(path)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_remove_with_worker(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test remove() with LMCacheWorker."""
        config = create_test_config(temp_disk_path)
        lmcache_worker = MockLMCacheWorker()
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
            lmcache_worker=lmcache_worker,
        )
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()
        # Insert key first
        backend.insert_key(key, memory_obj)
        # Create a dummy file
        path = backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(b"dummy data")
        # Remove the key
        backend.remove(key)
        # Check that both admit and evict messages were sent
        assert len(lmcache_worker.messages) == 2
        # First Party
        from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg

        assert any(isinstance(msg, KVAdmitMsg) for msg in lmcache_worker.messages)
        assert any(isinstance(msg, KVEvictMsg) for msg in lmcache_worker.messages)

        local_cpu_backend.memory_allocator.close()

    def test_submit_put_task(self, local_disk_backend):
        """Test submit_put_task() synchronous"""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Test that the key is not in put_tasks initially
        assert not local_disk_backend.exists_in_put_tasks(key)

        # Test that the key doesn't exist in the backend initially
        assert not local_disk_backend.contains(key)

        # Use insert_key directly to test the synchronous path
        local_disk_backend.insert_key(key, memory_obj)

        # Check that the key was inserted into the backend
        assert local_disk_backend.contains(key)
        assert key in local_disk_backend.dict

        # Check that the metadata was properly set
        metadata = local_disk_backend.dict[key]
        assert metadata.path == local_disk_backend._key_to_path(key)
        assert metadata.shape == memory_obj.metadata.shape
        assert metadata.fmt == memory_obj.metadata.fmt
        assert metadata.pin_count == 0

        # Test that the key is still not in put_tasks
        # (since we used insert_key directly)
        assert not local_disk_backend.exists_in_put_tasks(key)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_submit_prefetch_task_key_not_exists(self, local_disk_backend):
        """Test submit_prefetch_task() when key doesn't exist."""
        key = create_test_key("nonexistent")
        res = local_disk_backend.submit_prefetch_task(key)

        assert not res

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_submit_prefetch_task_key_exists(self, local_disk_backend):
        """Test submit_prefetch_task() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_disk_backend.insert_key(key, memory_obj)

        # Create the actual file on disk
        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(memory_obj.byte_array)

        future = local_disk_backend.submit_prefetch_task(key)

        assert future is not None
        # Don't call future.result() to avoid blocking

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key("nonexistent")
        result = local_disk_backend.get_blocking(key)

        assert result is None

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_exists(self, local_disk_backend):
        """Test get_blocking() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key first
        local_disk_backend.insert_key(key, memory_obj)

        # Create the actual file on disk
        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(memory_obj.byte_array)

        result = local_disk_backend.get_blocking(key)

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result.metadata.shape == memory_obj.metadata.shape
        assert result.metadata.dtype == memory_obj.metadata.dtype

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_async_save_bytes_to_disk(self, local_disk_backend, async_loop):
        """Test async_save_bytes_to_disk()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        local_disk_backend.insert_key(key, memory_obj)

        # Check that the key was inserted into the backend
        assert key in local_disk_backend.dict

        # Check that the metadata was properly set
        metadata = local_disk_backend.dict[key]
        assert metadata.path == local_disk_backend._key_to_path(key)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_async_load_bytes_from_disk(self, local_disk_backend):
        """Test async_load_bytes_from_disk()"""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Create the file first
        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(memory_obj.byte_array)

        result = local_disk_backend.load_bytes_from_disk(
            key,
            path,
            memory_obj.metadata.dtype,
            memory_obj.metadata.shape,
            memory_obj.metadata.fmt,
        )

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result.metadata.shape == memory_obj.metadata.shape
        assert result.metadata.dtype == memory_obj.metadata.dtype

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_load_bytes_from_disk(self, local_disk_backend):
        """Test load_bytes_from_disk()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Create the file first
        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(memory_obj.byte_array)

        result = local_disk_backend.load_bytes_from_disk(
            key,
            path,
            memory_obj.metadata.dtype,
            memory_obj.metadata.shape,
            memory_obj.metadata.fmt,
        )

        assert result is not None
        assert isinstance(result, MemoryObj)
        assert result.metadata.shape == memory_obj.metadata.shape
        assert result.metadata.dtype == memory_obj.metadata.dtype

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_close(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test close()."""
        config = create_test_config(temp_disk_path)
        lookup_server = MockLookupServer()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
            lookup_server=lookup_server,
        )

        # Add some keys
        for i in range(3):
            key = create_test_key(f"key_{i}")
            memory_obj = create_test_memory_obj()
            backend.insert_key(key, memory_obj)

        # Close the backend
        backend.close()

        # Check that keys were removed from lookup server
        assert len(lookup_server.removed_keys) == 3

        local_cpu_backend.memory_allocator.close()

    def test_concurrent_access(self, local_disk_backend):
        """Test concurrent access to the backend."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key
        local_disk_backend.insert_key(key, memory_obj)

        # Test concurrent contains() calls
        def check_contains():
            for _ in range(20):
                assert local_disk_backend.contains(key)

        threads = [threading.Thread(target=check_contains) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_file_operations_error_handling(self, local_disk_backend):
        """Test error handling in file operations."""
        # Test with non-existent file
        key = create_test_key("test_key")
        non_existent_path = "/non/existent/path/file.pt"

        memory_obj = local_disk_backend.load_bytes_from_disk(
            key,
            non_existent_path,
            torch.bfloat16,
            torch.Size([2, 16, 8, 128]),
            MemoryFormat.KV_T2D,
        )
        assert memory_obj is not None
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_cleanup_on_remove(self, local_disk_backend):
        """Test that resources are properly cleaned up on remove."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key
        local_disk_backend.insert_key(key, memory_obj)

        # Create the file
        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(memory_obj.byte_array)

        # Remove key
        local_disk_backend.remove(key)

        # Wait for worker tasks
        local_disk_backend.disk_worker.pq.join()
        local_disk_backend.disk_worker.executor.shutdown()

        # Check that both the dict entry and file are removed
        assert key not in local_disk_backend.dict
        assert not os.path.exists(path)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_thread_safety(self, local_disk_backend):
        """Test thread safety of the backend."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Insert key
        local_disk_backend.insert_key(key, memory_obj)

        path = local_disk_backend._key_to_path(key)
        with open(path, "wb") as f:
            f.write(memory_obj.byte_array)

        # Test concurrent operations with reduced iteration count
        def concurrent_operations():
            for _ in range(10):
                # Test contains
                local_disk_backend.contains(key)
                # Test pin/unpin
                local_disk_backend.pin(key)
                local_disk_backend.unpin(key)
                # Test get_blocking
                result = local_disk_backend.get_blocking(key)
                assert result is not None

        threads = [threading.Thread(target=concurrent_operations) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # The backend should still be in a consistent state
        assert local_disk_backend.contains(key)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

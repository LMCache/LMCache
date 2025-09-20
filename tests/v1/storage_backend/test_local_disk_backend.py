# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
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


def create_test_config(
    disk_path: str, max_disk_size: float = 1.0, **overrides
):
    """Create a test configuration for LocalDiskBackend."""
    config_kwargs = dict(
        chunk_size=256,
        local_disk=disk_path,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
    )
    config_kwargs.update(overrides)
    config = LMCacheEngineConfig.from_defaults(**config_kwargs)
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheEngineMetadata."""
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
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
    return LocalCPUBackend(config, memory_allocator=memory_allocator)


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
        lmcache_worker = MockLMCacheWorker()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key(1)
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key(2)
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key(2)
        result = local_disk_backend.get_blocking(key)

        assert result is None

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_async_load_bytes_from_disk(self, local_disk_backend):
        """Test async_load_bytes_from_disk()"""
        key = create_test_key(3)
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
        key = create_test_key(3)
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

    def test_file_operations_error_handling(self, local_disk_backend):
        """Test error handling in file operations."""
        # Test with non-existent file
        key = create_test_key(3)
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

    def test_disk_persistence_restore_and_prefetch(
        self, temp_disk_path, async_loop, memory_allocator
    ):
        """Disk entries are restored and prefetched to CPU on restart."""
        config = create_test_config(
            temp_disk_path,
            local_disk_persistence=True,
            populate_disk_cache_to_cpu_on_start=True,
        )
        cpu_backend = LocalCPUBackend(config, memory_allocator=memory_allocator)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=cpu_backend,
            dst_device="cuda",
        )

        key = create_test_key(42)
        memory_obj = create_test_memory_obj()
        data_path = backend._key_to_path(key)
        meta_path = data_path + ".meta"

        try:
            backend.async_save_bytes_to_disk(key, memory_obj)
            assert os.path.exists(data_path)
            assert os.path.exists(meta_path)
        finally:
            backend.close()
            cpu_backend.memory_allocator.close()

        restore_config = create_test_config(
            temp_disk_path,
            local_disk_persistence=True,
            populate_disk_cache_to_cpu_on_start=True,
        )
        restore_cpu_backend = LocalCPUBackend(
            restore_config, memory_allocator=memory_allocator
        )
        restored_backend = LocalDiskBackend(
            config=restore_config,
            loop=async_loop,
            local_cpu_backend=restore_cpu_backend,
            dst_device="cuda",
        )

        try:
            assert key in restored_backend.dict
            restored_meta = restored_backend.dict[key]
            assert restored_meta.shape == memory_obj.metadata.shape
            assert restored_meta.dtype == memory_obj.metadata.dtype
            assert restored_meta.fmt == memory_obj.metadata.fmt
            assert restored_meta.size == os.path.getsize(data_path)
            assert restored_backend.current_cache_size == float(restored_meta.size)
            assert restored_backend.usage == restored_meta.size

            cpu_obj = restored_backend.local_cpu_backend.get_blocking(key)
            assert cpu_obj is not None
            assert cpu_obj.metadata.shape == memory_obj.metadata.shape
            assert cpu_obj.metadata.dtype == memory_obj.metadata.dtype
            cpu_obj.ref_count_down()
        finally:
            restored_backend.close()
            restore_cpu_backend.memory_allocator.close()

    def test_remove_deletes_metadata_file(
        self, temp_disk_path, async_loop, memory_allocator
    ):
        """Removing an entry deletes both data and metadata files."""
        config = create_test_config(
            temp_disk_path,
            local_disk_persistence=True,
        )
        cpu_backend = LocalCPUBackend(config, memory_allocator=memory_allocator)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=cpu_backend,
            dst_device="cuda",
        )

        key = create_test_key(55)
        memory_obj = create_test_memory_obj()
        data_path = backend._key_to_path(key)
        meta_path = data_path + ".meta"

        try:
            backend.async_save_bytes_to_disk(key, memory_obj)
            assert os.path.exists(data_path)
            assert os.path.exists(meta_path)

            removed = backend.remove(key)
            assert removed
            assert not os.path.exists(data_path)
            assert not os.path.exists(meta_path)
            assert key not in backend.dict
            assert backend.usage == 0
        finally:
            backend.close()
            cpu_backend.memory_allocator.close()

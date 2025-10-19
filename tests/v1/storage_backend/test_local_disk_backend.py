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


def create_test_config(disk_path: str, max_disk_size: float = 1.0):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
    )
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


@pytest.fixture
def odirect_backend_factory(temp_disk_path, async_loop, local_cpu_backend):
    """Factory fixture for creating LocalDiskBackend with configurable O_DIRECT.

    Yields a factory function that creates a backend with the specified
    use_odirect setting. Handles cleanup automatically via context manager
    pattern.
    """
    backends_created = []

    def _create_backend(use_odirect: bool = False):
        config = create_test_config(temp_disk_path)
        if use_odirect:
            config.extra_config = {"use_odirect": True}
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )
        backends_created.append(backend)
        return backend

    yield _create_backend

    # Cleanup: close memory allocator once after all backends are done
    if backends_created:
        local_cpu_backend.memory_allocator.close()


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

    def test_odirect_config_disabled(self, odirect_backend_factory):
        """Test O_DIRECT is disabled by default."""
        backend = odirect_backend_factory(use_odirect=False)
        assert backend.use_odirect is False

    def test_odirect_config_enabled(self, odirect_backend_factory):
        """Test O_DIRECT can be enabled via extra_config."""
        backend = odirect_backend_factory(use_odirect=True)
        assert backend.use_odirect is True

    def test_write_with_odirect_aligned(self, temp_disk_path, odirect_backend_factory):
        """Test write_file with O_DIRECT enabled and aligned data."""
        backend = odirect_backend_factory(use_odirect=True)

        # Create data aligned to block size
        bs = backend.os_disk_bs
        test_data = b"X" * (bs * 4)  # 4 blocks
        test_path = os.path.join(temp_disk_path, "test_odirect_aligned.bin")

        # Write should succeed
        backend.write_file(test_data, test_path)

        # Verify file exists and content matches
        assert os.path.exists(test_path)
        with open(test_path, "rb") as f:
            content = f.read()
        assert content == test_data

    def test_write_with_odirect_misaligned(
        self, temp_disk_path, odirect_backend_factory
    ):
        """Test write_file with O_DIRECT enabled and misaligned data gets padded.

        Note: On filesystems that don't support O_DIRECT (e.g., tmpfs), this will
        fall back to buffered I/O without padding, which is acceptable.
        """
        backend = odirect_backend_factory(use_odirect=True)

        # Create data NOT aligned to block size
        bs = backend.os_disk_bs
        test_data = b"Y" * (bs * 2 + 100)  # 2 blocks + 100 bytes
        test_path = os.path.join(temp_disk_path, "test_odirect_misaligned.bin")

        # Write should succeed
        backend.write_file(test_data, test_path)

        # Verify file exists
        assert os.path.exists(test_path)

        # Check file size
        file_size = os.path.getsize(test_path)
        expected_padded_size = ((len(test_data) + bs - 1) // bs) * bs

        # Read content
        with open(test_path, "rb") as f:
            content = f.read()

        # Verify original data is preserved
        assert content[: len(test_data)] == test_data

        # Check if O_DIRECT actually worked (indicated by padding)
        if file_size == expected_padded_size:
            # O_DIRECT succeeded - verify padding
            assert content[len(test_data) :] == b"\x00" * (file_size - len(test_data))
        elif file_size == len(test_data):
            # O_DIRECT failed, fell back to buffered I/O (e.g., on tmpfs)
            # This is acceptable behavior
            pass
        else:
            # Unexpected file size
            pytest.fail(
                f"Unexpected file size: {file_size}. "
                f"Expected either {expected_padded_size} (O_DIRECT) "
                f"or {len(test_data)} (buffered fallback)"
            )

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
from lmcache.v1.memory_management import MemoryFormat
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
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
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


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

    def test_disk_cache_metadata_get_num_tokens(self):
        """Test DiskCacheMetadata.get_num_tokens()."""
        metadata = DiskCacheMetadata(
            path="/tmp/test.pt",
            size=128,
            shape=torch.Size((2, 16, 8, 128)),
            dtype=torch.bfloat16,
            fmt=MemoryFormat.KV_T2D,
        )

        assert metadata.get_num_tokens() == 16

    def test_disk_cache_metadata_get_num_tokens_missing_fmt_or_shape(self, capsys):
        """Test DiskCacheMetadata.get_num_tokens() without shape/fmt metadata."""
        metadata = DiskCacheMetadata(path="/tmp/test.pt", size=128)

        assert metadata.get_num_tokens() == 0

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
    
    def test_clear_removes_only_evictable_entries_and_counts_tokens(
        self, local_disk_backend, monkeypatch
    ):
        """Test clear() only removes evictable entries and sums token counts."""
        evictable_key = create_test_key(3)
        pinned_key = create_test_key(4)

        evictable_meta = DiskCacheMetadata(
            path="/tmp/evictable.pt",
            size=64,
            shape=torch.Size((2, 11, 8, 128)),
            dtype=torch.bfloat16,
            fmt=MemoryFormat.KV_T2D,
        )
        pinned_meta = DiskCacheMetadata(
            path="/tmp/pinned.pt",
            size=32,
            shape=torch.Size((2, 7, 8, 128)),
            dtype=torch.bfloat16,
            fmt=MemoryFormat.KV_T2D,
            pin_count=1,
        )

        local_disk_backend.dict[evictable_key] = evictable_meta
        local_disk_backend.dict[pinned_key] = pinned_meta
        local_disk_backend.current_cache_size = evictable_meta.size + pinned_meta.size

        removed_keys = []

        def fake_remove(key, force=True):
            # do not remove real pt file
            removed_keys.append((key, force))
            local_disk_backend.dict.pop(key, None)
            return True

        monkeypatch.setattr(local_disk_backend, "remove", fake_remove)

        num_cleared_tokens = local_disk_backend.clear()

        assert num_cleared_tokens == 11
        assert removed_keys == [(evictable_key, False)]
        assert evictable_key not in local_disk_backend.dict
        assert pinned_key in local_disk_backend.dict
        assert local_disk_backend.current_cache_size == pinned_meta.size

        local_disk_backend.local_cpu_backend.memory_allocator.close()
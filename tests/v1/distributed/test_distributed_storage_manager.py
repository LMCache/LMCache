# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for StorageManager.
"""

# Standard

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)

try:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
except ImportError:
    # Skip tests if L1Manager cannot be imported
    pytest.skip(
        "Skipping because StorageManager cannot be imported", allow_module_level=True
    )

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def should_use_lazy_alloc() -> bool:
    """Determine if lazy allocation should be used based on CUDA availability."""
    return torch.cuda.is_available()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_memory_config():
    """Create a basic L1MemoryManagerConfig for testing."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,  # 4KB
    )


@pytest.fixture
def small_memory_config():
    """Create a small L1MemoryManagerConfig to test memory exhaustion."""
    return L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,  # 64MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,
    )


@pytest.fixture
def basic_l1_config(basic_memory_config):
    """Create a basic L1ManagerConfig for testing."""
    return L1ManagerConfig(
        memory_config=basic_memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def small_l1_config(small_memory_config):
    """Create a small L1ManagerConfig to test memory exhaustion."""
    return L1ManagerConfig(
        memory_config=small_memory_config,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


@pytest.fixture
def basic_storage_manager_config(basic_l1_config):
    """Create a basic StorageManagerConfig for testing"""
    return StorageManagerConfig(
        l1_manager_config=basic_l1_config,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
        ),
    )


@pytest.fixture
def small_storage_manager_config(small_l1_config):
    """Create a small StorageManagerConfig to test memory exhaustion."""
    return StorageManagerConfig(
        l1_manager_config=small_l1_config,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
        ),
    )


@pytest.fixture
def basic_layout():
    """Create a basic MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def large_layout():
    """Create a large MemoryLayoutDesc that will exhaust small memory.

    Each allocation is 8MB (2M elements * 4 bytes).
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size([2048, 1024])],  # 2M elements * 4 bytes = 8MB
        dtypes=[torch.float32],
    )


def make_object_key(chunk_hash: int, model_name: str = "test_model", kv_rank: int = 0):
    """Helper to create ObjectKey instances."""
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model_name, kv_rank=kv_rank)


# =============================================================================
# Tests
# =============================================================================


class TestStorageManagerBasic:
    """Tests for basic functionality of StorageManager."""

    def test_basic_reserve_write(self, basic_storage_manager_config, basic_layout):
        """Test basic reserve and write functionality."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_key = make_object_key(chunk_hash=12345)

        # Reserve space for the object
        ret = storage_manager.reserve_write([object_key], basic_layout, mode="new")
        assert object_key in ret
        assert ret[object_key] is not None

        # Should not have any error
        storage_manager.finish_write([object_key])

        storage_manager.close()

    def test_reserve_write_multiple_keys(
        self, basic_storage_manager_config, basic_layout
    ):
        """Test reserve_write with multiple keys."""
        storage_manager = StorageManager(basic_storage_manager_config)

        keys = [make_object_key(i) for i in range(5)]

        ret = storage_manager.reserve_write(keys, basic_layout, mode="new")

        # All keys should be allocated
        assert len(ret) == len(keys)
        for key in keys:
            assert key in ret
            assert ret[key] is not None

        storage_manager.close()

    def test_reserve_write_oom(self, small_storage_manager_config, large_layout):
        """Test reserve_write raises L1Error on out-of-memory."""
        storage_manager = StorageManager(small_storage_manager_config)

        keys = [make_object_key(i) for i in range(20)]

        ret = storage_manager.reserve_write(keys, large_layout, mode="new")

        # At least some of the keys could be allocated
        assert len(ret) < len(keys)

        # If some keys were allocated, they should not be None
        for key, obj in ret.items():
            assert obj is not None

        storage_manager.close()

    def test_basic_prefetch(self, basic_storage_manager_config, basic_layout):
        """Test basic prefetch functionality."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(5)]

        # Write keys into storage manager
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="new")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch all the objects
        handle = storage_manager.submit_prefetch_task(object_keys)

        hit_count = storage_manager.query_prefetch_status(handle)
        assert hit_count is not None
        assert hit_count == len(object_keys)

        storage_manager.close()

    def test_prefetch_partial_prefix_hits(
        self, basic_storage_manager_config, basic_layout
    ):
        """Test prefetch with partial hits."""
        # 5 keys: 0, 1, 3, 4 are written, 2 is missing
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(5)]

        # Write only some keys into storage manager
        keys_to_write = [object_keys[0], object_keys[1], object_keys[3], object_keys[4]]
        ret = storage_manager.reserve_write(keys_to_write, basic_layout, mode="new")
        for key in keys_to_write:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch all the objects
        handle = storage_manager.submit_prefetch_task(object_keys)

        hit_count = storage_manager.query_prefetch_status(handle)
        assert hit_count is not None
        assert hit_count == 2  # Only 2 keys were written

        # The last 2 keys should be "writable"
        ret = storage_manager.reserve_write(
            object_keys[3:], basic_layout, mode="update"
        )
        for key in object_keys[3:]:
            assert key in ret
            assert ret[key] is not None

        storage_manager.close()

    def test_read_prefetched_basic(self, basic_storage_manager_config, basic_layout):
        """Test reading prefetched objects."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(3)]

        # Write keys into storage manager
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="new")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch all the objects
        handle = storage_manager.submit_prefetch_task(object_keys)

        hit_count = storage_manager.query_prefetch_status(handle)
        assert hit_count is not None
        assert hit_count == len(object_keys)

        # Read the prefetched objects
        with storage_manager.read_prefetched_results(object_keys) as retrieved_objects:
            assert retrieved_objects is not None
            assert len(retrieved_objects) == len(object_keys)

        # Finish reading
        storage_manager.finish_read_prefetched(object_keys)

        # Now the objects should be writable again
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="update")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None

        storage_manager.close()

    def test_read_prefetched_not_found(
        self, basic_storage_manager_config, basic_layout
    ):
        """Test reading prefetched objects that were not found."""
        storage_manager = StorageManager(basic_storage_manager_config)

        object_keys = [make_object_key(i) for i in range(5)]

        # Write all objects into storage manager
        ret = storage_manager.reserve_write(object_keys, basic_layout, mode="new")
        for key in object_keys:
            assert key in ret
            assert ret[key] is not None
        storage_manager.finish_write(list(ret.keys()))

        # Prefetch objects except the first one
        handle = storage_manager.submit_prefetch_task(object_keys[1:])
        hit_count = storage_manager.query_prefetch_status(handle)
        assert hit_count is not None
        assert hit_count == len(object_keys) - 1

        # Attempt to read all the objects, should get None
        with storage_manager.read_prefetched_results(object_keys) as retrieved_objects:
            assert retrieved_objects is None

        # Remaining 4 objects should still be writable (i.e., no dangling read locks)
        ret = storage_manager.reserve_write(
            object_keys[1:], basic_layout, mode="update"
        )
        for key in object_keys[1:]:
            assert key in ret
            assert ret[key] is not None
        storage_manager.close()

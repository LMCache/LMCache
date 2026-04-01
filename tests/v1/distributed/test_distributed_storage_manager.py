# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for StorageManager.
"""

# Standard
import time

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
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig

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


def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """Poll until a predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_status(
    sm: StorageManager,
    handle,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> int | None:
    """Poll query_prefetch_status until it returns a non-None value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


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
        handle = storage_manager.submit_prefetch_task(object_keys, basic_layout)

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
        handle = storage_manager.submit_prefetch_task(object_keys, basic_layout)

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
        handle = storage_manager.submit_prefetch_task(object_keys, basic_layout)

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
        handle = storage_manager.submit_prefetch_task(object_keys[1:], basic_layout)
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


# =============================================================================
# Tests for multi-reader (count / num_readers) support
# =============================================================================


class TestStorageManagerMultiReader:
    """Tests for the num_readers / count parameters.

    These parameters allow multiple workers (e.g. MLA with
    TP > 1) to each hold an independent read lock on the same
    prefetched object.
    """

    def test_prefetch_with_extra_count(
        self, basic_storage_manager_config, basic_layout
    ):
        """submit_prefetch_task(extra_count=N-1) acquires N locks."""
        sm = StorageManager(basic_storage_manager_config)
        keys = [make_object_key(i) for i in range(3)]

        # Write keys
        ret = sm.reserve_write(keys, basic_layout, mode="new")
        assert len(ret) == len(keys)
        sm.finish_write(list(ret.keys()))

        extra_count = 2  # total = 1 + 2 = 3 locks
        handle = sm.submit_prefetch_task(keys, basic_layout, extra_count=extra_count)
        hit = sm.query_prefetch_status(handle)
        assert hit == len(keys)

        # Release with matching extra_count
        sm.finish_read_prefetched(keys, extra_count=extra_count)

        # All locks released -> objects writable again
        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == len(keys)

        sm.close()

    def test_finish_read_prefetched_partial_extra_count(
        self, basic_storage_manager_config, basic_layout
    ):
        """Partial extra_count release leaves locks held."""
        sm = StorageManager(basic_storage_manager_config)
        keys = [make_object_key(i) for i in range(2)]

        ret = sm.reserve_write(keys, basic_layout, mode="new")
        sm.finish_write(list(ret.keys()))

        extra_count = 3  # total = 1 + 3 = 4 locks
        handle = sm.submit_prefetch_task(keys, basic_layout, extra_count=extra_count)
        hit = sm.query_prefetch_status(handle)
        assert hit == len(keys)

        # Release 2 of 4 (1 + extra_count=1)
        sm.finish_read_prefetched(keys, extra_count=1)

        # Objects should NOT be writable (2 locks remain)
        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == 0

        # Release remaining 2 (1 + extra_count=1)
        sm.finish_read_prefetched(keys, extra_count=1)

        # Now writable
        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == len(keys)

        sm.close()

    def test_prefetch_skipped_keys_released_with_extra(
        self, basic_storage_manager_config, basic_layout
    ):
        """Non-prefix L1 hits are released with correct extra.

        Keys {0,1,3,4} exist, key 2 is missing.  The prefix
        hits are {0,1}; keys {3,4} must have their N locks
        released to avoid dangling locks.
        """
        sm = StorageManager(basic_storage_manager_config)
        all_keys = [make_object_key(i) for i in range(5)]
        existing = [all_keys[i] for i in [0, 1, 3, 4]]

        ret = sm.reserve_write(existing, basic_layout, mode="new")
        sm.finish_write(list(ret.keys()))

        extra_count = 1  # total = 1 + 1 = 2 locks
        handle = sm.submit_prefetch_task(
            all_keys, basic_layout, extra_count=extra_count
        )
        hit = sm.query_prefetch_status(handle)
        # Only prefix {0,1} count as hits
        assert hit is not None
        assert hit == 2

        # Finish the prefix hits
        sm.finish_read_prefetched(all_keys[:2], extra_count=extra_count)

        # Keys {3,4} should be writable (skipped locks released)
        ret = sm.reserve_write(
            [all_keys[3], all_keys[4]],
            basic_layout,
            mode="update",
        )
        assert len(ret) == 2

        sm.close()

    def test_extra_count_default_is_zero(
        self, basic_storage_manager_config, basic_layout
    ):
        """Default extra_count=0 behaves same as before."""
        sm = StorageManager(basic_storage_manager_config)
        keys = [make_object_key(i) for i in range(3)]

        ret = sm.reserve_write(keys, basic_layout, mode="new")
        sm.finish_write(list(ret.keys()))

        handle = sm.submit_prefetch_task(keys, basic_layout)
        hit = sm.query_prefetch_status(handle)
        assert hit == len(keys)

        # Single finish is enough
        sm.finish_read_prefetched(keys)

        ret = sm.reserve_write(keys, basic_layout, mode="update")
        assert len(ret) == len(keys)

        sm.close()


# =============================================================================
# L2 Prefetch Integration Tests
# =============================================================================


@pytest.fixture
def l2_storage_manager_config(basic_l1_config):
    """Create a StorageManagerConfig with one MockL2Adapter."""
    return StorageManagerConfig(
        l1_manager_config=basic_l1_config,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
        ),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[
                MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0),
            ],
        ),
    )


class TestStorageManagerL2Prefetch:
    """Tests for prefetching from L2 through StorageManager."""

    def _write_keys_and_wait_for_l2(
        self,
        sm: StorageManager,
        keys: list[ObjectKey],
        layout: MemoryLayoutDesc,
    ) -> None:
        """Write keys to L1 via StorageManager and wait for L2 store."""
        ret = sm.reserve_write(keys, layout, mode="new")
        assert len(ret) == len(keys)
        sm.finish_write(list(ret.keys()))

        # Wait for StoreController to propagate all keys to L2
        adapter = sm._l2_adapters[0]
        ok = wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for k in keys),  # type: ignore
            timeout=10.0,
        )
        assert ok, "Keys should be stored in L2 by StoreController"

    def test_prefetch_from_l2(self, l2_storage_manager_config, basic_layout):
        """Write to L1 → store to L2 → clear L1 → prefetch from L2."""
        sm = StorageManager(l2_storage_manager_config)
        keys = [make_object_key(i) for i in range(5)]

        self._write_keys_and_wait_for_l2(sm, keys, basic_layout)

        # Brief sleep to let StoreController release read locks
        # after L2 store completion, then clear L1
        time.sleep(0.05)
        sm.clear()
        used, _ = sm._l1_manager.get_memory_usage()
        assert used == 0, f"L1 should be empty after clear, but {used} bytes used"

        # Prefetch — L1 has 0 hits, L2 should have all 5
        handle = sm.submit_prefetch_task(keys, basic_layout)
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 5, f"Expected 5 hits from L2, got {hit_count}"

        # Verify keys are read-locked in L1 after prefetch
        with sm.read_prefetched_results(keys) as objs:
            assert objs is not None
            assert len(objs) == len(keys)

        sm.finish_read_prefetched(keys)
        sm.close()

    def test_prefetch_mixed_l1_l2(self, l2_storage_manager_config, basic_layout):
        """Some keys in L1, rest in L2 → combined prefix hits."""
        sm = StorageManager(l2_storage_manager_config)

        # Use distinct hash ranges for L1-only vs L2 keys
        all_keys = [make_object_key(i) for i in range(5)]
        l2_only_keys = all_keys[2:]  # keys 2, 3, 4 only in L2

        # Write all keys to L1 (StoreController will push to L2)
        self._write_keys_and_wait_for_l2(sm, all_keys, basic_layout)

        # Delete only keys 2, 3, 4 from L1 so they must come from L2
        sm._l1_manager.delete(l2_only_keys)

        # Prefetch all 5 keys: first 2 from L1, next 3 from L2
        handle = sm.submit_prefetch_task(all_keys, basic_layout)
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 5, (
            f"Expected 5 combined hits (2 L1 + 3 L2), got {hit_count}"
        )

        # Clean up read locks
        sm.finish_read_prefetched(all_keys)
        sm.close()

    def test_prefetch_nothing_in_l2(self, l2_storage_manager_config, basic_layout):
        """Prefetch keys not in L2 → returns 0 L2 hits."""
        sm = StorageManager(l2_storage_manager_config)

        # Don't write anything — keys exist nowhere
        keys = [make_object_key(i) for i in range(3)]

        handle = sm.submit_prefetch_task(keys, basic_layout)
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 0, f"Expected 0 hits, got {hit_count}"

        sm.close()

    def test_prefetch_l2_partial_prefix(self, l2_storage_manager_config, basic_layout):
        """L2 has keys {0,1,3,4} but not 2 → L2 returns prefix of 2."""
        sm = StorageManager(l2_storage_manager_config)

        all_keys = [make_object_key(i) for i in range(5)]
        # Write only keys 0, 1, 3, 4 (skip key 2)
        keys_to_write = [all_keys[i] for i in [0, 1, 3, 4]]
        self._write_keys_and_wait_for_l2(sm, keys_to_write, basic_layout)

        # Brief sleep to let StoreController release read locks
        # after L2 store completion, then clear L1
        time.sleep(0.05)
        sm.clear()
        used, _ = sm._l1_manager.get_memory_usage()
        assert used == 0, f"L1 should be empty after clear, but {used} bytes used"

        handle = sm.submit_prefetch_task(all_keys, basic_layout)
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None, "Prefetch should complete"
        assert hit_count == 2, (
            f"Expected 2 prefix hits from L2 (gap at index 2), got {hit_count}"
        )

        # Only prefix keys {0, 1} should be readable
        with sm.read_prefetched_results(all_keys[:2]) as objs:
            assert objs is not None
            assert len(objs) == 2

        sm.finish_read_prefetched(all_keys[:2])
        sm.close()

    def test_prefetch_l1_prefix_plus_l2_continuation(
        self, l2_storage_manager_config, basic_layout
    ):
        """L1 has keys {0,1}, L2 has {2,3,4} → combined prefix of 5."""
        sm = StorageManager(l2_storage_manager_config)

        all_keys = [make_object_key(i) for i in range(5)]

        # Write all keys → StoreController stores all to L2
        self._write_keys_and_wait_for_l2(sm, all_keys, basic_layout)

        # Delete keys {2,3,4} from L1 only, keeping them in L2
        sm._l1_manager.delete(all_keys[2:])

        # Prefetch: L1 prefix hits = 2 (keys 0,1), L2 loads {2,3,4} → total = 5
        handle = sm.submit_prefetch_task(all_keys, basic_layout)
        hit_count = wait_for_prefetch_status(sm, handle)

        assert hit_count is not None
        assert hit_count == 5, f"Expected 5 total hits (2 L1 + 3 L2), got {hit_count}"

        sm.finish_read_prefetched(all_keys)
        sm.close()

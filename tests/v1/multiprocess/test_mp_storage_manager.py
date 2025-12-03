# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.mp_storage_manager import (
    MemoryExhaustedError,
    MPStorageManager,
)


# Fixtures
@pytest.fixture
def storage_manager():
    """Create a storage manager with 1GB buffer for testing."""
    manager = MPStorageManager(cpu_buffer_size=1.0)
    yield manager
    # Cleanup after test
    manager.close()


@pytest.fixture
def small_storage_manager():
    """Create a storage manager with very small buffer to test memory exhaustion."""
    manager = MPStorageManager(cpu_buffer_size=0.001)  # 1MB
    yield manager
    # Cleanup after test
    manager.close()


@pytest.fixture
def test_keys():
    """Create a list of test keys."""
    return [IPCCacheEngineKey.from_int_hash("model1", 1, 0, i) for i in range(10)]


@pytest.fixture
def test_shape():
    """Standard test shape for tensors."""
    return (2, 16, 16, 128)


@pytest.fixture
def test_dtype():
    """Standard test dtype for tensors."""
    return torch.float16


@pytest.fixture
def test_format():
    """Standard test memory format."""
    return MemoryFormat.KV_2LTD


# Tests for __init__
class TestInit:
    def test_initialization(self):
        """Test that storage manager initializes correctly."""
        manager = MPStorageManager(cpu_buffer_size=1.0)
        assert manager is not None
        manager.close()

    def test_initialization_with_different_sizes(self):
        """Test initialization with various buffer sizes."""
        sizes = [0.1, 0.5, 1.0, 2.0]
        for size in sizes:
            manager = MPStorageManager(cpu_buffer_size=size)
            assert manager is not None
            manager.close()


# Tests for reserve()
class TestReserve:
    def test_reserve_basic(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test basic reservation of memory objects."""
        keys = test_keys[:3]
        handle, reserved_dict = storage_manager.reserve(
            keys, test_shape, test_dtype, test_format
        )

        # Verify return types
        assert isinstance(reserved_dict, dict)

        # All keys should be allocated
        assert len(reserved_dict) == len(keys)

        # All requested keys should be in the reserved dict
        for key in keys:
            assert key in reserved_dict
            assert reserved_dict[key] is not None

    def test_reserve_incremental_handles(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that reserve handles are unique."""
        handle1, _ = storage_manager.reserve(
            [test_keys[0]], test_shape, test_dtype, test_format
        )
        handle2, _ = storage_manager.reserve(
            [test_keys[1]], test_shape, test_dtype, test_format
        )
        handle3, _ = storage_manager.reserve(
            [test_keys[2]], test_shape, test_dtype, test_format
        )

        # Handles should be unique
        assert handle2 != handle1
        assert handle3 != handle2

    def test_reserve_skip_already_reserved(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that already reserved keys are skipped."""
        keys = test_keys[:3]

        # First reservation
        handle1, reserved_dict1 = storage_manager.reserve(
            keys, test_shape, test_dtype, test_format
        )
        assert len(reserved_dict1) == len(keys)

        # Second reservation with same keys
        handle2, reserved_dict2 = storage_manager.reserve(
            keys, test_shape, test_dtype, test_format
        )
        assert len(reserved_dict2) == 0  # All should be skipped, no new allocations

    def test_reserve_skip_already_committed(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that already committed keys are skipped."""
        keys = test_keys[:3]

        # Reserve and commit
        handle1, reserved_dict1 = storage_manager.reserve(
            keys, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle1)

        # Try to reserve again
        handle2, reserved_dict2 = storage_manager.reserve(
            keys, test_shape, test_dtype, test_format
        )
        assert len(reserved_dict2) == 0  # All should be skipped, no new allocations

    def test_reserve_partial_skip(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test reservation with some keys already existing."""
        # Reserve first 4 keys
        handle1, reserved_dict1 = storage_manager.reserve(
            test_keys[:4], test_shape, test_dtype, test_format
        )
        assert len(reserved_dict1) == 4

        # Try to reserve keys 2, 3, 4, 5 (keys 2 and 3 are already reserved)
        keys_mixed = test_keys[2:6]
        handle2, reserved_dict2 = storage_manager.reserve(
            keys_mixed, test_shape, test_dtype, test_format
        )

        # Only last two should be allocated (keys 4 and 5)
        assert len(reserved_dict2) == 2
        assert test_keys[4] in reserved_dict2
        assert test_keys[5] in reserved_dict2
        assert test_keys[2] not in reserved_dict2  # Already reserved
        assert test_keys[3] not in reserved_dict2  # Already reserved

    def test_reserve_empty_keys(
        self, storage_manager, test_shape, test_dtype, test_format
    ):
        """Test reservation with empty key list."""
        handle, reserved_dict = storage_manager.reserve(
            [], test_shape, test_dtype, test_format
        )
        assert len(reserved_dict) == 0

    def test_reserve_memory_exhaustion(
        self, small_storage_manager, test_keys, test_dtype, test_format
    ):
        """Test that MemoryExhaustedError is raised when memory is exhausted."""
        # Try to allocate very large tensors that exceed buffer
        large_shape = (2, 100, 1000, 1000)  # Very large shape

        with pytest.raises(MemoryExhaustedError):
            small_storage_manager.reserve(
                test_keys[:10], large_shape, test_dtype, MemoryFormat.KV_2LTD
            )

    def test_reserve_memory_eviction(
        self, small_storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        # Try to reserve and commit a lot of small tensors (total size is large)
        large_shape = (2, 50, 50, 256)  # Moderate size
        small_shape = (2, 5, 50, 256)  # Small size (1/10 of large)

        # First, reserve a single key with large shape should fail
        with pytest.raises(MemoryExhaustedError):
            small_storage_manager.reserve(
                test_keys[:1], large_shape, test_dtype, test_format
            )

        # Now, reserve and commit multiple small tensors to fill up memory
        for i in range(10):
            keys = [test_keys[i]]
            handle, _ = small_storage_manager.reserve(
                keys, small_shape, test_dtype, test_format
            )
            small_storage_manager.commit(handle)

        assert small_storage_manager.memcheck()


# Tests for commit()
class TestCommit:
    def test_commit_basic(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test basic commit operation."""
        keys = test_keys[:3]
        handle, _ = storage_manager.reserve(keys, test_shape, test_dtype, test_format)

        # Should not raise any exception
        storage_manager.commit(handle)

    def test_commit_invalid_handle(self, storage_manager):
        """Test commit with invalid handle raises RuntimeError."""
        invalid_handle = 99999

        with pytest.raises(RuntimeError):
            storage_manager.commit(invalid_handle)

    def test_commit_twice_same_handle(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that committing same handle twice raises RuntimeError."""
        keys = test_keys[:3]
        handle, _ = storage_manager.reserve(keys, test_shape, test_dtype, test_format)

        # First commit should succeed
        storage_manager.commit(handle)

        # Second commit with same handle should fail
        with pytest.raises(RuntimeError):
            storage_manager.commit(handle)


# Tests for lookup()
class TestLookup:
    def test_lookup_empty_storage(self, storage_manager, test_keys):
        """Test lookup on empty storage returns 0."""
        found = storage_manager.lookup(test_keys[:3])
        assert found == 0

    def test_lookup_after_commit(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test lookup after committing keys."""
        keys = test_keys[:5]
        handle, _ = storage_manager.reserve(keys, test_shape, test_dtype, test_format)
        storage_manager.commit(handle)

        # Lookup all keys
        found = storage_manager.lookup(keys)
        assert found == len(keys)

    def test_lookup_prefix_matching(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that lookup uses prefix matching (stops at first not found)."""
        # Commit keys 0, 1, 2
        keys_committed = test_keys[:3]
        handle, _ = storage_manager.reserve(
            keys_committed, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Lookup keys 0, 1, 2, 3, 4
        # Should return 3 (stops at key 3 which is not found)
        found = storage_manager.lookup(test_keys[:5])
        assert found == 3

    def test_lookup_with_gap_in_middle(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test lookup with gap in middle (prefix matching behavior)."""
        # Commit keys 0, 2, 4 (skipping 1 and 3)
        keys_to_commit = [test_keys[0], test_keys[2], test_keys[4]]
        handle, _ = storage_manager.reserve(
            keys_to_commit, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Lookup keys 0, 1, 2, 3, 4
        # Should return 1 (stops at key 1 which is not found)
        found = storage_manager.lookup(test_keys[:5])
        assert found == 1

    def test_lookup_only_checks_committed(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that lookup only checks committed keys, not reserved ones."""
        keys = test_keys[:3]
        # Reserve but don't commit
        storage_manager.reserve(keys, test_shape, test_dtype, test_format)

        # Lookup should return 0 (reserved keys are not visible)
        found = storage_manager.lookup(keys)
        assert found == 0

    def test_lookup_empty_keys(self, storage_manager):
        """Test lookup with empty key list."""
        found = storage_manager.lookup([])
        assert found == 0

    def test_lookup_partial_prefix(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test lookup returns correct count for partial prefix match."""
        # Commit first 3 keys
        handle, _ = storage_manager.reserve(
            test_keys[:3], test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Lookup just first 2 keys
        found = storage_manager.lookup(test_keys[:2])
        assert found == 2

    def test_lookup_lock_objects(
        self, small_storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that lookup locks the objects so they cannot be evicted."""
        small_shape = (2, 4, 50, 256)  # Small size
        # Reserve and commit multiple small tensors to fill up memory
        target_keys = test_keys[:1]
        handle, _ = small_storage_manager.reserve(
            target_keys, small_shape, test_dtype, test_format
        )
        small_storage_manager.commit(handle)

        # Now try to reserve and commit other objects
        for key in test_keys[1:]:
            handle, _ = small_storage_manager.reserve(
                [key], small_shape, test_dtype, test_format
            )
            small_storage_manager.commit(handle)

        assert small_storage_manager.memcheck()

        # Should not be able to retrieve the first object
        # because it's evicted
        with pytest.raises(RuntimeError):
            with small_storage_manager.retrieve(target_keys) as _:
                pass

        # Now reserve and commit the first object again
        handle, _ = small_storage_manager.reserve(
            target_keys, small_shape, test_dtype, test_format
        )
        small_storage_manager.commit(handle)

        # Now lookup
        found = small_storage_manager.lookup(target_keys)
        assert found == 1

        # Try to store more objects to force eviction
        for key in test_keys[1:]:
            handle, _ = small_storage_manager.reserve(
                [key], small_shape, test_dtype, test_format
            )
            small_storage_manager.commit(handle)
        assert small_storage_manager.memcheck()

        # Now retrieving the first object should work
        with small_storage_manager.retrieve(target_keys) as objects:
            assert len(objects) == 1
            assert objects[0] is not None


# Tests for retrieve()
class TestRetrieve:
    def test_retrieve_basic(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test basic retrieve operation."""
        keys = test_keys[:3]
        handle, reserved_dict = storage_manager.reserve(
            keys, test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Retrieve the objects
        with storage_manager.retrieve(keys) as objects_retrieved:
            assert len(objects_retrieved) == len(keys)
            assert all(obj is not None for obj in objects_retrieved)

    def test_retrieve_key_not_found(self, storage_manager, test_keys):
        """Test retrieve raises RuntimeError when key is not found."""
        # Try to retrieve non-existent keys
        with pytest.raises(RuntimeError):
            with storage_manager.retrieve(test_keys[:3]) as _:
                pass

    def test_retrieve_partial_keys_not_found(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test retrieve raises RuntimeError when some keys are not found."""
        # Commit first 2 keys
        handle, _ = storage_manager.reserve(
            test_keys[:2], test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle)

        # Try to retrieve 3 keys (third one doesn't exist)
        with pytest.raises(RuntimeError):
            with storage_manager.retrieve(test_keys[:3]) as _:
                pass

    def test_retrieve_reserved_not_committed(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that retrieve fails for reserved but not committed keys."""
        keys = test_keys[:3]
        # Reserve but don't commit
        storage_manager.reserve(keys, test_shape, test_dtype, test_format)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            with storage_manager.retrieve(keys) as _:
                pass

    def test_retrieve_empty_keys(self, storage_manager):
        """Test retrieve with empty key list."""
        with storage_manager.retrieve([]) as objects:
            assert len(objects) == 0

    def test_retrieve_returns_same_objects(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that multiple retrieves return the same objects."""
        keys = test_keys[:3]
        handle, _ = storage_manager.reserve(keys, test_shape, test_dtype, test_format)
        storage_manager.commit(handle)

        # Retrieve twice
        with storage_manager.retrieve(keys) as objects1:
            with storage_manager.retrieve(keys) as objects2:
                # Should be the same objects
                assert len(objects1) == len(objects2)
                for obj1, obj2 in zip(objects1, objects2, strict=False):
                    assert obj1 is obj2

    def test_retrieve_cleared(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test that retrieve fails after objects are cleared (if applicable)."""
        keys = test_keys[:3]
        handle, _ = storage_manager.reserve(keys, test_shape, test_dtype, test_format)
        storage_manager.commit(handle)

        # Retrieve once successfully
        with storage_manager.retrieve(keys) as objects:
            assert len(objects) == len(keys)

        # clear objects
        storage_manager.clear()

        # Attempt to retrieve again should fail
        with pytest.raises(RuntimeError):
            with storage_manager.retrieve(keys) as objects:
                pass


# Tests for prefetch()
class TestPrefetch:
    def test_prefetch_not_implemented(self, storage_manager, test_keys):
        """Test that prefetch raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            storage_manager.prefetch(test_keys[:3])


# Tests for close()
class TestClose:
    def test_close_basic(self):
        """Test that close can be called successfully."""
        manager = MPStorageManager(cpu_buffer_size=0.5)
        manager.close()

    def test_close_after_operations(self, test_keys, test_dtype, test_format):
        """Test that close works after performing operations."""
        manager = MPStorageManager(cpu_buffer_size=1.0)
        shape = (2, 16, 16, 128)

        # Perform some operations
        handle, _ = manager.reserve(test_keys[:3], shape, test_dtype, test_format)
        manager.commit(handle)
        manager.lookup(test_keys[:3])
        with manager.retrieve(test_keys[:3]) as _:
            pass

        # Close should work without issues
        manager.close()


# Integration tests
class TestIntegration:
    def test_full_workflow(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test complete workflow: reserve -> commit -> lookup -> retrieve."""
        keys = test_keys[:5]

        # Reserve
        handle, reserved_dict = storage_manager.reserve(
            keys, test_shape, test_dtype, test_format
        )
        assert len(reserved_dict) == len(keys)
        for key in keys:
            assert key in reserved_dict

        # Commit
        storage_manager.commit(handle)

        # Lookup
        found = storage_manager.lookup(keys)
        assert found == len(keys)

        # Retrieve
        with storage_manager.retrieve(keys) as retrieved_objects:
            assert len(retrieved_objects) == len(keys)

    def test_multiple_reserve_commit_cycles(
        self, storage_manager, test_keys, test_shape, test_dtype, test_format
    ):
        """Test multiple reserve-commit cycles."""
        # First cycle
        handle1, _ = storage_manager.reserve(
            test_keys[:3], test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle1)

        # Second cycle
        handle2, _ = storage_manager.reserve(
            test_keys[3:6], test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle2)

        # Third cycle
        handle3, _ = storage_manager.reserve(
            test_keys[6:9], test_shape, test_dtype, test_format
        )
        storage_manager.commit(handle3)

        # Verify all keys are retrievable
        with storage_manager.retrieve(test_keys[:9]) as all_objects:
            assert len(all_objects) == 9


# Multi-threaded tests
class TestThreadSafety:
    def test_concurrent_reserves(self, storage_manager, test_dtype, test_format):
        """Test concurrent reserve operations from multiple threads."""
        num_threads = 10
        keys_per_thread = 5
        shape = (2, 10, 16, 64)

        def reserve_keys(thread_id):
            keys = [
                IPCCacheEngineKey.from_int_hash("model1", 1, 0, thread_id * 100 + i)
                for i in range(keys_per_thread)
            ]
            handle, reserved_dict = storage_manager.reserve(
                keys, shape, test_dtype, test_format
            )
            return handle, reserved_dict

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reserve_keys, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # All operations should succeed
        assert len(results) == num_threads

        # All handles should be unique
        handles = [r[0] for r in results]
        assert len(set(handles)) == num_threads

        # All reserved dicts should have the correct number of keys
        for handle, reserved_dict in results:
            assert len(reserved_dict) == keys_per_thread

    def test_concurrent_reserve_and_commit(
        self, storage_manager, test_dtype, test_format
    ):
        """Test concurrent reserve and commit operations."""
        num_threads = 10
        shape = (2, 10, 16, 64)

        def reserve_and_commit(thread_id):
            keys = [
                IPCCacheEngineKey.from_int_hash("model1", 1, 0, thread_id * 100 + i)
                for i in range(5)
            ]
            handle, reserved_dict = storage_manager.reserve(
                keys, shape, test_dtype, MemoryFormat.KV_2LTD
            )
            storage_manager.commit(handle)
            return keys

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(reserve_and_commit, i) for i in range(num_threads)
            ]
            all_keys = [f.result() for f in as_completed(futures)]

        # Verify all keys are committed and retrievable
        for keys in all_keys:
            found = storage_manager.lookup(keys)
            assert found == len(keys)
            with storage_manager.retrieve(keys) as objects:
                assert len(objects) == len(keys)

    def test_concurrent_reserve_same_keys(
        self, storage_manager, test_keys, test_dtype, test_format
    ):
        """Test concurrent reserve operations with same keys
        (should skip duplicates).
        """
        num_threads = 10
        shape = (2, 10, 16, 64)
        keys = test_keys[:5]  # Same keys for all threads

        def reserve_keys():
            handle, reserved_dict = storage_manager.reserve(
                keys, shape, test_dtype, test_format
            )
            allocated_count = len(reserved_dict)
            return handle, allocated_count

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reserve_keys) for _ in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # Total number of allocated objects across all threads should equal len(keys)
        # (since duplicate keys should be skipped)
        total_allocated = sum(r[1] for r in results)
        assert total_allocated == len(keys)

        # At least one thread should have allocated all keys
        assert any(r[1] == len(keys) for r in results)

    def test_concurrent_lookup_and_retrieve(
        self, storage_manager, test_keys, test_dtype, test_format
    ):
        """Test concurrent lookup and retrieve operations."""
        # First, commit some keys
        keys = test_keys[:10]
        shape = (2, 10, 16, 64)
        handle, _ = storage_manager.reserve(keys, shape, test_dtype, test_format)
        storage_manager.commit(handle)

        num_threads = 20

        def lookup_and_retrieve():
            # Lookup
            found = storage_manager.lookup(keys[:5])
            # Retrieve
            with storage_manager.retrieve(keys[:5]) as objects:
                return found, len(objects)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(lookup_and_retrieve) for _ in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # All operations should succeed and return consistent results
        for found, retrieved_count in results:
            assert found == 5
            assert retrieved_count == 5

    def test_interleaved_reserve_commit_lookup(
        self, storage_manager, test_dtype, test_format
    ):
        """Test interleaved reserve, commit, and lookup operations."""
        num_threads = 10
        shape = (2, 10, 16, 64)
        barrier = threading.Barrier(num_threads)

        def thread_operation(thread_id):
            # Create unique keys for this thread
            keys = [
                IPCCacheEngineKey.from_int_hash("model1", 1, 0, thread_id * 100 + i)
                for i in range(3)
            ]

            # Reserve
            handle, reserved_dict = storage_manager.reserve(
                keys, shape, test_dtype, test_format
            )

            # Wait for all threads to complete reserve
            barrier.wait()

            # Commit
            storage_manager.commit(handle)

            # Wait for all threads to complete commit
            barrier.wait()

            # Lookup
            found = storage_manager.lookup(keys)

            return len(keys), found

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_operation, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # All lookups should find their keys
        for expected, found in results:
            assert found == expected

    def test_stress_test_many_operations(
        self, storage_manager, test_dtype, test_format
    ):
        """Stress test with many concurrent operations."""
        num_threads = 50
        operations_per_thread = 10
        shape = (2, 10, 16, 32)

        def perform_operations(thread_id):
            for i in range(operations_per_thread):
                keys = [
                    IPCCacheEngineKey.from_int_hash(
                        "model1", 1, 0, thread_id * 1000 + i * 10 + j
                    )
                    for j in range(3)
                ]

                # Reserve
                handle, reserved_dict = storage_manager.reserve(
                    keys, shape, test_dtype, test_format
                )

                # Commit
                storage_manager.commit(handle)

                # Lookup
                found = storage_manager.lookup(keys)
                assert found == len(keys)

                # Retrieve
                with storage_manager.retrieve(keys) as retrieved:
                    assert len(retrieved) == len(keys)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(perform_operations, i) for i in range(num_threads)
            ]
            for f in as_completed(futures):
                f.result()  # Will raise exception if any operation failed

    def test_concurrent_handle_allocation(
        self, storage_manager, test_dtype, test_format
    ):
        """Test that reserve handles are allocated correctly under concurrency."""
        num_threads = 100
        shape = (2, 10, 16, 32)

        def get_handle(thread_id):
            keys = [IPCCacheEngineKey.from_int_hash("model1", 1, 0, thread_id)]
            handle, _ = storage_manager.reserve(keys, shape, test_dtype, test_format)
            return handle

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_handle, i) for i in range(num_threads)]
            handles = [f.result() for f in as_completed(futures)]

        # All handles should be unique
        assert len(set(handles)) == num_threads

        # Handles should be in range [0, num_threads)
        assert all(0 <= h < num_threads for h in handles)

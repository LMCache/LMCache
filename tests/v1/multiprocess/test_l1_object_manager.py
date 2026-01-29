# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for L1ObjectManager.

These tests verify the behavior of L1ObjectManager as described in the
interface docstrings. The tests focus on black-box testing without
accessing private members.

Test Coverage:
1. prereserve_forced() - Thread-safe pre-reservation with FORCED semantics
2. postreserve_must() - Post-reservation with ALL-OR-NOTHING semantics
3. cancel_prereserve_forced() - Cancellation with FORCED semantics

Future tests can be added for:
- mark_reserved()
- commit()
- lock() / unlock()
- lookup_and_lock()
- query_states()
- delete_committed()
- drop_reserved()
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata

if TYPE_CHECKING:
    from lmcache.v1.memory_management import MemoryAllocatorInterface

# First Party
from lmcache.v1.multiprocess.distributed.api import ObjectKey
from lmcache.v1.multiprocess.distributed.config import L1ObjectManagerConfig
from lmcache.v1.multiprocess.distributed.error import L1ObjectManagerError, strerror
from lmcache.v1.multiprocess.distributed.internal_api import L1OperationResult
from lmcache.v1.multiprocess.distributed.object_manager import (
    L1ObjectManager,
)

# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockMemoryObj(MemoryObj):
    """
    A mock implementation of MemoryObj for testing purposes.

    This class provides a minimal implementation of the MemoryObj interface
    that can be used in unit tests without requiring actual memory allocation.
    """

    def __init__(self, obj_id: int = 0):
        """Initialize the mock memory object with a unique identifier."""
        metadata = MemoryObjMetadata(
            shape=torch.Size([2, 32, 256, 1024]),
            dtype=torch.bfloat16,
            address=0,
            phy_size=32 << 20,
            ref_count=1,
        )
        super().__init__(metadata)
        self.obj_id = obj_id
        self._valid = True
        self._ref_count = 0
        self._pinned = False

    @property
    def raw_data(self):
        return None

    def invalidate(self):
        self._valid = False

    def is_valid(self):
        return self._valid

    def get_size(self) -> int:
        return 1024

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> Optional[torch.dtype]:
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        return [self.meta.shape]

    def get_dtypes(self) -> list[torch.dtype]:
        return [self.meta.dtype]

    def get_memory_format(self) -> MemoryFormat:
        return self.meta.fmt

    def get_physical_size(self) -> int:
        return 1024

    def pin(self) -> bool:
        self._pinned = True
        return True

    def ref_count_up(self):
        self._ref_count += 1

    def unpin(self) -> bool:
        self._pinned = False
        return True

    def ref_count_down(self):
        self._ref_count -= 1

    def get_ref_count(self) -> int:
        return self._ref_count

    def get_num_tokens(self) -> int:
        return 0

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return None

    @property
    def byte_array(self) -> bytes:
        return b"\x00" * 1024

    @property
    def data_ptr(self) -> int:
        return 0

    @property
    def is_pinned(self) -> bool:
        return self._pinned

    @property
    def can_evict(self) -> bool:
        return not self._pinned and self._ref_count == 0

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        return None


# =============================================================================
# Helper Functions
# =============================================================================


def create_object_key(
    chunk_hash: int, model_name: str = "test_model", kv_rank: int = 0
) -> ObjectKey:
    """Create an ObjectKey for testing."""
    return ObjectKey(chunk_hash=chunk_hash, model_name=model_name, kv_rank=kv_rank)


def create_object_keys(
    count: int, model_name: str = "test_model", kv_rank: int = 0
) -> list[ObjectKey]:
    """Create a list of unique ObjectKeys for testing."""
    return [create_object_key(i, model_name, kv_rank) for i in range(count)]


def create_mock_memory_objs(count: int) -> list[MockMemoryObj]:
    """Create a list of mock memory objects for testing."""
    return [MockMemoryObj(obj_id=i) for i in range(count)]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a basic L1ObjectManagerConfig for testing."""
    return L1ObjectManagerConfig()


@pytest.fixture
def manager(config):
    """Create a fresh L1ObjectManager instance for each test."""
    return L1ObjectManager(config)


@pytest.fixture
def keys_3():
    """Create 3 unique ObjectKeys for testing."""
    return create_object_keys(3)


@pytest.fixture
def keys_5():
    """Create 5 unique ObjectKeys for testing."""
    return create_object_keys(5)


@pytest.fixture
def keys_10():
    """Create 10 unique ObjectKeys for testing."""
    return create_object_keys(10)


@pytest.fixture
def memory_objs_3():
    """Create 3 mock memory objects for testing."""
    return create_mock_memory_objs(3)


@pytest.fixture
def memory_objs_5():
    """Create 5 mock memory objects for testing."""
    return create_mock_memory_objs(5)


# =============================================================================
# Base Test Class for Common Patterns
# =============================================================================


class L1ObjectManagerTestBase:
    """
    Base class for L1ObjectManager tests providing common utilities.

    Subclasses can use these methods for consistent test patterns.
    """

    # =========================================================================
    # Result Assertion Helpers
    # =========================================================================

    @staticmethod
    def assert_result_successful(
        result: L1OperationResult, expected_success_count: int
    ):
        """Assert that the result is successful with expected number of success keys."""
        assert result.is_successful(), (
            f"Expected success but got error: {strerror(result.error)}"
        )
        assert len(result.success_keys) == expected_success_count
        assert len(result.failed_keys) == 0

    @staticmethod
    def assert_result_has_error(
        result: L1OperationResult, expected_error: L1ObjectManagerError
    ):
        """Assert that the result contains the expected error."""
        assert not result.is_successful()
        assert result.error.has_error(expected_error), (
            f"Expected error {expected_error} but got {strerror(result.error)}"
        )

    @staticmethod
    def assert_keys_in_success(result: L1OperationResult, keys: list[ObjectKey]):
        """Assert that all specified keys are in success_keys."""
        for key in keys:
            assert key in result.success_keys, f"Key {key} not in success_keys"

    @staticmethod
    def assert_keys_in_failed(result: L1OperationResult, keys: list[ObjectKey]):
        """Assert that all specified keys are in failed_keys."""
        for key in keys:
            assert key in result.failed_keys, f"Key {key} not in failed_keys"

    # =========================================================================
    # State Verification Helpers (using query_states)
    # =========================================================================

    @staticmethod
    def assert_keys_not_exist(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys do not exist in the manager."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            assert not state.exists(), f"Key {key} should not exist but does"

    @staticmethod
    def assert_keys_exist(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys exist in the manager."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            assert state.exists(), f"Key {key} should exist but doesn't"

    @staticmethod
    def assert_keys_reserved(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys are in reserved state."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            assert state.exists(), f"Key {key} should exist but doesn't"
            assert state.is_reserved(), f"Key {key} should be reserved but is not"
            assert not state.is_committed(), f"Key {key} should not be committed"

    @staticmethod
    def assert_keys_committed(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys are in committed state."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            assert state.exists(), f"Key {key} should exist but doesn't"
            assert state.is_committed(), f"Key {key} should be committed but is not"
            assert not state.is_reserved(), f"Key {key} should not be reserved"

    @staticmethod
    def assert_keys_have_memory_obj(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys have associated memory objects."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            assert state.exists(), f"Key {key} should exist but doesn't"
            assert state.memory_obj is not None, f"Key {key} should have memory_obj"

    @staticmethod
    def assert_keys_no_memory_obj(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys do not have associated memory objects."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            # Key may or may not exist, but if it does, it should have no memory_obj
            if state.exists():
                assert state.memory_obj is None, f"Key {key} should not have memory_obj"

    @staticmethod
    def assert_keys_locked(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys are locked."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            assert state.exists(), f"Key {key} should exist but doesn't"
            assert state.is_locked(), f"Key {key} should be locked but is not"

    @staticmethod
    def assert_keys_unlocked(manager: L1ObjectManager, keys: list[ObjectKey]):
        """Assert that all specified keys are not locked."""
        states = manager.query_states(keys)
        for key, state in zip(keys, states, strict=False):
            assert state.exists(), f"Key {key} should exist but doesn't"
            assert not state.is_locked(), f"Key {key} should not be locked"


# =============================================================================
# Tests for L1ObjectManager.prereserve_forced()
# =============================================================================


class TestPrereserveForced(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.prereserve_forced() method.

    Per the docstring:
    - Thread-safe function to pre-reserve keys without memory objects
    - Uses FORCED semantics: skips keys that are already committed or reserved
    - Returns L1OperationResult with success_keys and failed_keys
    - error is SUCCESS if all keys reserved, KEYS_ALREADY_EXIST if some exist
    """

    def test_prereserve_single_key_success(self, manager, keys_3):
        """Test pre-reserving a single key successfully."""
        key = keys_3[0]

        # Verify key doesn't exist before
        self.assert_keys_not_exist(manager, [key])

        result = manager.prereserve_forced([key])

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key is now reserved (without memory obj)
        self.assert_keys_reserved(manager, [key])
        self.assert_keys_no_memory_obj(manager, [key])

    def test_prereserve_multiple_keys_success(self, manager, keys_5):
        """Test pre-reserving multiple keys successfully."""
        result = manager.prereserve_forced(keys_5)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys are reserved
        self.assert_keys_reserved(manager, keys_5)

    def test_prereserve_empty_keys_success(self, manager):
        """Test pre-reserving empty key list returns success."""
        result = manager.prereserve_forced([])

        self.assert_result_successful(result, expected_success_count=0)

    def test_prereserve_skips_already_reserved_keys(self, manager, keys_5):
        """Test that pre-reserving already reserved keys returns KEYS_ALREADY_EXIST."""
        # First reservation should succeed
        result1 = manager.prereserve_forced(keys_5)
        self.assert_result_successful(result1, expected_success_count=5)

        # Second reservation of same keys should fail
        result2 = manager.prereserve_forced(keys_5)

        self.assert_result_has_error(result2, L1ObjectManagerError.KEYS_ALREADY_EXIST)
        assert len(result2.success_keys) == 0
        assert len(result2.failed_keys) == 5
        self.assert_keys_in_failed(result2, keys_5)

    def test_prereserve_partial_overlap_forced_semantics(self, manager, keys_5):
        """
        Test FORCED semantics: reserves new keys and skips existing ones.

        Per docstring: "This function will skip the keys that are already
        committed or reserved."
        """
        # Reserve first 3 keys
        first_keys = keys_5[:3]
        result1 = manager.prereserve_forced(first_keys)
        self.assert_result_successful(result1, expected_success_count=3)

        # Try to reserve all 5 keys (first 3 already exist)
        result2 = manager.prereserve_forced(keys_5)

        # Should have partial success (only 2 new keys succeed)
        assert len(result2.success_keys) == 2
        assert len(result2.failed_keys) == 3
        self.assert_keys_in_success(result2, keys_5[3:])
        self.assert_keys_in_failed(result2, first_keys)
        self.assert_result_has_error(result2, L1ObjectManagerError.KEYS_ALREADY_EXIST)
        self.assert_keys_reserved(manager, keys_5)

    def test_prereserve_different_models_same_hash(self, manager):
        """Test that keys with same hash but different model names are distinct."""
        key1 = create_object_key(chunk_hash=100, model_name="model_a")
        key2 = create_object_key(chunk_hash=100, model_name="model_b")

        result1 = manager.prereserve_forced([key1])
        result2 = manager.prereserve_forced([key2])

        self.assert_result_successful(result1, expected_success_count=1)
        self.assert_result_successful(result2, expected_success_count=1)

    def test_prereserve_different_kv_rank_same_hash(self, manager):
        """Test that keys with same hash but different kv_rank are distinct."""
        key1 = create_object_key(chunk_hash=100, kv_rank=0)
        key2 = create_object_key(chunk_hash=100, kv_rank=1)

        result1 = manager.prereserve_forced([key1])
        result2 = manager.prereserve_forced([key2])

        self.assert_result_successful(result1, expected_success_count=1)
        self.assert_result_successful(result2, expected_success_count=1)

    def test_prereserve_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        # Reserve all keys first
        manager.prereserve_forced(keys_3)

        # Try to reserve again
        result = manager.prereserve_forced(keys_3)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_ALREADY_EXIST


class TestPrereserveForcedThreadSafety(L1ObjectManagerTestBase):
    """
    Thread-safety tests for prereserve_forced().

    Per the docstring: "When multiple threads trying to reserve on the same
    set of keys key, the expected behavior is each of the thread will get
    some of the keys."
    """

    def test_prereserve_concurrent_disjoint_keys(self, manager):
        """Test concurrent pre-reservation of disjoint key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        def reserve_task(thread_id):
            try:
                # Each thread gets its own unique set of keys
                keys = create_object_keys(
                    keys_per_thread, model_name=f"model_{thread_id}"
                )
                result = manager.prereserve_forced(keys)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=reserve_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All reservations should succeed (disjoint keys)
        for result in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )

    def test_prereserve_concurrent_same_keys_partitioned(self, manager):
        """
        Test concurrent pre-reservation of same keys.

        Per docstring: "When multiple threads trying to reserve on the same
        set of keys, the expected behavior is each of the thread will get
        some of the keys."
        """
        num_threads = 10
        shared_keys = create_object_keys(100)
        results = []
        exceptions = []
        lock = threading.Lock()

        def reserve_task():
            try:
                result = manager.prereserve_forced(shared_keys)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [threading.Thread(target=reserve_task) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # Total success keys across all threads should equal total unique keys
        total_success = sum(len(r.success_keys) for r in results)
        assert total_success == len(shared_keys), (
            f"Expected {len(shared_keys)} total successes, got {total_success}"
        )

        # Each key should be successfully reserved exactly once
        all_success_keys = []
        for r in results:
            all_success_keys.extend(r.success_keys)
        assert len(set(all_success_keys)) == len(all_success_keys), (
            "Some keys were reserved multiple times"
        )
        self.assert_keys_reserved(manager, shared_keys)

    def test_prereserve_high_contention(self, manager):
        """Test pre-reservation under high thread contention."""
        num_threads = 20
        iterations = 50
        exceptions = []
        lock = threading.Lock()

        def reserve_task(thread_id):
            try:
                for i in range(iterations):
                    key = create_object_key(
                        chunk_hash=i, model_name=f"model_{thread_id}_{i}"
                    )
                    manager.prereserve_forced([key])
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(reserve_task, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"


# =============================================================================
# Tests for L1ObjectManager.postreserve_must()
# =============================================================================


class TestPostreserveMust(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.postreserve_must() method.

    Per the docstring:
    - Thread-safe function to post-reserve keys with memory objects
    - Uses ALL-OR-NOTHING semantics: if any key fails, no objects are associated
    - Keys should be already pre-reserved
    - Returns KEYS_NOT_RESERVED if keys are not in reserved state
    - Returns ENTRY_NOT_EMPTY if keys already have memory objects
    """

    def test_postreserve_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test post-reserving a single pre-reserved key successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Pre-reserve first
        prereserve_result = manager.prereserve_forced([key])
        self.assert_result_successful(prereserve_result, expected_success_count=1)

        # Verify key is reserved without memory obj
        self.assert_keys_reserved(manager, [key])
        self.assert_keys_no_memory_obj(manager, [key])

        # Post-reserve with memory object
        result = manager.postreserve_must([key], [obj])

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key is still reserved but now has memory obj
        self.assert_keys_reserved(manager, [key])
        self.assert_keys_have_memory_obj(manager, [key])

    def test_postreserve_multiple_keys_success(self, manager, keys_5, memory_objs_5):
        """Test post-reserving multiple pre-reserved keys successfully."""
        # Pre-reserve first
        prereserve_result = manager.prereserve_forced(keys_5)
        self.assert_result_successful(prereserve_result, expected_success_count=5)

        # Post-reserve with memory objects
        result = manager.postreserve_must(keys_5, memory_objs_5)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)
        self.assert_keys_reserved(manager, keys_5)
        self.assert_keys_have_memory_obj(manager, keys_5)

    def test_postreserve_empty_keys_success(self, manager):
        """Test post-reserving empty key list returns success."""
        result = manager.postreserve_must([], [])

        self.assert_result_successful(result, expected_success_count=0)

    def test_postreserve_not_prereserved_fails(self, manager, keys_3, memory_objs_3):
        """Test that post-reserving non-prereserved keys fails."""
        # Don't pre-reserve, directly try to post-reserve
        result = manager.postreserve_must(keys_3, memory_objs_3)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)
        assert len(result.success_keys) == 0
        self.assert_keys_not_exist(manager, keys_3)

    def test_postreserve_partial_prereserved_all_or_nothing(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test ALL-OR-NOTHING semantics: if one key fails, none are post-reserved.

        Per docstring: "If any key fails, the function will not associate
        any memory objects with the keys."
        """
        # Only pre-reserve first 3 keys
        prereserve_result = manager.prereserve_forced(keys_5[:3])
        self.assert_result_successful(prereserve_result, expected_success_count=3)
        self.assert_keys_reserved(manager, keys_5[:3])
        self.assert_keys_no_memory_obj(manager, keys_5[:3])

        # Try to post-reserve all 5 keys (last 2 are not pre-reserved)
        result = manager.postreserve_must(keys_5, memory_objs_5)

        # Should fail (ALL-OR-NOTHING)
        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)

        # No keys should be in success_keys due to rollback
        # The first 3 keys were processed but should have been rolled back
        assert len(result.success_keys) == 0 or len(result.failed_keys) > 0
        self.assert_keys_reserved(manager, keys_5[:3])
        self.assert_keys_no_memory_obj(manager, keys_5[:3])
        self.assert_keys_not_exist(manager, keys_5[3:])

    def test_postreserve_already_has_object_fails(self, manager, keys_3, memory_objs_3):
        """
        Test that post-reserving keys that already have memory objects fails.

        Per docstring: "Returns ENTRY_NOT_EMPTY if some keys are already
        associated with memory objects."
        """
        key = keys_3[0]
        obj1 = memory_objs_3[0]
        obj2 = memory_objs_3[1]

        # Pre-reserve and post-reserve once
        manager.prereserve_forced([key])
        result1 = manager.postreserve_must([key], [obj1])
        self.assert_result_successful(result1, expected_success_count=1)

        # Try to post-reserve again with different object
        result2 = manager.postreserve_must([key], [obj2])

        self.assert_result_has_error(result2, L1ObjectManagerError.ENTRY_NOT_EMPTY)
        self.assert_keys_reserved(manager, [key])
        self.assert_keys_have_memory_obj(manager, [key])
        state = manager.query_states([key])[0]
        assert state.memory_obj is not None
        assert state.memory_obj is obj1

    def test_postreserve_rollback_on_failure(self, manager, keys_5, memory_objs_5):
        """
        Test that successful entries are rolled back when later entry fails.

        This verifies the ALL-OR-NOTHING semantics by ensuring that even
        successfully processed keys are reverted on failure.
        """
        # Pre-reserve first 3 keys only
        manager.prereserve_forced(keys_5[:3])

        # Try to post-reserve all 5 (4th will fail)
        _ = manager.postreserve_must(keys_5, memory_objs_5)

        # After rollback, first 3 keys should still be pre-reserved (no memory obj)
        # We can verify this by successfully post-reserving them again
        second_result = manager.postreserve_must(keys_5[:3], memory_objs_5[:3])
        self.assert_result_successful(second_result, expected_success_count=3)
        self.assert_keys_reserved(manager, keys_5[:3])
        self.assert_keys_have_memory_obj(manager, keys_5[:3])

    def test_postreserve_skipped_keys_on_failure(self, manager, keys_5, memory_objs_5):
        """
        Test that keys after the failed key are marked as skipped.

        Per docstring, the result should include skipped_keys for keys
        that weren't processed due to earlier failure.
        """
        # Only pre-reserve key at index 0, 1, 3, 4 (skip index 2)
        manager.prereserve_forced([keys_5[0], keys_5[1]])

        # Try to post-reserve all 5 keys - key[2] will fail, key[3:] should be skipped
        result = manager.postreserve_must(keys_5, memory_objs_5)

        # Keys after the failed one should be skipped
        # Since key[2] fails, keys[3:] should be in skipped_keys
        assert len(result.skipped_keys) > 0 or len(result.failed_keys) > 0


class TestPostreserveMustThreadSafety(L1ObjectManagerTestBase):
    """Thread-safety tests for postreserve_must()."""

    def test_postreserve_concurrent_different_keys(self, manager):
        """Test concurrent post-reservation of different key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        def reserve_and_post_task(thread_id):
            try:
                keys = create_object_keys(
                    keys_per_thread, model_name=f"model_{thread_id}"
                )
                objs = create_mock_memory_objs(keys_per_thread)

                # Pre-reserve
                manager.prereserve_forced(keys)

                # Post-reserve
                result = manager.postreserve_must(keys, objs)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=reserve_and_post_task, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All post-reservations should succeed
        for result in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )

    def test_postreserve_concurrent_same_keys_race(self, manager):
        """
        Test concurrent post-reservation of same keys (race condition).

        Only one thread should succeed in post-reserving each key.
        """
        num_threads = 10
        shared_keys = create_object_keys(5)
        results = []
        exceptions = []
        lock = threading.Lock()

        # Pre-reserve keys first (single-threaded)
        prereserve_result = manager.prereserve_forced(shared_keys)
        self.assert_result_successful(prereserve_result, expected_success_count=5)

        def post_task(thread_id):
            try:
                objs = create_mock_memory_objs(len(shared_keys))
                result = manager.postreserve_must(shared_keys, objs)
                with lock:
                    results.append((thread_id, result))
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=post_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # Only one thread should fully succeed (ALL-OR-NOTHING)
        successful_results = [r for _, r in results if r.is_successful()]
        assert len(successful_results) <= 1, (
            f"Multiple threads succeeded: {len(successful_results)}"
        )

        self.assert_keys_reserved(manager, shared_keys)
        self.assert_keys_have_memory_obj(manager, shared_keys)


# =============================================================================
# Tests for L1ObjectManager.cancel_prereserve_forced()
# =============================================================================


class TestCancelPrereserveForced(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.cancel_prereserve_forced() method.

    Per the docstring:
    - Thread-safe function to cancel pre-reservation
    - Uses FORCED semantics: skips non-reserved or committed keys
    - Returns KEYS_NOT_RESERVED if some keys are not reserved
    - Returns ENTRY_NOT_EMPTY if some keys already have memory objects
    """

    def test_cancel_single_key_success(self, manager, keys_3):
        """Test cancelling a single pre-reserved key successfully."""
        key = keys_3[0]

        # Pre-reserve first
        manager.prereserve_forced([key])

        # Verify key is reserved before cancel
        self.assert_keys_reserved(manager, [key])

        # Cancel pre-reservation
        result = manager.cancel_prereserve_forced([key])

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key no longer exists
        self.assert_keys_not_exist(manager, [key])

    def test_cancel_multiple_keys_success(self, manager, keys_5):
        """Test cancelling multiple pre-reserved keys successfully."""
        # Pre-reserve first
        manager.prereserve_forced(keys_5)

        # Verify all keys are reserved
        self.assert_keys_reserved(manager, keys_5)

        # Cancel pre-reservation
        result = manager.cancel_prereserve_forced(keys_5)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys no longer exist
        self.assert_keys_not_exist(manager, keys_5)

    def test_cancel_empty_keys_success(self, manager):
        """Test cancelling empty key list returns success."""
        result = manager.cancel_prereserve_forced([])

        self.assert_result_successful(result, expected_success_count=0)

    def test_cancel_not_reserved_fails(self, manager, keys_3):
        """Test that cancelling non-reserved keys fails."""
        # Don't pre-reserve, directly try to cancel
        result = manager.cancel_prereserve_forced(keys_3)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)
        assert len(result.success_keys) == 0
        assert len(result.failed_keys) == 3
        self.assert_keys_not_exist(manager, keys_3)

    def test_cancel_with_memory_object_fails(self, manager, keys_3, memory_objs_3):
        """
        Test that cancelling keys with memory objects fails.

        Per docstring: "If the keys are already associated with memory objects
        (i.e., committed), the cancellation will fail for those keys."
        """
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Pre-reserve and post-reserve
        manager.prereserve_forced([key])
        manager.postreserve_must([key], [obj])

        # Try to cancel (should fail because object is associated)
        result = manager.cancel_prereserve_forced([key])

        self.assert_result_has_error(result, L1ObjectManagerError.ENTRY_NOT_EMPTY)
        assert len(result.success_keys) == 0

    def test_cancel_partial_success_forced_semantics(self, manager, keys_5):
        """
        Test FORCED semantics: cancels valid keys and skips invalid ones.
        """
        # Pre-reserve first 3 keys only
        manager.prereserve_forced(keys_5[:3])

        # Try to cancel all 5 keys
        result = manager.cancel_prereserve_forced(keys_5)

        # Should have partial success
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2
        self.assert_keys_in_success(result, keys_5[:3])
        self.assert_keys_in_failed(result, keys_5[3:])
        self.assert_keys_not_exist(manager, keys_5)

    def test_cancel_mixed_states_forced_semantics(self, manager, keys_5, memory_objs_5):
        """
        Test FORCED semantics with mixed key states.

        - Some keys: pre-reserved (can cancel)
        - Some keys: pre-reserved + post-reserved (cannot cancel)
        - Some keys: not reserved (cannot cancel)
        """
        # Pre-reserve keys 0, 1, 2
        manager.prereserve_forced(keys_5[:3])

        # Post-reserve key 0 (now has memory object)
        manager.postreserve_must([keys_5[0]], [memory_objs_5[0]])

        # Try to cancel all 5 keys
        result = manager.cancel_prereserve_forced(keys_5)

        # Key 0: ENTRY_NOT_EMPTY (has memory object)
        # Keys 1, 2: SUCCESS
        # Keys 3, 4: KEYS_NOT_RESERVED
        assert len(result.success_keys) == 2
        self.assert_keys_in_success(result, [keys_5[1], keys_5[2]])
        assert len(result.failed_keys) == 3

        # Keys 0 should still be reserved and hs object
        # Other keys should not exist
        self.assert_keys_reserved(manager, [keys_5[0]])
        self.assert_keys_have_memory_obj(manager, [keys_5[0]])
        self.assert_keys_not_exist(manager, keys_5[1:])

    def test_cancel_allows_rereserve(self, manager, keys_3):
        """Test that cancelled keys can be pre-reserved again."""
        # Pre-reserve
        result1 = manager.prereserve_forced(keys_3)
        self.assert_result_successful(result1, expected_success_count=3)

        # Cancel
        cancel_result = manager.cancel_prereserve_forced(keys_3)
        self.assert_result_successful(cancel_result, expected_success_count=3)

        # Re-reserve should succeed
        result2 = manager.prereserve_forced(keys_3)
        self.assert_result_successful(result2, expected_success_count=3)

    def test_cancel_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        # Don't reserve any keys
        result = manager.cancel_prereserve_forced(keys_3)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_NOT_RESERVED


class TestCancelPrereserveForcedThreadSafety(L1ObjectManagerTestBase):
    """Thread-safety tests for cancel_prereserve_forced()."""

    def test_cancel_concurrent_disjoint_keys(self, manager):
        """Test concurrent cancellation of disjoint key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Pre-reserve all keys first (single-threaded setup)
        all_keys = []
        for i in range(num_threads):
            keys = create_object_keys(keys_per_thread, model_name=f"model_{i}")
            all_keys.append(keys)
            manager.prereserve_forced(keys)

        def cancel_task(thread_id):
            try:
                result = manager.cancel_prereserve_forced(all_keys[thread_id])
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=cancel_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All cancellations should succeed
        for result in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )

    def test_cancel_concurrent_same_keys(self, manager):
        """
        Test concurrent cancellation of same keys.

        Only one thread should successfully cancel each key.
        """
        num_threads = 10
        shared_keys = create_object_keys(20)
        results = []
        exceptions = []
        lock = threading.Lock()

        # Pre-reserve shared keys
        manager.prereserve_forced(shared_keys)

        def cancel_task():
            try:
                result = manager.cancel_prereserve_forced(shared_keys)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [threading.Thread(target=cancel_task) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # Total success cancellations should equal number of keys
        total_success = sum(len(r.success_keys) for r in results)
        assert total_success == len(shared_keys), (
            f"Expected {len(shared_keys)} total successes, got {total_success}"
        )

        self.assert_keys_not_exist(manager, shared_keys)


# =============================================================================
# Integration Tests
# =============================================================================


class TestReserveOperationsIntegration(L1ObjectManagerTestBase):
    """
    Integration tests for the complete reserve workflow:
    prereserve_forced -> postreserve_must -> cancel_prereserve_forced
    """

    def test_full_reserve_cycle(self, manager, keys_5, memory_objs_5):
        """Test complete reserve workflow: prereserve -> postreserve."""
        # Pre-reserve
        prereserve_result = manager.prereserve_forced(keys_5)
        self.assert_result_successful(prereserve_result, expected_success_count=5)

        # Post-reserve
        postreserve_result = manager.postreserve_must(keys_5, memory_objs_5)
        self.assert_result_successful(postreserve_result, expected_success_count=5)

    def test_prereserve_cancel_rereserve_cycle(self, manager, keys_3):
        """Test cycle: prereserve -> cancel -> prereserve."""
        # First cycle
        result1 = manager.prereserve_forced(keys_3)
        self.assert_result_successful(result1, expected_success_count=3)

        cancel_result = manager.cancel_prereserve_forced(keys_3)
        self.assert_result_successful(cancel_result, expected_success_count=3)

        # Second cycle
        result2 = manager.prereserve_forced(keys_3)
        self.assert_result_successful(result2, expected_success_count=3)

    def test_partial_workflow_with_cancellation(self, manager, keys_5, memory_objs_5):
        """
        Test partial workflow where some keys are cancelled before post-reserve.
        """
        # Pre-reserve all 5 keys
        prereserve_result = manager.prereserve_forced(keys_5)
        self.assert_result_successful(prereserve_result, expected_success_count=5)

        # Cancel first 2 keys
        cancel_result = manager.cancel_prereserve_forced(keys_5[:2])
        self.assert_result_successful(cancel_result, expected_success_count=2)

        # Post-reserve remaining 3 keys
        postreserve_result = manager.postreserve_must(keys_5[2:], memory_objs_5[2:])
        self.assert_result_successful(postreserve_result, expected_success_count=3)

        # Verify cancelled keys can be re-reserved
        rereserve_result = manager.prereserve_forced(keys_5[:2])
        self.assert_result_successful(rereserve_result, expected_success_count=2)

    def test_concurrent_mixed_operations(self, manager):
        """
        Test concurrent execution of mixed operations.

        Multiple threads performing prereserve, postreserve, and cancel
        operations simultaneously on different key sets.
        """
        num_threads = 4
        operations_per_thread = 10
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(operations_per_thread):
                    keys = create_object_keys(3, model_name=f"model_{thread_id}_{i}")
                    objs = create_mock_memory_objs(3)

                    # Pre-reserve
                    manager.prereserve_forced(keys)

                    # Randomly either post-reserve or cancel
                    if i % 2 == 0:
                        manager.postreserve_must(keys, objs)
                    else:
                        manager.cancel_prereserve_forced(keys)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Concurrent operation errors: {exceptions}"


# =============================================================================
# Tests for L1ObjectManager.commit()
# =============================================================================


class TestCommit(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.commit() method.

    Per the docstring:
    - Thread-safe function to change state from "reserved" to "committed"
    - Keys must be reserved AND have memory objects (post-reserved)
    - Supports both FORCED and ALL-OR-NOTHING semantics via `force` parameter
    - Returns KEYS_NOT_RESERVED if keys are not properly reserved
    """

    def test_commit_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test committing a single fully-reserved key successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Pre-reserve and post-reserve
        manager.prereserve_forced([key])
        manager.postreserve_must([key], [obj])

        # Verify key is reserved before commit
        self.assert_keys_reserved(manager, [key])

        # Commit
        result = manager.commit([key], force=False)

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key is now committed
        self.assert_keys_committed(manager, [key])
        self.assert_keys_have_memory_obj(manager, [key])

    def test_commit_multiple_keys_success(self, manager, keys_5, memory_objs_5):
        """Test committing multiple fully-reserved keys successfully."""
        # Pre-reserve and post-reserve
        manager.prereserve_forced(keys_5)
        manager.postreserve_must(keys_5, memory_objs_5)

        # Verify all keys are reserved
        self.assert_keys_reserved(manager, keys_5)

        # Commit all
        result = manager.commit(keys_5, force=False)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys are now committed
        self.assert_keys_committed(manager, keys_5)

    def test_commit_empty_keys_success(self, manager):
        """Test committing empty key list returns success."""
        result = manager.commit([], force=False)

        self.assert_result_successful(result, expected_success_count=0)

    def test_commit_not_reserved_fails(self, manager, keys_3):
        """Test that committing non-reserved keys fails."""
        result = manager.commit(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)

    def test_commit_only_prereserved_fails(self, manager, keys_3):
        """
        Test that committing pre-reserved keys without memory objects fails.

        Per docstring: Keys must be reserved AND have memory objects.
        """
        # Only pre-reserve (no post-reserve with memory object)
        manager.prereserve_forced(keys_3)

        result = manager.commit(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)

    def test_commit_all_or_nothing_semantics(self, manager, keys_5, memory_objs_5):
        """
        Test ALL-OR-NOTHING semantics: if one key fails, none are committed.
        """
        # Pre-reserve and post-reserve first 3 keys only
        manager.prereserve_forced(keys_5[:3])
        manager.postreserve_must(keys_5[:3], memory_objs_5[:3])

        # Try to commit all 5 keys (last 2 will fail)
        result = manager.commit(keys_5, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)

        # Due to rollback, no keys should be in success_keys
        assert len(result.success_keys) == 0

        # Keys after failure should be skipped
        assert len(result.skipped_keys) > 0 or len(result.failed_keys) > 0

        # Verify first 3 keys are still reserved (rollback worked)
        self.assert_keys_reserved(manager, keys_5[:3])

        # Verify last 2 keys don't exist
        self.assert_keys_not_exist(manager, keys_5[3:])

        # The first 3 keys are still reserved, so should be able to commit them
        result = manager.commit(keys_5[:3], force=False)
        self.assert_result_successful(result, expected_success_count=3)
        self.assert_keys_in_success(result, keys_5[:3])

        # Verify first 3 keys are now committed
        self.assert_keys_committed(manager, keys_5[:3])

    def test_commit_forced_semantics_partial_success(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test FORCED semantics: commits valid keys and skips invalid ones.
        """
        # Pre-reserve and post-reserve first 3 keys only
        manager.prereserve_forced(keys_5[:3])
        manager.postreserve_must(keys_5[:3], memory_objs_5[:3])

        # Commit all 5 keys with force=True
        result = manager.commit(keys_5, force=True)

        # Should have partial success (3 committed, 2 failed)
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2
        self.assert_keys_in_success(result, keys_5[:3])
        self.assert_keys_in_failed(result, keys_5[3:])

        # Verify first 3 keys are now committed
        self.assert_keys_committed(manager, keys_5[:3])

        # Verify last 2 keys don't exist
        self.assert_keys_not_exist(manager, keys_5[3:])

    def test_commit_forced_semantics_all_fail(self, manager, keys_3):
        """Test FORCED semantics when all keys fail."""
        # Don't reserve any keys
        result = manager.commit(keys_3, force=True)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)
        assert len(result.success_keys) == 0
        assert len(result.failed_keys) == 3

    def test_commit_idempotent_check(self, manager, keys_3, memory_objs_3):
        """Test that committing already-committed keys fails."""
        # Pre-reserve, post-reserve, and commit
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)
        result1 = manager.commit(keys_3, force=False)
        self.assert_result_successful(result1, expected_success_count=3)

        # Verify keys are committed
        self.assert_keys_committed(manager, keys_3)

        # Try to commit again
        result2 = manager.commit(keys_3, force=False)

        # Should fail (keys are no longer in reserved state)
        self.assert_result_has_error(result2, L1ObjectManagerError.KEYS_NOT_RESERVED)

        # Verify keys are still committed (state unchanged)
        self.assert_keys_committed(manager, keys_3)

    def test_commit_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        result = manager.commit(keys_3, force=True)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_NOT_RESERVED


class TestCommitThreadSafety(L1ObjectManagerTestBase):
    """Thread-safety tests for commit()."""

    def test_commit_concurrent_disjoint_keys(self, manager):
        """Test concurrent commit of disjoint key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        def commit_task(thread_id):
            try:
                keys = create_object_keys(
                    keys_per_thread, model_name=f"model_{thread_id}"
                )
                objs = create_mock_memory_objs(keys_per_thread)

                # Pre-reserve and post-reserve
                manager.prereserve_forced(keys)
                manager.postreserve_must(keys, objs)

                # Commit
                result = manager.commit(keys, force=False)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=commit_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All commits should succeed
        for result in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )

    def test_commit_high_contention(self, manager):
        """Test commit under high thread contention."""
        num_threads = 20
        iterations = 20
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(iterations):
                    keys = create_object_keys(2, model_name=f"model_{thread_id}_{i}")
                    objs = create_mock_memory_objs(2)

                    manager.prereserve_forced(keys)
                    manager.postreserve_must(keys, objs)
                    manager.commit(keys, force=False)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"


# =============================================================================
# Tests for L1ObjectManager.mark_reserved_must()
# =============================================================================


class TestMarkReservedMust(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.mark_reserved_must() method.

    Per the docstring:
    - Thread-safe function to change state from "committed" to "reserved"
    - Keys must be "committed", "unlocked", and not "temporary"
    - Uses ALL-OR-NOTHING semantics
    - Returns KEYS_NOT_COMMITTED, KEYS_ALREADY_LOCKED, or KEYS_ARE_TEMPORARY
    """

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful(), (
            f"Failed to prepare committed keys: {result.error}"
        )

    def test_mark_reserved_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test marking a single committed key as reserved successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare committed key
        self._prepare_committed_keys(manager, [key], [obj])

        # Verify key is committed before
        self.assert_keys_committed(manager, [key])

        # Mark as reserved
        result = manager.mark_reserved_must([key])

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key is now reserved
        self.assert_keys_reserved(manager, [key])
        self.assert_keys_have_memory_obj(manager, [key])

    def test_mark_reserved_multiple_keys_success(self, manager, keys_5, memory_objs_5):
        """Test marking multiple committed keys as reserved successfully."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        # Verify all keys are committed
        self.assert_keys_committed(manager, keys_5)

        # Mark as reserved
        result = manager.mark_reserved_must(keys_5)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys are now reserved
        self.assert_keys_reserved(manager, keys_5)

    def test_mark_reserved_empty_keys_success(self, manager):
        """Test marking empty key list as reserved returns success."""
        result = manager.mark_reserved_must([])

        self.assert_result_successful(result, expected_success_count=0)

    def test_mark_reserved_not_committed_fails(self, manager, keys_3):
        """Test that marking non-committed keys fails."""
        result = manager.mark_reserved_must(keys_3)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

    def test_mark_reserved_only_prereserved_fails(self, manager, keys_3):
        """Test that marking pre-reserved (not committed) keys fails."""
        # Only pre-reserve
        manager.prereserve_forced(keys_3)

        result = manager.mark_reserved_must(keys_3)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

    def test_mark_reserved_all_or_nothing_semantics(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test ALL-OR-NOTHING semantics: if one key fails, none are marked.
        """
        # Only commit first 3 keys
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Try to mark all 5 keys (last 2 will fail)
        result = manager.mark_reserved_must(keys_5)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

        # Due to rollback, no keys should remain in success_keys
        assert len(result.success_keys) == 0

        # Skipped keys should be present
        assert len(result.skipped_keys) > 0 or len(result.failed_keys) > 0

        # Verify first 3 keys are still committed (rollback worked)
        self.assert_keys_committed(manager, keys_5[:3])

        # Verify last 2 keys don't exist
        self.assert_keys_not_exist(manager, keys_5[3:])

    def test_mark_reserved_rollback_on_failure(self, manager, keys_5, memory_objs_5):
        """
        Test that successful entries are rolled back when later entry fails.

        After rollback, keys should still be in committed state.
        """
        # Only commit first 3 keys
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Try to mark all 5 keys (4th will fail)
        result = manager.mark_reserved_must(keys_5)
        assert not result.is_successful()

        # Verify first 3 keys are still committed (rollback worked)
        self.assert_keys_committed(manager, keys_5[:3])

        # After rollback, first 3 keys should still be committed
        # We can verify by successfully marking them again
        second_result = manager.mark_reserved_must(keys_5[:3])
        self.assert_result_successful(second_result, expected_success_count=3)

        # Verify first 3 keys are now reserved
        self.assert_keys_reserved(manager, keys_5[:3])

    def test_mark_reserved_then_commit_again(self, manager, keys_3, memory_objs_3):
        """Test that marked-reserved keys can be committed again."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)

        # Verify keys are committed
        self.assert_keys_committed(manager, keys_3)

        # Mark as reserved
        mark_result = manager.mark_reserved_must(keys_3)
        self.assert_result_successful(mark_result, expected_success_count=3)

        # Verify keys are now reserved
        self.assert_keys_reserved(manager, keys_3)

        # Commit again
        commit_result = manager.commit(keys_3, force=False)
        self.assert_result_successful(commit_result, expected_success_count=3)

        # Verify keys are committed again
        self.assert_keys_committed(manager, keys_3)

    def test_mark_reserved_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        result = manager.mark_reserved_must(keys_3)

        # Due to ALL-OR-NOTHING, only first failing key is in failed_keys
        assert len(result.failed_keys) == len(result.failed_reasons)
        if result.failed_keys:
            assert result.failed_reasons[0] == L1ObjectManagerError.KEYS_NOT_COMMITTED


class TestMarkReservedMustThreadSafety(L1ObjectManagerTestBase):
    """Thread-safety tests for mark_reserved_must()."""

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful()

    def test_mark_reserved_concurrent_disjoint_keys(self, manager):
        """Test concurrent mark_reserved of disjoint key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare all keys as committed first (single-threaded setup)
        all_keys = []
        all_objs = []
        for i in range(num_threads):
            keys = create_object_keys(keys_per_thread, model_name=f"model_{i}")
            objs = create_mock_memory_objs(keys_per_thread)
            all_keys.append(keys)
            all_objs.append(objs)
            self._prepare_committed_keys(manager, keys, objs)

        def mark_task(thread_id):
            try:
                result = manager.mark_reserved_must(all_keys[thread_id])
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=mark_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All mark operations should succeed
        for result in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )

    def test_mark_reserved_concurrent_same_keys_race(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test concurrent mark_reserved of same keys (race condition).

        Only one thread should succeed in marking each key.
        """
        num_threads = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare committed keys (single-threaded)
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        def mark_task(thread_id):
            try:
                result = manager.mark_reserved_must(keys_5)
                with lock:
                    results.append((thread_id, result))
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=mark_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # Only one thread should fully succeed (ALL-OR-NOTHING)
        successful_results = [r for _, r in results if r.is_successful()]
        assert len(successful_results) <= 1, (
            f"Multiple threads succeeded: {len(successful_results)}"
        )

    def test_mark_reserved_high_contention(self, manager):
        """Test mark_reserved under high thread contention."""
        num_threads = 10
        iterations = 10
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(iterations):
                    keys = create_object_keys(2, model_name=f"model_{thread_id}_{i}")
                    objs = create_mock_memory_objs(2)

                    # prereserve -> postreserve -> commit -> mark_reserved -> commit
                    manager.prereserve_forced(keys)
                    manager.postreserve_must(keys, objs)
                    manager.commit(keys, force=False)
                    manager.mark_reserved_must(keys)
                    manager.commit(keys, force=False)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"


# =============================================================================
# Extended Integration Tests
# =============================================================================


class TestCommitAndMarkReservedIntegration(L1ObjectManagerTestBase):
    """
    Integration tests combining commit and mark_reserved_must operations.
    """

    def test_full_state_cycle(self, manager, keys_3, memory_objs_3):
        """
        Test complete state cycle:
        prereserve -> postreserve -> commit -> mark_reserved -> commit
        """
        # Pre-reserve
        prereserve_result = manager.prereserve_forced(keys_3)
        self.assert_result_successful(prereserve_result, expected_success_count=3)

        # Post-reserve
        postreserve_result = manager.postreserve_must(keys_3, memory_objs_3)
        self.assert_result_successful(postreserve_result, expected_success_count=3)

        # Commit
        commit_result1 = manager.commit(keys_3, force=False)
        self.assert_result_successful(commit_result1, expected_success_count=3)

        # Mark as reserved (for update)
        mark_result = manager.mark_reserved_must(keys_3)
        self.assert_result_successful(mark_result, expected_success_count=3)

        # Commit again
        commit_result2 = manager.commit(keys_3, force=False)
        self.assert_result_successful(commit_result2, expected_success_count=3)

    def test_multiple_mark_reserve_commit_cycles(self, manager, keys_3, memory_objs_3):
        """Test multiple cycles of mark_reserved -> commit."""
        # Initial setup: prereserve -> postreserve -> commit
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)
        manager.commit(keys_3, force=False)

        # Multiple cycles
        for _ in range(5):
            mark_result = manager.mark_reserved_must(keys_3)
            self.assert_result_successful(mark_result, expected_success_count=3)

            commit_result = manager.commit(keys_3, force=False)
            self.assert_result_successful(commit_result, expected_success_count=3)

    def test_concurrent_commit_and_mark_reserved(self, manager):
        """
        Test concurrent commit and mark_reserved operations on different keys.
        """
        num_threads = 4
        keys_per_thread = 5
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(10):
                    keys = create_object_keys(
                        keys_per_thread, model_name=f"model_{thread_id}_{i}"
                    )
                    objs = create_mock_memory_objs(keys_per_thread)

                    # Full workflow
                    manager.prereserve_forced(keys)
                    manager.postreserve_must(keys, objs)
                    manager.commit(keys, force=False)
                    manager.mark_reserved_must(keys)
                    manager.commit(keys, force=False)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Concurrent operation errors: {exceptions}"


# =============================================================================
# Tests for L1ObjectManager.lock()
# =============================================================================


class TestLock(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.lock() method.

    Per the docstring:
    - Thread-safe function to add lock counter for committed keys
    - Keys must be committed
    - Supports both FORCED and ALL-OR-NOTHING semantics via `force` parameter
    - Returns KEYS_NOT_COMMITTED if keys are not committed (reserved or not exist)
    """

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful(), (
            f"Failed to prepare committed keys: {result.error}"
        )

    def test_lock_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test locking a single committed key successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare committed key
        self._prepare_committed_keys(manager, [key], [obj])

        # Verify key is committed and unlocked
        self.assert_keys_committed(manager, [key])
        self.assert_keys_unlocked(manager, [key])

        # Lock
        result = manager.lock([key], force=False)

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key is now locked
        self.assert_keys_locked(manager, [key])

    def test_lock_multiple_keys_success(self, manager, keys_5, memory_objs_5):
        """Test locking multiple committed keys successfully."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        # Verify all keys are committed and unlocked
        self.assert_keys_committed(manager, keys_5)
        self.assert_keys_unlocked(manager, keys_5)

        # Lock all
        result = manager.lock(keys_5, force=False)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys are now locked
        self.assert_keys_locked(manager, keys_5)

    def test_lock_not_committed_fails(self, manager, keys_3):
        """Test that locking non-existent keys fails."""
        result = manager.lock(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

    def test_lock_reserved_keys_fails(self, manager, keys_3, memory_objs_3):
        """Test that locking reserved (not committed) keys fails."""
        # Only pre-reserve and post-reserve (don't commit)
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)

        result = manager.lock(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

        # Verify keys are still reserved
        self.assert_keys_reserved(manager, keys_3)

    def test_lock_all_or_nothing_semantics(self, manager, keys_5, memory_objs_5):
        """
        Test ALL-OR-NOTHING semantics: if one key fails, none are locked.
        """
        # Only commit first 3 keys
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Try to lock all 5 keys (last 2 will fail)
        result = manager.lock(keys_5, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

        # Due to rollback, no keys should be in success_keys
        assert len(result.success_keys) == 0

        # Verify first 3 keys are still unlocked (rollback worked)
        self.assert_keys_unlocked(manager, keys_5[:3])

        # Now lock only the first 3 keys (should succeed)
        result2 = manager.lock(keys_5[:3], force=False)
        self.assert_result_successful(result2, expected_success_count=3)

        # Verify first 3 keys are now locked
        self.assert_keys_locked(manager, keys_5[:3])

    def test_lock_forced_semantics_partial_success(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test FORCED semantics: locks valid keys and skips invalid ones.
        """
        # Only commit first 3 keys
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Lock all 5 keys with force=True
        result = manager.lock(keys_5, force=True)

        # Should have partial success (3 locked, 2 failed)
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2
        self.assert_keys_in_success(result, keys_5[:3])
        self.assert_keys_in_failed(result, keys_5[3:])

        # Verify first 3 keys are now locked
        self.assert_keys_locked(manager, keys_5[:3])

    def test_lock_multiple_times(self, manager, keys_3, memory_objs_3):
        """Test that locking a key multiple times increases the lock counter."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare committed key
        self._prepare_committed_keys(manager, [key], [obj])

        # Lock multiple times
        for i in range(3):
            result = manager.lock([key], force=False)
            self.assert_result_successful(result, expected_success_count=1)
            self.assert_keys_locked(manager, [key])

    def test_lock_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        result = manager.lock(keys_3, force=True)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_NOT_COMMITTED


# =============================================================================
# Tests for L1ObjectManager.unlock()
# =============================================================================


class TestUnlock(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.unlock() method.

    Per the docstring:
    - Thread-safe function to decrease lock counter for committed keys
    - Keys should be committed
    - Supports both FORCED and ALL-OR-NOTHING semantics via `force` parameter
    - Returns KEYS_NOT_COMMITTED if keys are not committed
    - Recommended to use force=True
    """

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful(), (
            f"Failed to prepare committed keys: {result.error}"
        )

    def _prepare_locked_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed and locked state."""
        self._prepare_committed_keys(manager, keys, memory_objs)
        result = manager.lock(keys, force=False)
        assert result.is_successful(), f"Failed to lock keys: {result.error}"

    def test_unlock_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test unlocking a single locked key successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare locked key
        self._prepare_locked_keys(manager, [key], [obj])

        # Verify key is locked
        self.assert_keys_locked(manager, [key])

        # Unlock
        result = manager.unlock([key], force=False)

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key is now unlocked
        self.assert_keys_unlocked(manager, [key])

    def test_unlock_multiple_keys_success(self, manager, keys_5, memory_objs_5):
        """Test unlocking multiple locked keys successfully."""
        # Prepare locked keys
        self._prepare_locked_keys(manager, keys_5, memory_objs_5)

        # Verify all keys are locked
        self.assert_keys_locked(manager, keys_5)

        # Unlock all
        result = manager.unlock(keys_5, force=False)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys are now unlocked
        self.assert_keys_unlocked(manager, keys_5)

    def test_unlock_empty_keys_success(self, manager):
        """Test unlocking empty key list returns success."""
        result = manager.unlock([], force=False)

        self.assert_result_successful(result, expected_success_count=0)

    def test_unlock_committed_but_not_locked(self, manager, keys_3, memory_objs_3):
        """Test unlocking committed but not locked keys succeeds (decreases counter)."""
        # Prepare committed (but not locked) keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)

        # Verify keys are unlocked
        self.assert_keys_unlocked(manager, keys_3)

        # Unlock should still succeed (even if not locked)
        result = manager.unlock(keys_3, force=False)

        self.assert_result_successful(result, expected_success_count=3)

        # Keys should still be unlocked
        self.assert_keys_unlocked(manager, keys_3)

    def test_unlock_not_committed_fails(self, manager, keys_3):
        """Test that unlocking non-existent keys fails."""
        result = manager.unlock(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

    def test_unlock_reserved_keys_fails(self, manager, keys_3, memory_objs_3):
        """Test that unlocking reserved (not committed) keys fails."""
        # Only pre-reserve and post-reserve (don't commit)
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)

        result = manager.unlock(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

        # Verify keys are still reserved
        self.assert_keys_reserved(manager, keys_3)

    def test_unlock_all_or_nothing_semantics(self, manager, keys_5, memory_objs_5):
        """
        Test ALL-OR-NOTHING semantics: if one key fails, none are unlocked.
        """
        # Only commit and lock first 3 keys
        self._prepare_locked_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Verify first 3 keys are locked
        self.assert_keys_locked(manager, keys_5[:3])

        # Try to unlock all 5 keys (last 2 will fail)
        result = manager.unlock(keys_5, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

        # Due to rollback, no keys should be in success_keys
        assert len(result.success_keys) == 0

        # Verify first 3 keys are still locked (rollback worked)
        self.assert_keys_locked(manager, keys_5[:3])

    def test_unlock_forced_semantics_partial_success(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test FORCED semantics: unlocks valid keys and skips invalid ones.
        """
        # Only commit and lock first 3 keys
        self._prepare_locked_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Unlock all 5 keys with force=True
        result = manager.unlock(keys_5, force=True)

        # Should have partial success (3 unlocked, 2 failed)
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2
        self.assert_keys_in_success(result, keys_5[:3])
        self.assert_keys_in_failed(result, keys_5[3:])

        # Verify first 3 keys are now unlocked
        self.assert_keys_unlocked(manager, keys_5[:3])

    def test_unlock_multiple_locks(self, manager, keys_3, memory_objs_3):
        """Test that unlocking requires matching number of unlock calls."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare committed key
        self._prepare_committed_keys(manager, [key], [obj])

        # Lock multiple times
        for _ in range(3):
            manager.lock([key], force=False)

        # Verify key is locked
        self.assert_keys_locked(manager, [key])

        # Unlock twice - should still be locked
        manager.unlock([key], force=False)
        manager.unlock([key], force=False)
        self.assert_keys_locked(manager, [key])

        # Unlock third time - should be unlocked now
        manager.unlock([key], force=False)
        self.assert_keys_unlocked(manager, [key])

    def test_unlock_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        result = manager.unlock(keys_3, force=True)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_NOT_COMMITTED


# =============================================================================
# Integration Tests for Lock/Unlock
# =============================================================================


class TestLockUnlockIntegration(L1ObjectManagerTestBase):
    """
    Integration tests combining lock and unlock operations with other operations.
    """

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        manager.commit(keys, force=False)

    def test_lock_unlock_cycle(self, manager, keys_3, memory_objs_3):
        """Test complete lock/unlock cycle."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)

        # Lock
        lock_result = manager.lock(keys_3, force=False)
        self.assert_result_successful(lock_result, expected_success_count=3)
        self.assert_keys_locked(manager, keys_3)

        # Unlock
        unlock_result = manager.unlock(keys_3, force=False)
        self.assert_result_successful(unlock_result, expected_success_count=3)
        self.assert_keys_unlocked(manager, keys_3)

    def test_multiple_lock_unlock_cycles(self, manager, keys_3, memory_objs_3):
        """Test multiple lock/unlock cycles."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)

        # Multiple cycles
        for _ in range(5):
            lock_result = manager.lock(keys_3, force=False)
            self.assert_result_successful(lock_result, expected_success_count=3)
            self.assert_keys_locked(manager, keys_3)

            unlock_result = manager.unlock(keys_3, force=False)
            self.assert_result_successful(unlock_result, expected_success_count=3)
            self.assert_keys_unlocked(manager, keys_3)

    def test_full_workflow_with_lock_unlock(self, manager, keys_3, memory_objs_3):
        """
        Test complete workflow:
        prereserve -> postreserve -> commit -> lock -> unlock -> mark_reserved -> commit
        """
        # Pre-reserve
        prereserve_result = manager.prereserve_forced(keys_3)
        self.assert_result_successful(prereserve_result, expected_success_count=3)

        # Post-reserve
        postreserve_result = manager.postreserve_must(keys_3, memory_objs_3)
        self.assert_result_successful(postreserve_result, expected_success_count=3)

        # Commit
        commit_result = manager.commit(keys_3, force=False)
        self.assert_result_successful(commit_result, expected_success_count=3)
        self.assert_keys_committed(manager, keys_3)

        # Lock
        lock_result = manager.lock(keys_3, force=False)
        self.assert_result_successful(lock_result, expected_success_count=3)
        self.assert_keys_locked(manager, keys_3)

        # Unlock
        unlock_result = manager.unlock(keys_3, force=False)
        self.assert_result_successful(unlock_result, expected_success_count=3)
        self.assert_keys_unlocked(manager, keys_3)

        # Mark as reserved (for update)
        mark_result = manager.mark_reserved_must(keys_3)
        self.assert_result_successful(mark_result, expected_success_count=3)
        self.assert_keys_reserved(manager, keys_3)

        # Commit again
        commit_result2 = manager.commit(keys_3, force=False)
        self.assert_result_successful(commit_result2, expected_success_count=3)
        self.assert_keys_committed(manager, keys_3)

    def test_concurrent_lock_unlock_operations(self, manager):
        """
        Test concurrent lock and unlock operations on different keys.
        """
        num_threads = 4
        keys_per_thread = 5
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(10):
                    keys = create_object_keys(
                        keys_per_thread, model_name=f"model_{thread_id}_{i}"
                    )
                    objs = create_mock_memory_objs(keys_per_thread)

                    # Full workflow
                    manager.prereserve_forced(keys)
                    manager.postreserve_must(keys, objs)
                    manager.commit(keys, force=False)
                    manager.lock(keys, force=False)
                    manager.unlock(keys, force=False)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Concurrent operation errors: {exceptions}"


# =============================================================================
# Tests for L1ObjectManager.lookup_and_lock_forced()
# =============================================================================


class TestLookupAndLockForced(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.lookup_and_lock_forced() method.

    Per the docstring:
    - Lookup keys in "committed" ones, lock found ones and return memory objects
    - Atomic "lookup and lock" operation
    - Uses FORCED semantics: tries its best to lookup and lock, skips keys not found
    - Returns tuple of (L1OperationResult, list[MemoryObj])
    - Returns KEYS_NOT_FOUND if some keys are not found
    """

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful(), (
            f"Failed to prepare committed keys: {result.error}"
        )

    def test_lookup_and_lock_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test lookup and lock a single committed key successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare committed key
        self._prepare_committed_keys(manager, [key], [obj])

        # Verify key is committed and unlocked
        self.assert_keys_committed(manager, [key])
        self.assert_keys_unlocked(manager, [key])

        # Lookup and lock
        result, objs = manager.lookup_and_lock_forced([key])

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify returned memory objects
        assert len(objs) == 1
        assert objs[0] is obj

        # Verify key is now locked
        self.assert_keys_locked(manager, [key])

    def test_lookup_and_lock_multiple_keys_success(
        self, manager, keys_5, memory_objs_5
    ):
        """Test lookup and lock multiple committed keys successfully."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        # Verify all keys are committed and unlocked
        self.assert_keys_committed(manager, keys_5)
        self.assert_keys_unlocked(manager, keys_5)

        # Lookup and lock all
        result, objs = manager.lookup_and_lock_forced(keys_5)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify returned memory objects
        assert len(objs) == 5
        for i, obj in enumerate(objs):
            assert obj is memory_objs_5[i]

        # Verify all keys are now locked
        self.assert_keys_locked(manager, keys_5)

    def test_lookup_and_lock_empty_keys_success(self, manager):
        """Test lookup and lock empty key list returns success with empty list."""
        result, objs = manager.lookup_and_lock_forced([])

        self.assert_result_successful(result, expected_success_count=0)
        assert len(objs) == 0

    def test_lookup_and_lock_not_found_fails(self, manager, keys_3):
        """Test that lookup and lock of non-existent keys returns KEYS_NOT_FOUND."""
        result, objs = manager.lookup_and_lock_forced(keys_3)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_FOUND)
        assert len(result.success_keys) == 0
        assert len(result.failed_keys) == 3
        assert len(objs) == 0

    def test_lookup_and_lock_reserved_keys_fails(self, manager, keys_3, memory_objs_3):
        """Test that lookup and lock of reserved (not committed) keys fails."""
        # Only pre-reserve and post-reserve (don't commit)
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)

        result, objs = manager.lookup_and_lock_forced(keys_3)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_FOUND)
        assert len(result.success_keys) == 0
        assert len(objs) == 0

        # Verify keys are still reserved
        self.assert_keys_reserved(manager, keys_3)

    def test_lookup_and_lock_forced_semantics_partial_success(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test FORCED semantics: locks found keys and skips not-found ones.

        Per docstring: "This function uses 'FORCED' semantics. It will try its best
        to lookup and lock the given keys, and skip the keys that are not found."
        """
        # Only commit first 3 keys
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Lookup and lock all 5 keys (last 2 not found)
        result, objs = manager.lookup_and_lock_forced(keys_5)

        # Should have partial success (3 found and locked, 2 not found)
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2
        self.assert_keys_in_success(result, keys_5[:3])
        self.assert_keys_in_failed(result, keys_5[3:])

        # Verify returned memory objects match success_keys order
        assert len(objs) == 3
        for i, obj in enumerate(objs):
            assert obj is memory_objs_5[i]

        # Verify first 3 keys are now locked
        self.assert_keys_locked(manager, keys_5[:3])

    def test_lookup_and_lock_all_not_found_forced(self, manager, keys_3):
        """Test FORCED semantics when all keys not found."""
        result, objs = manager.lookup_and_lock_forced(keys_3)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_FOUND)
        assert len(result.success_keys) == 0
        assert len(result.failed_keys) == 3
        assert len(objs) == 0

    def test_lookup_and_lock_memory_objs_order_matches_success_keys(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test that returned memory objects are in the same order as success_keys.

        Per docstring: "list[MemoryObj]: Memory objects for the successfully locked
        keys, in the same order as success_keys."
        """
        # Commit only keys at indices 0, 2, 4 (not 1, 3)
        selected_indices = [0, 2, 4]
        selected_keys = [keys_5[i] for i in selected_indices]
        selected_objs = [memory_objs_5[i] for i in selected_indices]

        self._prepare_committed_keys(manager, selected_keys, selected_objs)

        # Lookup and lock all 5 keys
        result, objs = manager.lookup_and_lock_forced(keys_5)

        # Verify success_keys and objs match
        assert len(result.success_keys) == 3
        assert len(objs) == 3

        # Each obj should match the corresponding success_key
        for success_key, returned_obj in zip(result.success_keys, objs, strict=False):
            idx = selected_keys.index(success_key)
            assert returned_obj is selected_objs[idx]

    def test_lookup_and_lock_atomicity(self, manager, keys_3, memory_objs_3):
        """
        Test that lookup and lock are atomic.

        Per docstring: "This function will ensure that the 'lookup and lock' are
        atomic, which means that once the function returns, the caller is guaranteed
        to have the lock on the returned memory objects."
        """
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)

        # Lookup and lock
        result, objs = manager.lookup_and_lock_forced(keys_3)

        self.assert_result_successful(result, expected_success_count=3)

        # Immediately verify all keys are locked
        self.assert_keys_locked(manager, keys_3)

        # Verify we have valid memory objects
        assert len(objs) == 3
        for obj in objs:
            assert obj is not None

    def test_lookup_and_lock_already_locked_increases_counter(
        self, manager, keys_3, memory_objs_3
    ):
        """Test that looking up already-locked keys increases the lock counter."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare committed key
        self._prepare_committed_keys(manager, [key], [obj])

        # First lookup and lock
        result1, objs1 = manager.lookup_and_lock_forced([key])
        self.assert_result_successful(result1, expected_success_count=1)
        self.assert_keys_locked(manager, [key])

        # Second lookup and lock (should still succeed - lock counter increases)
        result2, objs2 = manager.lookup_and_lock_forced([key])
        self.assert_result_successful(result2, expected_success_count=1)
        self.assert_keys_locked(manager, [key])

        # Verify same memory object returned
        assert objs1[0] is objs2[0]

        # After one unlock, key should still be locked
        manager.unlock([key], force=True)
        self.assert_keys_locked(manager, [key])

        # After second unlock, key should be unlocked
        manager.unlock([key], force=True)
        self.assert_keys_unlocked(manager, [key])

    def test_lookup_and_lock_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        result, objs = manager.lookup_and_lock_forced(keys_3)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_NOT_FOUND


class TestLookupAndLockForcedThreadSafety(L1ObjectManagerTestBase):
    """Thread-safety tests for lookup_and_lock_forced()."""

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful()

    def test_lookup_and_lock_concurrent_disjoint_keys(self, manager):
        """Test concurrent lookup_and_lock of disjoint key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare all keys as committed first (single-threaded setup)
        all_keys = []
        all_objs = []
        for i in range(num_threads):
            keys = create_object_keys(keys_per_thread, model_name=f"model_{i}")
            objs = create_mock_memory_objs(keys_per_thread)
            all_keys.append(keys)
            all_objs.append(objs)
            self._prepare_committed_keys(manager, keys, objs)

        def lookup_task(thread_id):
            try:
                result, objs = manager.lookup_and_lock_forced(all_keys[thread_id])
                with lock:
                    results.append((result, objs))
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=lookup_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All lookups should succeed
        for result, objs in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )
            assert len(objs) == keys_per_thread

    def test_lookup_and_lock_concurrent_same_keys(self, manager, keys_5, memory_objs_5):
        """
        Test concurrent lookup_and_lock of same keys.

        All threads should succeed (FORCED semantics), and all should get valid
        memory objects since the keys are already committed.
        """
        num_threads = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare committed keys (single-threaded)
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        def lookup_task(thread_id):
            try:
                result, objs = manager.lookup_and_lock_forced(keys_5)
                with lock:
                    results.append((thread_id, result, objs))
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=lookup_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All threads should succeed (they all lookup same committed keys)
        for thread_id, result, objs in results:
            self.assert_result_successful(result, expected_success_count=5)
            assert len(objs) == 5

        # Keys should be locked (with lock counter = num_threads)
        self.assert_keys_locked(manager, keys_5)

    def test_lookup_and_lock_vs_mark_reserved_contention(self, manager):
        """
        Test concurrent lookup_and_lock vs mark_reserved_must operations.

        This tests the interaction between:
        - Thread A: lookup_and_lock on committed objects, verify locked state via
          query_states, then unlock
        - Thread B: try to mark_reserved_must on committed objects one by one,
          wait a bit, then commit them back

        Expected behavior:
        - mark_reserved_must should fail on locked objects (KEYS_ALREADY_LOCKED)
        - After all operations complete, system should be in consistent state:
          all objects committed and unlocked
        """
        # Standard
        import random
        import time

        num_keys = 20
        iterations = 30
        exceptions = []
        lock = threading.Lock()
        stop_event = threading.Event()

        # Prepare committed keys
        keys = create_object_keys(num_keys)
        objs = create_mock_memory_objs(num_keys)
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, objs)
        manager.commit(keys, force=False)

        # Verify initial state
        self.assert_keys_committed(manager, keys)
        self.assert_keys_unlocked(manager, keys)

        def thread_a_lookup_and_lock():
            """
            Thread A: lookup_and_lock on some committed objects, verify they're
            locked via query_states, then unlock.
            """
            try:
                for _ in range(iterations):
                    # Select random subset of keys
                    num_to_lock = random.randint(1, min(5, num_keys))
                    selected_keys = random.sample(keys, num_to_lock)

                    # Lookup and lock
                    result, _ = manager.lookup_and_lock_forced(selected_keys)

                    # Process only successfully locked keys
                    if result.success_keys:
                        # Verify all success_keys are now locked via query_states
                        for key in result.success_keys:
                            states = manager.query_states([key])
                            state = states[0]
                            # Object should exist and be locked
                            assert state.exists(), f"Key {key} should exist after lock"
                            assert state.is_locked(), f"Key {key} should be locked"
                            assert state.is_committed(), (
                                f"Key {key} should be committed"
                            )

                        # Small delay to create contention window
                        time.sleep(random.uniform(0.0001, 0.001))

                        # Verify they're still there and locked before unlock
                        for key in result.success_keys:
                            states = manager.query_states([key])
                            state = states[0]
                            assert state.exists(), f"Key {key} should still exist"
                            # Note: might be unlocked by TTL expiry in real scenario,
                            # but in test it should still be locked

                        # Unlock
                        manager.unlock(result.success_keys, force=True)

                    if stop_event.is_set():
                        break
            except Exception as e:
                with lock:
                    exceptions.append(("thread_a", e))

        def thread_b_mark_reserved():
            """
            Thread B: try to mark committed objects as reserved one by one,
            wait a bit, then commit them back.
            """
            try:
                for _ in range(iterations):
                    # Select a random key to try to mark as reserved
                    key = random.choice(keys)

                    # Try to mark as reserved (one key at a time to avoid rollback)
                    result = manager.mark_reserved_must([key])

                    if result.is_successful():
                        # Successfully marked as reserved
                        # Verify state
                        states = manager.query_states([key])
                        state = states[0]
                        assert state.exists(), (
                            f"Key {key} should exist after mark_reserved"
                        )
                        assert state.is_reserved(), f"Key {key} should be reserved"

                        # Small delay
                        time.sleep(random.uniform(0.0001, 0.0005))

                        # Commit back
                        commit_result = manager.commit([key], force=False)
                        assert commit_result.is_successful(), (
                            f"Commit after mark_reserved should succeed for key {key}"
                        )
                    else:
                        # mark_reserved_must failed - expected if key was locked
                        # or in reserved state (by another iteration)
                        pass

                    if stop_event.is_set():
                        break
            except Exception as e:
                with lock:
                    exceptions.append(("thread_b", e))

        # Start both threads
        thread_a = threading.Thread(target=thread_a_lookup_and_lock)
        thread_b = threading.Thread(target=thread_b_mark_reserved)

        thread_a.start()
        thread_b.start()

        # Wait for both threads to complete
        thread_a.join(timeout=30)
        thread_b.join(timeout=30)

        # Signal stop if threads are still running
        stop_event.set()

        # Check for exceptions
        assert len(exceptions) == 0, f"Thread errors: {exceptions}"

        # Verify final state consistency:
        # All keys should be committed and unlocked (no reserved state)
        final_states = manager.query_states(keys)
        for key, state in zip(keys, final_states, strict=False):
            assert state.exists(), f"Key {key} should exist at end"
            assert state.is_committed(), (
                f"Key {key} should be committed at end, "
                f"got reserved={state.is_reserved()}"
            )
            # Note: some keys might still be locked if thread_a didn't finish unlocking
            # But after proper completion, all should be unlocked

        # Unlock any remaining locks to ensure clean state
        manager.unlock(keys, force=True)

        # Final verification: all unlocked
        self.assert_keys_committed(manager, keys)
        self.assert_keys_unlocked(manager, keys)


# =============================================================================
# Tests for L1ObjectManager.delete_committed()
# =============================================================================


class TestDeleteCommitted(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.delete_committed() method.

    Per the docstring:
    - Remove keys from the object manager
    - Keys should be committed but not locked
    - Supports both FORCED and ALL-OR-NOTHING semantics via `force` parameter
    - Returns KEYS_NOT_COMMITTED if some keys are not committed
    - Returns KEYS_ALREADY_LOCKED if some keys are locked
    - Does NOT free memory - caller needs to manually free
    """

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful(), (
            f"Failed to prepare committed keys: {result.error}"
        )

    def test_delete_committed_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test deleting a single committed (unlocked) key successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Prepare committed key
        self._prepare_committed_keys(manager, [key], [obj])

        # Verify key is committed and unlocked
        self.assert_keys_committed(manager, [key])
        self.assert_keys_unlocked(manager, [key])

        # Delete
        result = manager.delete_committed([key], force=False)

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key no longer exists
        self.assert_keys_not_exist(manager, [key])

    def test_delete_committed_multiple_keys_success(
        self, manager, keys_5, memory_objs_5
    ):
        """Test deleting multiple committed (unlocked) keys successfully."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        # Verify all keys are committed and unlocked
        self.assert_keys_committed(manager, keys_5)
        self.assert_keys_unlocked(manager, keys_5)

        # Delete all
        result = manager.delete_committed(keys_5, force=False)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys no longer exist
        self.assert_keys_not_exist(manager, keys_5)

    def test_delete_committed_empty_keys_success(self, manager):
        """Test deleting empty key list returns success."""
        result = manager.delete_committed([], force=False)

        self.assert_result_successful(result, expected_success_count=0)

    def test_delete_committed_not_committed_fails(self, manager, keys_3):
        """Test that deleting non-existent keys fails."""
        result = manager.delete_committed(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

    def test_delete_committed_reserved_keys_fails(self, manager, keys_3, memory_objs_3):
        """Test that deleting reserved (not committed) keys fails."""
        # Only pre-reserve and post-reserve (don't commit)
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)

        result = manager.delete_committed(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

        # Verify keys are still reserved
        self.assert_keys_reserved(manager, keys_3)

    def test_delete_committed_locked_keys_fails(self, manager, keys_3, memory_objs_3):
        """
        Test that deleting locked keys fails.

        Per docstring: "Returns KEYS_ALREADY_LOCKED if some keys are locked."
        """
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)

        # Lock the keys
        manager.lock(keys_3, force=False)

        # Verify keys are locked
        self.assert_keys_locked(manager, keys_3)

        # Try to delete (should fail)
        result = manager.delete_committed(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_ALREADY_LOCKED)

        # Verify keys still exist and are still locked
        self.assert_keys_committed(manager, keys_3)
        self.assert_keys_locked(manager, keys_3)

    def test_delete_committed_all_or_nothing_semantics(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test ALL-OR-NOTHING semantics: if one key fails, none are deleted.
        """
        # Only commit first 3 keys
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Try to delete all 5 keys (last 2 will fail - not committed)
        result = manager.delete_committed(keys_5, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_COMMITTED)

        # Due to rollback, no keys should be in success_keys
        assert len(result.success_keys) == 0

        # Skipped keys should be present
        assert len(result.skipped_keys) > 0 or len(result.failed_keys) > 0

        # Verify first 3 keys are still committed (rollback worked)
        self.assert_keys_committed(manager, keys_5[:3])

    def test_delete_committed_all_or_nothing_with_locked_key(
        self, manager, keys_5, memory_objs_5
    ):
        """Test ALL-OR-NOTHING when one key is locked."""
        # Commit all 5 keys
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        # Lock only the third key
        manager.lock([keys_5[2]], force=False)

        # Try to delete all 5 keys (3rd will fail - locked)
        result = manager.delete_committed(keys_5, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_ALREADY_LOCKED)

        # Due to rollback, no keys should be in success_keys
        assert len(result.success_keys) == 0

        # Verify all keys still exist (rollback worked)
        self.assert_keys_committed(manager, keys_5)

    def test_delete_committed_forced_semantics_partial_success(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test FORCED semantics: deletes valid keys and skips invalid ones.
        """
        # Only commit first 3 keys
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Delete all 5 keys with force=True
        result = manager.delete_committed(keys_5, force=True)

        # Should have partial success (3 deleted, 2 failed)
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2
        self.assert_keys_in_success(result, keys_5[:3])
        self.assert_keys_in_failed(result, keys_5[3:])

        # Verify first 3 keys no longer exist
        self.assert_keys_not_exist(manager, keys_5[:3])

        # Verify last 2 keys still don't exist
        self.assert_keys_not_exist(manager, keys_5[3:])

    def test_delete_committed_forced_with_locked_key(
        self, manager, keys_5, memory_objs_5
    ):
        """Test FORCED semantics when some keys are locked."""
        # Commit all 5 keys
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        # Lock keys at indices 1 and 3
        locked_keys = [keys_5[1], keys_5[3]]
        manager.lock(locked_keys, force=False)

        # Delete all 5 keys with force=True
        result = manager.delete_committed(keys_5, force=True)

        # Should have partial success (3 deleted, 2 locked fail)
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2

        # Verify locked keys still exist
        self.assert_keys_committed(manager, locked_keys)
        self.assert_keys_locked(manager, locked_keys)

        # Verify non-locked keys are deleted
        non_locked_keys = [keys_5[0], keys_5[2], keys_5[4]]
        self.assert_keys_not_exist(manager, non_locked_keys)

    def test_delete_committed_allows_rereserve(self, manager, keys_3, memory_objs_3):
        """Test that deleted keys can be pre-reserved again."""
        # Prepare and delete committed keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)
        delete_result = manager.delete_committed(keys_3, force=False)
        self.assert_result_successful(delete_result, expected_success_count=3)

        # Verify keys don't exist
        self.assert_keys_not_exist(manager, keys_3)

        # Re-reserve should succeed
        prereserve_result = manager.prereserve_forced(keys_3)
        self.assert_result_successful(prereserve_result, expected_success_count=3)

        # Verify keys are now reserved
        self.assert_keys_reserved(manager, keys_3)

    def test_delete_committed_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        result = manager.delete_committed(keys_3, force=True)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_NOT_COMMITTED

    def test_delete_committed_mixed_failure_reasons_forced(
        self, manager, keys_5, memory_objs_5
    ):
        """Test FORCED semantics with mixed failure reasons."""
        # Commit keys 0, 1, 2
        self._prepare_committed_keys(manager, keys_5[:3], memory_objs_5[:3])

        # Lock key 1
        manager.lock([keys_5[1]], force=False)

        # Try to delete all 5 keys with force=True
        # key 0: success (committed, unlocked)
        # key 1: fail KEYS_ALREADY_LOCKED (committed, locked)
        # key 2: success (committed, unlocked)
        # key 3: fail KEYS_NOT_COMMITTED
        # key 4: fail KEYS_NOT_COMMITTED
        result = manager.delete_committed(keys_5, force=True)

        # Should have 2 success, 3 fail
        assert len(result.success_keys) == 2
        assert len(result.failed_keys) == 3

        # Verify failure reasons are correct
        for key, reason in zip(result.failed_keys, result.failed_reasons, strict=False):
            if key == keys_5[1]:
                assert reason == L1ObjectManagerError.KEYS_ALREADY_LOCKED
            else:
                assert reason == L1ObjectManagerError.KEYS_NOT_COMMITTED


class TestDeleteCommittedThreadSafety(L1ObjectManagerTestBase):
    """Thread-safety tests for delete_committed()."""

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        result = manager.commit(keys, force=False)
        assert result.is_successful()

    def test_delete_committed_concurrent_disjoint_keys(self, manager):
        """Test concurrent delete_committed of disjoint key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare all keys as committed first (single-threaded setup)
        all_keys = []
        for i in range(num_threads):
            keys = create_object_keys(keys_per_thread, model_name=f"model_{i}")
            objs = create_mock_memory_objs(keys_per_thread)
            all_keys.append(keys)
            self._prepare_committed_keys(manager, keys, objs)

        def delete_task(thread_id):
            try:
                result = manager.delete_committed(all_keys[thread_id], force=False)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=delete_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All deletes should succeed
        for result in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )

    def test_delete_committed_concurrent_same_keys(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test concurrent delete_committed of same keys.

        Only one thread should successfully delete each key.
        """
        num_threads = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare committed keys (single-threaded)
        self._prepare_committed_keys(manager, keys_5, memory_objs_5)

        def delete_task(thread_id):
            try:
                result = manager.delete_committed(keys_5, force=True)
                with lock:
                    results.append((thread_id, result))
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=delete_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # Total success deletions should equal number of keys
        total_success = sum(len(r.success_keys) for _, r in results)
        assert total_success == len(keys_5), (
            f"Expected {len(keys_5)} total successes, got {total_success}"
        )

        # Verify all keys are deleted
        self.assert_keys_not_exist(manager, keys_5)

    def test_delete_committed_high_contention(self, manager):
        """Test delete_committed under high thread contention."""
        num_threads = 20
        iterations = 20
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(iterations):
                    keys = create_object_keys(3, model_name=f"model_{thread_id}_{i}")
                    objs = create_mock_memory_objs(3)

                    # Full workflow with delete
                    manager.prereserve_forced(keys)
                    manager.postreserve_must(keys, objs)
                    manager.commit(keys, force=False)

                    # Delete
                    result = manager.delete_committed(keys, force=False)
                    assert result.is_successful()

                    # Verify deleted
                    states = manager.query_states(keys)
                    for state in states:
                        assert not state.exists()
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"


# =============================================================================
# Tests for L1ObjectManager.delete_reserved()
# =============================================================================


class TestDeleteReserved(L1ObjectManagerTestBase):
    """
    Tests for L1ObjectManager.delete_reserved() method.

    Per the docstring:
    - Drop from the reserved keys
    - Supports both FORCED and ALL-OR-NOTHING semantics via `force` parameter
    - Returns KEYS_NOT_RESERVED if some keys are not reserved
    - Does NOT free memory - caller needs to manually free
    """

    def test_delete_reserved_single_key_success(self, manager, keys_3, memory_objs_3):
        """Test deleting a single reserved key successfully."""
        key = keys_3[0]
        obj = memory_objs_3[0]

        # Pre-reserve and post-reserve (reserved with memory object)
        manager.prereserve_forced([key])
        manager.postreserve_must([key], [obj])

        # Verify key is reserved
        self.assert_keys_reserved(manager, [key])
        self.assert_keys_have_memory_obj(manager, [key])

        # Delete
        result = manager.delete_reserved([key], force=False)

        self.assert_result_successful(result, expected_success_count=1)
        self.assert_keys_in_success(result, [key])

        # Verify key no longer exists
        self.assert_keys_not_exist(manager, [key])

    def test_delete_reserved_multiple_keys_success(
        self, manager, keys_5, memory_objs_5
    ):
        """Test deleting multiple reserved keys successfully."""
        # Pre-reserve and post-reserve
        manager.prereserve_forced(keys_5)
        manager.postreserve_must(keys_5, memory_objs_5)

        # Verify all keys are reserved
        self.assert_keys_reserved(manager, keys_5)

        # Delete all
        result = manager.delete_reserved(keys_5, force=False)

        self.assert_result_successful(result, expected_success_count=5)
        self.assert_keys_in_success(result, keys_5)

        # Verify all keys no longer exist
        self.assert_keys_not_exist(manager, keys_5)

    def test_delete_reserved_prereserved_only(self, manager, keys_3):
        """Test deleting pre-reserved keys (without memory objects) succeeds."""
        # Only pre-reserve
        manager.prereserve_forced(keys_3)

        # Verify keys are reserved without memory object
        self.assert_keys_reserved(manager, keys_3)
        self.assert_keys_no_memory_obj(manager, keys_3)

        # Delete
        result = manager.delete_reserved(keys_3, force=False)

        self.assert_result_successful(result, expected_success_count=3)

        # Verify keys no longer exist
        self.assert_keys_not_exist(manager, keys_3)

    def test_delete_reserved_empty_keys_success(self, manager):
        """Test deleting empty key list returns success."""
        result = manager.delete_reserved([], force=False)

        self.assert_result_successful(result, expected_success_count=0)

    def test_delete_reserved_not_reserved_fails(self, manager, keys_3):
        """Test that deleting non-existent keys fails."""
        result = manager.delete_reserved(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)

    def test_delete_reserved_committed_keys_fails(self, manager, keys_3, memory_objs_3):
        """Test that deleting committed (not reserved) keys fails."""
        # Prepare committed keys (not reserved)
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)
        manager.commit(keys_3, force=False)

        # Verify keys are committed
        self.assert_keys_committed(manager, keys_3)

        # Try to delete reserved (should fail - they're committed)
        result = manager.delete_reserved(keys_3, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)

        # Verify keys are still committed
        self.assert_keys_committed(manager, keys_3)

    def test_delete_reserved_all_or_nothing_semantics(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test ALL-OR-NOTHING semantics: if one key fails, none are deleted.
        """
        # Only reserve first 3 keys
        manager.prereserve_forced(keys_5[:3])
        manager.postreserve_must(keys_5[:3], memory_objs_5[:3])

        # Try to delete all 5 keys (last 2 will fail - not reserved)
        result = manager.delete_reserved(keys_5, force=False)

        self.assert_result_has_error(result, L1ObjectManagerError.KEYS_NOT_RESERVED)

        # Due to rollback, no keys should be in success_keys
        assert len(result.success_keys) == 0

        # Skipped keys should be present
        assert len(result.skipped_keys) > 0 or len(result.failed_keys) > 0

        # Verify first 3 keys are still reserved (rollback worked)
        self.assert_keys_reserved(manager, keys_5[:3])

    def test_delete_reserved_rollback_on_failure(self, manager, keys_5, memory_objs_5):
        """
        Test that successful entries are rolled back when later entry fails.
        """
        # Only reserve first 3 keys
        manager.prereserve_forced(keys_5[:3])
        manager.postreserve_must(keys_5[:3], memory_objs_5[:3])

        # Try to delete all 5 keys (4th will fail)
        result = manager.delete_reserved(keys_5, force=False)
        assert not result.is_successful()

        # Verify first 3 keys are still reserved (rollback worked)
        self.assert_keys_reserved(manager, keys_5[:3])

        # After rollback, first 3 keys should still be deletable
        second_result = manager.delete_reserved(keys_5[:3], force=False)
        self.assert_result_successful(second_result, expected_success_count=3)

        # Verify first 3 keys are now deleted
        self.assert_keys_not_exist(manager, keys_5[:3])

    def test_delete_reserved_forced_semantics_partial_success(
        self, manager, keys_5, memory_objs_5
    ):
        """
        Test FORCED semantics: deletes valid keys and skips invalid ones.
        """
        # Only reserve first 3 keys
        manager.prereserve_forced(keys_5[:3])
        manager.postreserve_must(keys_5[:3], memory_objs_5[:3])

        # Delete all 5 keys with force=True
        result = manager.delete_reserved(keys_5, force=True)

        # Should have partial success (3 deleted, 2 failed)
        assert len(result.success_keys) == 3
        assert len(result.failed_keys) == 2
        self.assert_keys_in_success(result, keys_5[:3])
        self.assert_keys_in_failed(result, keys_5[3:])

        # Verify first 3 keys no longer exist
        self.assert_keys_not_exist(manager, keys_5[:3])

        # Verify last 2 keys still don't exist
        self.assert_keys_not_exist(manager, keys_5[3:])

    def test_delete_reserved_forced_mixed_states(self, manager, keys_5, memory_objs_5):
        """Test FORCED semantics with mixed states (some reserved, some committed)."""
        # Reserve keys 0, 1, 2
        manager.prereserve_forced(keys_5[:3])
        manager.postreserve_must(keys_5[:3], memory_objs_5[:3])

        # Commit keys 0 and 1 (so only key 2 is still reserved)
        manager.commit(keys_5[:2], force=False)

        # Verify states
        self.assert_keys_committed(manager, keys_5[:2])
        self.assert_keys_reserved(manager, [keys_5[2]])

        # Delete all 5 keys with force=True
        # key 0, 1: fail KEYS_NOT_RESERVED (committed, not reserved)
        # key 2: success (reserved)
        # key 3, 4: fail KEYS_NOT_RESERVED (not exist)
        result = manager.delete_reserved(keys_5, force=True)

        # Should have 1 success, 4 fail
        assert len(result.success_keys) == 1
        assert len(result.failed_keys) == 4
        self.assert_keys_in_success(result, [keys_5[2]])

        # Verify key 2 is deleted
        self.assert_keys_not_exist(manager, [keys_5[2]])

        # Verify keys 0, 1 are still committed
        self.assert_keys_committed(manager, keys_5[:2])

    def test_delete_reserved_allows_rereserve(self, manager, keys_3, memory_objs_3):
        """Test that deleted reserved keys can be pre-reserved again."""
        # Prepare reserved keys
        manager.prereserve_forced(keys_3)
        manager.postreserve_must(keys_3, memory_objs_3)

        # Delete
        delete_result = manager.delete_reserved(keys_3, force=False)
        self.assert_result_successful(delete_result, expected_success_count=3)

        # Verify keys don't exist
        self.assert_keys_not_exist(manager, keys_3)

        # Re-reserve should succeed
        prereserve_result = manager.prereserve_forced(keys_3)
        self.assert_result_successful(prereserve_result, expected_success_count=3)

        # Verify keys are now reserved
        self.assert_keys_reserved(manager, keys_3)

    def test_delete_reserved_failed_reasons_match_failed_keys(self, manager, keys_3):
        """Test that failed_reasons has same length as failed_keys."""
        result = manager.delete_reserved(keys_3, force=True)

        assert len(result.failed_keys) == len(result.failed_reasons)
        for reason in result.failed_reasons:
            assert reason == L1ObjectManagerError.KEYS_NOT_RESERVED


class TestDeleteReservedThreadSafety(L1ObjectManagerTestBase):
    """Thread-safety tests for delete_reserved()."""

    def test_delete_reserved_concurrent_disjoint_keys(self, manager):
        """Test concurrent delete_reserved of disjoint key sets."""
        num_threads = 5
        keys_per_thread = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare all keys as reserved first (single-threaded setup)
        all_keys = []
        for i in range(num_threads):
            keys = create_object_keys(keys_per_thread, model_name=f"model_{i}")
            objs = create_mock_memory_objs(keys_per_thread)
            all_keys.append(keys)
            manager.prereserve_forced(keys)
            manager.postreserve_must(keys, objs)

        def delete_task(thread_id):
            try:
                result = manager.delete_reserved(all_keys[thread_id], force=False)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=delete_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # All deletes should succeed
        for result in results:
            self.assert_result_successful(
                result, expected_success_count=keys_per_thread
            )

    def test_delete_reserved_concurrent_same_keys(self, manager, keys_5, memory_objs_5):
        """
        Test concurrent delete_reserved of same keys.

        Only one thread should successfully delete each key.
        """
        num_threads = 10
        results = []
        exceptions = []
        lock = threading.Lock()

        # Prepare reserved keys (single-threaded)
        manager.prereserve_forced(keys_5)
        manager.postreserve_must(keys_5, memory_objs_5)

        def delete_task(thread_id):
            try:
                result = manager.delete_reserved(keys_5, force=True)
                with lock:
                    results.append((thread_id, result))
            except Exception as e:
                with lock:
                    exceptions.append(e)

        threads = [
            threading.Thread(target=delete_task, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"

        # Total success deletions should equal number of keys
        total_success = sum(len(r.success_keys) for _, r in results)
        assert total_success == len(keys_5), (
            f"Expected {len(keys_5)} total successes, got {total_success}"
        )

        # Verify all keys are deleted
        self.assert_keys_not_exist(manager, keys_5)

    def test_delete_reserved_high_contention(self, manager):
        """Test delete_reserved under high thread contention."""
        num_threads = 20
        iterations = 20
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(iterations):
                    keys = create_object_keys(3, model_name=f"model_{thread_id}_{i}")
                    objs = create_mock_memory_objs(3)

                    # Reserve keys
                    manager.prereserve_forced(keys)
                    manager.postreserve_must(keys, objs)

                    # Delete reserved
                    result = manager.delete_reserved(keys, force=False)
                    assert result.is_successful()

                    # Verify deleted
                    states = manager.query_states(keys)
                    for state in states:
                        assert not state.exists()
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Thread-safety errors: {exceptions}"


# =============================================================================
# Integration Tests for Delete Operations
# =============================================================================


class TestDeleteOperationsIntegration(L1ObjectManagerTestBase):
    """
    Integration tests combining delete operations with other operations.
    """

    def _prepare_committed_keys(self, manager, keys, memory_objs):
        """Helper to prepare keys in committed state."""
        manager.prereserve_forced(keys)
        manager.postreserve_must(keys, memory_objs)
        manager.commit(keys, force=False)

    def test_delete_committed_after_unlock(self, manager, keys_3, memory_objs_3):
        """Test that keys can be deleted after being unlocked."""
        # Prepare committed keys
        self._prepare_committed_keys(manager, keys_3, memory_objs_3)

        # Lock keys
        manager.lock(keys_3, force=False)
        self.assert_keys_locked(manager, keys_3)

        # Try to delete (should fail - locked)
        result1 = manager.delete_committed(keys_3, force=False)
        self.assert_result_has_error(result1, L1ObjectManagerError.KEYS_ALREADY_LOCKED)

        # Unlock keys
        manager.unlock(keys_3, force=False)
        self.assert_keys_unlocked(manager, keys_3)

        # Now delete should succeed
        result2 = manager.delete_committed(keys_3, force=False)
        self.assert_result_successful(result2, expected_success_count=3)

        # Verify keys are deleted
        self.assert_keys_not_exist(manager, keys_3)

    def test_delete_reserved_vs_committed(self, manager, keys_5, memory_objs_5):
        """Test that delete_reserved only works on reserved keys."""
        # Reserve all 5 keys, commit first 3
        manager.prereserve_forced(keys_5)
        manager.postreserve_must(keys_5, memory_objs_5)
        manager.commit(keys_5[:3], force=False)

        # delete_reserved on committed keys should fail
        result1 = manager.delete_reserved(keys_5[:3], force=True)
        assert len(result1.success_keys) == 0
        assert len(result1.failed_keys) == 3

        # delete_reserved on reserved keys should succeed
        result2 = manager.delete_reserved(keys_5[3:], force=False)
        self.assert_result_successful(result2, expected_success_count=2)

        # delete_committed on committed keys should succeed
        result3 = manager.delete_committed(keys_5[:3], force=False)
        self.assert_result_successful(result3, expected_success_count=3)

        # Verify all keys are deleted
        self.assert_keys_not_exist(manager, keys_5)

    def test_full_lifecycle_with_delete(self, manager, keys_3, memory_objs_3):
        """
        Test complete lifecycle:
        prereserve -> postreserve -> commit -> lock -> unlock -> delete
        """
        # Pre-reserve
        prereserve_result = manager.prereserve_forced(keys_3)
        self.assert_result_successful(prereserve_result, expected_success_count=3)

        # Post-reserve
        postreserve_result = manager.postreserve_must(keys_3, memory_objs_3)
        self.assert_result_successful(postreserve_result, expected_success_count=3)

        # Commit
        commit_result = manager.commit(keys_3, force=False)
        self.assert_result_successful(commit_result, expected_success_count=3)

        # Lock
        lock_result = manager.lock(keys_3, force=False)
        self.assert_result_successful(lock_result, expected_success_count=3)

        # Unlock
        unlock_result = manager.unlock(keys_3, force=False)
        self.assert_result_successful(unlock_result, expected_success_count=3)

        # Delete
        delete_result = manager.delete_committed(keys_3, force=False)
        self.assert_result_successful(delete_result, expected_success_count=3)

        # Verify keys are deleted
        self.assert_keys_not_exist(manager, keys_3)

    def test_concurrent_delete_and_other_operations(self, manager):
        """
        Test concurrent delete operations with other operations on different keys.
        """
        num_threads = 4
        keys_per_thread = 5
        exceptions = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(10):
                    keys = create_object_keys(
                        keys_per_thread, model_name=f"model_{thread_id}_{i}"
                    )
                    objs = create_mock_memory_objs(keys_per_thread)

                    # Full workflow with delete
                    manager.prereserve_forced(keys)
                    manager.postreserve_must(keys, objs)

                    # Alternate between deleting reserved and committed
                    if i % 2 == 0:
                        manager.commit(keys, force=False)
                        manager.delete_committed(keys, force=False)
                    else:
                        manager.delete_reserved(keys, force=False)
            except Exception as e:
                with lock:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(exceptions) == 0, f"Concurrent operation errors: {exceptions}"

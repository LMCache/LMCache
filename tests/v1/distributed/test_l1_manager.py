# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for L1Manager.

These tests verify the behavior of L1Manager as described in the
interface docstrings. The tests focus on:

1. reserve_read() - Reserve read access for given keys
   - Returns KEY_NOT_EXIST if key does not exist (a key that is only being
     written -- a staging object -- counts as not existing)
   - Returns KEY_NOT_READABLE if key exists but is write-locked in place
   - Returns SUCCESS and MemoryObj if key is readable

2. unsafe_read() - Unsafe read without acquiring new read locks
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_NOT_READABLE if key is not read-locked
   - Returns SUCCESS and MemoryObj if key is read-locked

3. finish_read() - Finish read access for given keys
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_IN_WRONG_STATE if key is write-locked or non-read-locked
   - Returns SUCCESS on successful unlock
   - Deletes temporary objects when read count reaches zero

4. reserve_write() - Reserve write access for given keys
   - Returns KEY_NOT_WRITABLE if key exists but cannot be written, or the
     same tag already stages the key
   - Returns OUT_OF_MEMORY if allocation fails
   - Returns SUCCESS and MemoryObj on success; a non-resident key becomes a
     staging object owned by the tag, invisible until admitted

5. finish_write() - Finish write access for given keys
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_IN_WRONG_STATE if not write-locked or read-locked
   - Returns SUCCESS on admission (or on discard, if the key already became
     resident) and on in-place unlock

6. delete() - Delete keys from L1 cache
   - Returns KEY_NOT_EXIST if key does not exist
   - Returns KEY_IS_LOCKED if key is locked
   - Returns SUCCESS on successful deletion

7. get_object_state() - Debugging API to get internal state

8. close() - Close the L1Manager and free all resources

9. Staging objects (TestStaging*) - tags, admission, discard, expiry,
   eviction of abandoned reservations and staging-memory accounting
"""

# Standard
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from tests.v1.distributed.utils import should_use_lazy_alloc

try:
    # First Party
    from lmcache.v1.distributed.l1_manager import L1Manager
except ImportError:
    # Skip tests if L1Manager cannot be imported
    pytest.skip(
        "Skipping because L1 manager cannot be imported", allow_module_level=True
    )

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


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
def short_write_ttl_l1_config(basic_memory_config):
    """L1ManagerConfig whose write reservations expire after one second."""
    return L1ManagerConfig(
        memory_config=basic_memory_config,
        write_ttl_seconds=1,
        read_ttl_seconds=300,
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
# Tests for L1Manager.reserve_read()
# =============================================================================


class TestReserveRead:
    """
    Tests for L1Manager.reserve_read() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_NOT_READABLE: The key exists but is not readable.
    - Returns (L1Error, Optional[MemoryObj]) for each key.
    """

    def test_reserve_read_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_read returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_EXIST
        assert mem_obj is None

        manager.close()

    def test_reserve_read_staged_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """A key that is only being written (staging object) is invisible."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write (but don't finish) - the key is staged, not resident
        write_result = manager.reserve_write([key], [False], basic_layout)
        assert write_result[key][0] == L1Error.SUCCESS

        read_result = manager.reserve_read([key])

        assert read_result[key] == (L1Error.KEY_NOT_EXIST, None)

        # Admission makes it readable, with the very same buffer.
        manager.finish_write([key])
        read_result = manager.reserve_read([key])
        assert read_result[key][0] == L1Error.SUCCESS
        assert read_result[key][1] is write_result[key][1]

        manager.finish_read([key])
        manager.close()

    def test_reserve_read_in_place_write_locked_key_returns_key_not_readable(
        self, basic_l1_config, basic_layout
    ):
        """A resident key write-locked in place (mode="update") is not readable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        update = manager.reserve_write([key], [False], basic_layout, mode="update")
        assert update[key][0] == L1Error.SUCCESS

        read_result = manager.reserve_read([key])

        assert read_result[key] == (L1Error.KEY_NOT_READABLE, None)

        manager.finish_write([key])
        manager.close()

    def test_reserve_read_ready_key_returns_success(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_read returns SUCCESS for ready (unlocked) keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create object: reserve write -> finish write
        write_result = manager.reserve_write([key], [False], basic_layout)
        assert write_result[key][0] == L1Error.SUCCESS
        finish_result = manager.finish_write([key])
        assert finish_result[key] == L1Error.SUCCESS

        # Now reserve read
        read_result = manager.reserve_read([key])

        assert key in read_result
        error, mem_obj = read_result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None
        assert mem_obj.is_valid()

        manager.close()

    def test_reserve_read_multiple_keys(self, basic_l1_config, basic_layout):
        """Test reserve_read with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)
        key3 = make_object_key(3)

        # Create key1 as ready object
        manager.reserve_write([key1], [False], basic_layout)
        manager.finish_write([key1])

        # key2 does not exist
        # key3 is being written (staging object, not resident)
        manager.reserve_write([key3], [False], basic_layout)

        # Reserve read on all three
        result = manager.reserve_read([key1, key2, key3])

        assert result[key1][0] == L1Error.SUCCESS
        assert result[key1][1] is not None
        assert result[key2][0] == L1Error.KEY_NOT_EXIST
        assert result[key2][1] is None
        assert result[key3][0] == L1Error.KEY_NOT_EXIST
        assert result[key3][1] is None

        manager.close()

    def test_reserve_read_can_be_called_multiple_times(
        self, basic_l1_config, basic_layout
    ):
        """Test that multiple read reservations can be made on the same key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Multiple read reservations
        result1 = manager.reserve_read([key])
        result2 = manager.reserve_read([key])
        result3 = manager.reserve_read([key])

        assert result1[key][0] == L1Error.SUCCESS
        assert result2[key][0] == L1Error.SUCCESS
        assert result3[key][0] == L1Error.SUCCESS

        # Verify using get_object_state that read lock is held
        state = manager.get_object_state(key)
        assert state is not None
        # Check via available_for_read (should still be true since
        # read-locked is readable)
        assert state.available_for_read() is True

        manager.close()

    def test_reserve_read_with_multiple_read_locks(self, basic_l1_config, basic_layout):
        """Test reserve_read(read_locks=N) acquires N locks."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve with read_locks=3 -> 3 locks
        result = manager.reserve_read([key], read_locks=3)
        assert result[key][0] == L1Error.SUCCESS
        assert result[key][1] is not None

        # Need 3 finish_read() to fully release
        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert state.read_lock.is_locked()

        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert state.read_lock.is_locked()

        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()

    def test_reserve_read_default_is_single_read_lock(
        self, basic_l1_config, basic_layout
    ):
        """Default read_locks=1 acquires exactly 1 lock."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        manager.reserve_read([key])
        manager.finish_read([key])

        # Lock fully released after single finish_read
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()


# =============================================================================
# Tests for L1Manager.unsafe_read()
# =============================================================================


class TestUnsafeRead:
    """
    Tests for L1Manager.unsafe_read() method.

    Per the docstring:
    - This method does not acquire read locks.
    - Caller must ensure unsafe_read is called between reserve_read and finish_read.
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_NOT_READABLE: The key is not readable (not read-locked).
    - Returns (L1Error, Optional[MemoryObj]) for each key.
    """

    def test_unsafe_read_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.unsafe_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_EXIST
        assert mem_obj is None

        manager.close()

    def test_unsafe_read_non_read_locked_returns_key_not_readable(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns KEY_NOT_READABLE if not read-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not read-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Try unsafe_read without reserve_read
        result = manager.unsafe_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_READABLE
        assert mem_obj is None

        manager.close()

    def test_unsafe_read_staged_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read cannot see a staging object."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write without finishing: staging object only
        manager.reserve_write([key], [False], basic_layout)

        result = manager.unsafe_read([key])

        assert result[key] == (L1Error.KEY_NOT_EXIST, None)

        manager.close()

    def test_unsafe_read_read_locked_returns_success(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns SUCCESS for read-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object and reserve read
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS

        # unsafe_read should succeed on read-locked key
        result = manager.unsafe_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None
        assert mem_obj.is_valid()

        manager.close()

    def test_unsafe_read_returns_same_memory_obj_as_reserve_read(
        self, basic_l1_config, basic_layout
    ):
        """Test that unsafe_read returns the same MemoryObj as reserve_read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read and get memory object
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS
        reserved_mem_obj = reserve_result[key][1]

        # unsafe_read should return the same memory object
        unsafe_result = manager.unsafe_read([key])
        assert unsafe_result[key][0] == L1Error.SUCCESS
        unsafe_mem_obj = unsafe_result[key][1]

        # Should be the same object
        assert reserved_mem_obj is unsafe_mem_obj

        manager.close()

    def test_unsafe_read_multiple_times_without_adding_read_count(
        self, basic_l1_config, basic_layout
    ):
        """Test that multiple unsafe_reads don't add to read lock count."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Reserve read once
        manager.reserve_read([key])

        # Multiple unsafe_reads
        for _ in range(5):
            result = manager.unsafe_read([key])
            assert result[key][0] == L1Error.SUCCESS

        # Single finish_read should release the lock and delete temp object
        manager.finish_read([key])

        # Object should be deleted (only 1 read lock was held, not 6)
        assert manager.get_object_state(key) is None

        manager.close()

    def test_unsafe_read_multiple_keys(self, basic_l1_config, basic_layout):
        """Test unsafe_read with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)
        key3 = make_object_key(3)

        # key1: read-locked
        manager.reserve_write([key1], [False], basic_layout)
        manager.finish_write([key1])
        manager.reserve_read([key1])

        # key2: does not exist

        # key3: ready but not read-locked
        manager.reserve_write([key3], [False], basic_layout)
        manager.finish_write([key3])

        # unsafe_read on all three
        result = manager.unsafe_read([key1, key2, key3])

        assert result[key1][0] == L1Error.SUCCESS
        assert result[key1][1] is not None
        assert result[key2][0] == L1Error.KEY_NOT_EXIST
        assert result[key2][1] is None
        assert result[key3][0] == L1Error.KEY_NOT_READABLE
        assert result[key3][1] is None

        manager.close()

    def test_unsafe_read_between_reserve_and_finish(
        self, basic_l1_config, basic_layout
    ):
        """Test proper usage: unsafe_read between reserve_read and finish_read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Proper workflow: reserve_read -> unsafe_read -> finish_read
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS

        unsafe_result = manager.unsafe_read([key])
        assert unsafe_result[key][0] == L1Error.SUCCESS
        assert unsafe_result[key][1] is not None

        finish_result = manager.finish_read([key])
        assert finish_result[key] == L1Error.SUCCESS

        # After finish_read, unsafe_read should fail (not read-locked)
        result = manager.unsafe_read([key])
        assert result[key][0] == L1Error.KEY_NOT_READABLE

        manager.close()


# =============================================================================
# Tests for L1Manager.finish_read()
# =============================================================================


class TestFinishRead:
    """
    Tests for L1Manager.finish_read() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IN_WRONG_STATE: The key is write-locked or non-read-locked.
    - Will delete the object if it is temporary and read count reaches zero.
    """

    def test_finish_read_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_read returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.finish_read([key])

        assert key in result
        assert result[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_finish_read_success(self, basic_l1_config, basic_layout):
        """Test that finish_read returns SUCCESS after proper read reservation."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read, then finish read
        manager.reserve_read([key])
        result = manager.finish_read([key])

        assert result[key] == L1Error.SUCCESS

        manager.close()

    def test_finish_read_non_read_locked_returns_wrong_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_read returns KEY_IN_WRONG_STATE if not read-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not read-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Try to finish read without reserving
        result = manager.finish_read([key])

        assert result[key] == L1Error.KEY_IN_WRONG_STATE

        manager.close()

    def test_finish_read_staged_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_read cannot see a staging object."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write without finishing: staging object only
        manager.reserve_write([key], [False], basic_layout)

        result = manager.finish_read([key])

        assert result[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_finish_read_in_place_write_locked_returns_wrong_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_read returns KEY_IN_WRONG_STATE if write-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Resident object write-locked in place
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_write([key], [False], basic_layout, mode="update")

        result = manager.finish_read([key])

        assert result[key] == L1Error.KEY_IN_WRONG_STATE

        manager.finish_write([key])
        manager.close()

    def test_finish_read_temporary_object_deleted_when_count_zero(
        self, basic_l1_config, basic_layout
    ):
        """Test that temporary objects are deleted when read count reaches zero."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)  # is_temporary=True
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Verify object exists
        assert manager.get_object_state(key) is not None

        # Finish read - should delete the temporary object
        result = manager.finish_read([key])
        assert result[key] == L1Error.SUCCESS

        # Verify object is deleted
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_read_multiple_reads_count_down(self, basic_l1_config, basic_layout):
        """Test that multiple finish_reads count down properly."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Reserve read three times
        manager.reserve_read([key])
        manager.reserve_read([key])
        manager.reserve_read([key])

        # Finish read twice - object should still exist
        manager.finish_read([key])
        manager.finish_read([key])
        assert manager.get_object_state(key) is not None

        # Finish read third time - temporary object should be deleted
        manager.finish_read([key])
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_read_with_multiple_read_locks_releases_multiple(
        self, basic_l1_config, basic_layout
    ):
        """finish_read(read_locks=3) releases 3 locks at once."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Acquire 3 read locks (read_locks=3)
        manager.reserve_read([key], read_locks=3)

        # Release all 3 at once
        result = manager.finish_read([key], read_locks=3)
        assert result[key] == L1Error.SUCCESS

        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()

    def test_finish_read_partial_read_lock_release(self, basic_l1_config, basic_layout):
        """Partial read-lock release leaves remaining locks."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # 3 locks total (read_locks=3)
        manager.reserve_read([key], read_locks=3)

        # Release 2 of 3 (read_locks=2)
        manager.finish_read([key], read_locks=2)
        state = manager.get_object_state(key)
        assert state is not None
        assert state.read_lock.is_locked()

        # Release last one (read_locks=1)
        manager.finish_read([key])
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.read_lock.is_locked()

        manager.close()

    def test_finish_read_all_read_locks_deletes_temporary(
        self, basic_l1_config, basic_layout
    ):
        """Temp objects deleted when the release covers all read locks."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Acquire 3 read locks at once (read_locks=3)
        manager.reserve_read([key], read_locks=3)
        assert manager.get_object_state(key) is not None

        # Release all 3 at once -> temp deleted
        result = manager.finish_read([key], read_locks=3)
        assert result[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_read_temp_survives_partial_read_locks(
        self, basic_l1_config, basic_layout
    ):
        """Temp object survives a partial share release."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # 4 locks total (read_locks=4)
        manager.reserve_read([key], read_locks=4)

        # Release 2 of 4 -> still locked
        manager.finish_read([key], read_locks=2)
        assert manager.get_object_state(key) is not None

        # Release remaining 2 -> deleted
        manager.finish_read([key], read_locks=2)
        assert manager.get_object_state(key) is None

        manager.close()


# =============================================================================
# Tests for L1Manager.reserve_write()
# =============================================================================


class TestReserveWrite:
    """
    Tests for L1Manager.reserve_write() method.

    Per the docstring:
    - KEY_NOT_WRITABLE: The key exists but is not writable.
    - OUT_OF_MEMORY: Not enough memory to allocate for the object.
    - Returns (L1Error, Optional[MemoryObj]) for each key.
    """

    def test_reserve_write_new_key_returns_success(self, basic_l1_config, basic_layout):
        """Test that reserve_write returns SUCCESS for new keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.reserve_write([key], [False], basic_layout)

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None
        assert mem_obj.is_valid()

        manager.close()

    def test_reserve_write_write_locked_key_returns_not_writable(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_write returns KEY_NOT_WRITABLE for write-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # First reserve write
        result1 = manager.reserve_write([key], [False], basic_layout)
        assert result1[key][0] == L1Error.SUCCESS

        # Try to reserve write again while still write-locked
        result2 = manager.reserve_write([key], [False], basic_layout)

        assert result2[key][0] == L1Error.KEY_NOT_WRITABLE
        assert result2[key][1] is None

        manager.close()

    def test_reserve_write_read_locked_key_returns_not_writable(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_write returns KEY_NOT_WRITABLE for read-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Try to reserve write while read-locked
        result = manager.reserve_write([key], [False], basic_layout)

        assert result[key][0] == L1Error.KEY_NOT_WRITABLE
        assert result[key][1] is None

        manager.close()

    def test_reserve_write_temporary_key_returns_not_writable(
        self, basic_l1_config, basic_layout
    ):
        """Test that reserve_write returns KEY_NOT_WRITABLE for temporary objects."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)  # is_temporary=True
        manager.finish_write([key])

        # Try to reserve write on temporary object
        result = manager.reserve_write([key], [False], basic_layout)

        assert result[key][0] == L1Error.KEY_NOT_WRITABLE

        manager.close()

    def test_reserve_write_multiple_keys(self, basic_l1_config, basic_layout):
        """Test reserve_write with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(5)]
        is_temporary = [False] * 5

        result = manager.reserve_write(keys, is_temporary, basic_layout)

        for key in keys:
            assert key in result
            error, mem_obj = result[key]
            assert error == L1Error.SUCCESS
            assert mem_obj is not None

        manager.close()

    def test_reserve_write_out_of_memory(self, small_l1_config, large_layout):
        """Test that reserve_write returns OUT_OF_MEMORY when allocation fails."""
        manager = L1Manager(small_l1_config)
        keys = [make_object_key(i) for i in range(10)]
        is_temporary = [False] * 10

        # Request more memory than available (8MB * 10 = 80MB > 64MB)
        result = manager.reserve_write(keys, is_temporary, large_layout)

        # All keys should return OUT_OF_MEMORY
        for key in keys:
            assert result[key][0] == L1Error.OUT_OF_MEMORY
            assert result[key][1] is None

        manager.close()

    def test_reserve_write_new_mode(self, basic_l1_config, basic_layout):
        """Test that reserve_write returns KEY_NOT_WRITABLE for existing keys."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(5)]
        is_temporary = [False] * 5

        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="new")

        for key in keys:
            assert result[key][0] == L1Error.SUCCESS
            assert result[key][1] is not None

        # Commit the write
        result = manager.finish_write(keys)
        for key in keys:
            assert result[key] == L1Error.SUCCESS

        # Now try to reserve write again with mode="new"
        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="new")
        for key in keys:
            assert result[key][0] == L1Error.KEY_NOT_WRITABLE
            assert result[key][1] is None

        manager.close()

    def test_reserve_write_update_mode(self, basic_l1_config, basic_layout):
        """Test that reserve_write returns KEY_NOT_WRITABLE for new keys."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(5)]
        is_temporary = [False] * 5

        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="update")
        for key in keys:
            assert result[key][0] == L1Error.KEY_NOT_WRITABLE
            assert result[key][1] is None

        # Cannot finish write in update mode because keys not exist
        result = manager.finish_write(keys)
        for key in keys:
            assert result[key] == L1Error.KEY_NOT_EXIST

        # Now try to reserve write again with mode="new"
        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="new")
        for key in keys:
            assert result[key][0] == L1Error.SUCCESS
            assert result[key][1] is not None

        # Commit the write
        result = manager.finish_write(keys)
        for key in keys:
            assert result[key] == L1Error.SUCCESS

        # Now try to reserve write again with mode="update"
        result = manager.reserve_write(keys, is_temporary, basic_layout, mode="update")
        for key in keys:
            assert result[key][0] == L1Error.SUCCESS
            assert result[key][1] is not None

        manager.close()


# =============================================================================
# Tests for L1Manager.finish_write()
# =============================================================================


class TestFinishWrite:
    """
    Tests for L1Manager.finish_write() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IN_WRONG_STATE: The key is not write-locked, or it's read-locked.
    """

    def test_finish_write_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_write returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.finish_write([key])

        assert key in result
        assert result[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_finish_write_success(self, basic_l1_config, basic_layout):
        """Test that finish_write returns SUCCESS after proper write reservation."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write
        manager.reserve_write([key], [False], basic_layout)

        # Finish write
        result = manager.finish_write([key])

        assert result[key] == L1Error.SUCCESS

        # Verify object is now ready (not write-locked)
        state = manager.get_object_state(key)
        assert state is not None
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        manager.close()

    def test_finish_write_non_write_locked_returns_wrong_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that finish_write returns KEY_IN_WRONG_STATE if not write-locked."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not write-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Try to finish write again
        result = manager.finish_write([key])

        assert result[key] == L1Error.KEY_IN_WRONG_STATE

        manager.close()


# =============================================================================
# Tests for L1Manager.finish_write_and_reserve_read()
# =============================================================================


class TestFinishWriteAndReserveRead:
    """
    Tests for L1Manager.finish_write_and_reserve_read() method.

    This method atomically finishes write and acquires read lock,
    preventing a race window where eviction could interfere.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IN_WRONG_STATE: Not write-locked, or already read-locked.
    - SUCCESS: Write unlocked and read lock acquired atomically.
    """

    def test_normal_transition(self, basic_l1_config, basic_layout):
        """Test normal write-locked -> read-locked transition."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Reserve write (key is now write-locked)
        write_result = manager.reserve_write([key], [False], basic_layout)
        assert write_result[key][0] == L1Error.SUCCESS

        # Atomically finish write and reserve read
        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.SUCCESS
        assert mem_obj is not None

        # Verify state: write unlocked, read locked
        state = manager.get_object_state(key)
        assert state is not None
        assert not state.write_lock.is_locked()
        assert state.read_lock.is_locked()

        # Should be readable (not write-locked)
        assert state.available_for_read() is True
        # Should not be writable (read-locked)
        assert state.available_for_write() is False

        # Clean up read lock
        manager.finish_read([key])
        manager.close()

    def test_key_not_exist(self, basic_l1_config, basic_layout):
        """Test that non-existing key returns KEY_NOT_EXIST."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_NOT_EXIST
        assert mem_obj is None

        manager.close()

    def test_not_write_locked(self, basic_l1_config, basic_layout):
        """Test that non-write-locked key returns KEY_IN_WRONG_STATE."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object (not write-locked)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_IN_WRONG_STATE
        assert mem_obj is None

        manager.close()

    def test_already_read_locked(self, basic_l1_config, basic_layout):
        """Test that key with both write+read locks returns KEY_IN_WRONG_STATE.

        This is an unexpected state — normally a key shouldn't be both
        write-locked and read-locked simultaneously.
        """
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Resident object write-locked in place (only resident objects can
        # carry both locks; staging objects have no readers).
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_write([key], [False], basic_layout, mode="update")

        # Force a read lock via internal state (unusual state)
        state = manager.get_object_state(key)
        assert state is not None
        state.read_lock.lock()

        result = manager.finish_write_and_reserve_read([key])

        assert key in result
        error, mem_obj = result[key]
        assert error == L1Error.KEY_IN_WRONG_STATE
        assert mem_obj is None

        # Clean up
        state.read_lock.unlock()
        manager.close()

    def test_multiple_keys_mixed_results(self, basic_l1_config, basic_layout):
        """Test with multiple keys where some succeed and some fail."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)  # will not exist
        key3 = make_object_key(3)

        # key1: write-locked (should succeed)
        manager.reserve_write([key1], [False], basic_layout)

        # key3: ready, not write-locked (should fail)
        manager.reserve_write([key3], [False], basic_layout)
        manager.finish_write([key3])

        result = manager.finish_write_and_reserve_read([key1, key2, key3])

        # key1: SUCCESS
        assert result[key1][0] == L1Error.SUCCESS
        assert result[key1][1] is not None

        # key2: KEY_NOT_EXIST
        assert result[key2][0] == L1Error.KEY_NOT_EXIST
        assert result[key2][1] is None

        # key3: KEY_IN_WRONG_STATE (not write-locked)
        assert result[key3][0] == L1Error.KEY_IN_WRONG_STATE
        assert result[key3][1] is None

        # Clean up
        manager.finish_read([key1])
        manager.close()

    def test_can_unsafe_read_after_transition(self, basic_l1_config, basic_layout):
        """Test that unsafe_read works on the transitioned key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Write and transition
        manager.reserve_write([key], [False], basic_layout)
        result = manager.finish_write_and_reserve_read([key])
        assert result[key][0] == L1Error.SUCCESS

        # unsafe_read should work (key is read-locked)
        read_result = manager.unsafe_read([key])
        assert read_result[key][0] == L1Error.SUCCESS
        assert read_result[key][1] is not None

        manager.finish_read([key])
        manager.close()


# =============================================================================
# Tests for L1Manager.finish_write_and_delete()
# =============================================================================


class TestFinishWriteAndDelete:
    """Tests for L1Manager.finish_write_and_delete()."""

    def test_deletes_write_locked_keys_only(self, basic_l1_config, basic_layout):
        """Write-locked keys are deleted; other states error out intact."""
        manager = L1Manager(basic_l1_config)
        locked_key = make_object_key(1)
        ready_key = make_object_key(2)
        missing_key = make_object_key(3)

        manager.reserve_write([locked_key, ready_key], [False, False], basic_layout)
        manager.finish_write([ready_key])

        result = manager.finish_write_and_delete([locked_key, ready_key, missing_key])

        assert result[locked_key] == L1Error.SUCCESS
        assert manager.get_object_state(locked_key) is None
        assert result[ready_key] == L1Error.KEY_IN_WRONG_STATE
        assert manager.get_object_state(ready_key) is not None
        assert result[missing_key] == L1Error.KEY_NOT_EXIST

        manager.close()


# =============================================================================
# Tests for L1Manager.delete()
# =============================================================================


class TestDelete:
    """
    Tests for L1Manager.delete() method.

    Per the docstring:
    - KEY_NOT_EXIST: The key does not exist.
    - KEY_IS_LOCKED: The key is locked (either write-locked or read-locked).
    """

    def test_delete_non_existing_key_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """Test that delete returns KEY_NOT_EXIST for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        result = manager.delete([key])

        assert key in result
        assert result[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_delete_success(self, basic_l1_config, basic_layout):
        """Test that delete returns SUCCESS for unlocked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Verify object exists
        assert manager.get_object_state(key) is not None

        # Delete
        result = manager.delete([key])

        assert result[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_delete_write_locked_returns_key_is_locked(
        self, basic_l1_config, basic_layout
    ):
        """Test that delete returns KEY_IS_LOCKED for write-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create write-locked object
        manager.reserve_write([key], [False], basic_layout)

        # Try to delete
        result = manager.delete([key])

        assert result[key] == L1Error.KEY_IS_LOCKED

        manager.close()

    def test_delete_read_locked_returns_key_is_locked(
        self, basic_l1_config, basic_layout
    ):
        """Test that delete returns KEY_IS_LOCKED for read-locked keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Try to delete
        result = manager.delete([key])

        assert result[key] == L1Error.KEY_IS_LOCKED

        manager.close()

    def test_delete_multiple_keys(self, basic_l1_config, basic_layout):
        """Test delete with multiple keys in a single call."""
        manager = L1Manager(basic_l1_config)
        key1 = make_object_key(1)
        key2 = make_object_key(2)
        key3 = make_object_key(3)

        # key1: ready (unlocked)
        manager.reserve_write([key1], [False], basic_layout)
        manager.finish_write([key1])

        # key2: does not exist

        # key3: write-locked
        manager.reserve_write([key3], [False], basic_layout)

        result = manager.delete([key1, key2, key3])

        assert result[key1] == L1Error.SUCCESS
        assert result[key2] == L1Error.KEY_NOT_EXIST
        assert result[key3] == L1Error.KEY_IS_LOCKED

        manager.close()

    def test_force_delete_removes_write_locked_key(self, basic_l1_config, basic_layout):
        """force=True deletes a write-locked key that non-force refuses."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout)  # write-locked

        assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}
        assert manager.delete([key], force=True) == {key: L1Error.SUCCESS}
        assert manager.get_object_state(key) is None

        manager.close()

    def test_force_delete_removes_read_locked_key(self, basic_l1_config, basic_layout):
        """force=True deletes a read-locked key that non-force refuses."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])  # read-locked

        assert manager.delete([key]) == {key: L1Error.KEY_IS_LOCKED}
        assert manager.delete([key], force=True) == {key: L1Error.SUCCESS}
        assert manager.get_object_state(key) is None

        manager.close()

    def test_force_delete_missing_key_still_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """force does not invent keys: a missing key still reports KEY_NOT_EXIST."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(999)

        assert manager.delete([key], force=True) == {key: L1Error.KEY_NOT_EXIST}

        manager.close()


# =============================================================================
# Tests for L1Manager.get_object_state()
# =============================================================================


class TestGetObjectState:
    """
    Tests for L1Manager.get_object_state() method.

    Per the docstring:
    - Returns the L1ObjectState if the object exists, None otherwise.
    """

    def test_get_object_state_non_existing_returns_none(
        self, basic_l1_config, basic_layout
    ):
        """Test that get_object_state returns None for non-existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        state = manager.get_object_state(key)

        assert state is None

        manager.close()

    def test_get_object_state_existing_returns_state(
        self, basic_l1_config, basic_layout
    ):
        """Test that get_object_state returns L1ObjectState for existing keys."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        state = manager.get_object_state(key)

        assert state is not None
        # Verify we can use the state's methods
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        manager.close()

    def test_get_object_state_staged_returns_none(self, basic_l1_config, basic_layout):
        """A staging object is not reported as the key's object."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [False], basic_layout)

        assert manager.get_object_state(key) is None
        assert manager.report_status()["staging_object_count"] == 1

        manager.close()

    def test_get_object_state_in_place_write_locked(
        self, basic_l1_config, basic_layout
    ):
        """Test get_object_state for objects write-locked in place."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_write([key], [False], basic_layout, mode="update")

        state = manager.get_object_state(key)

        assert state is not None
        assert state.available_for_read() is False
        assert state.available_for_write() is False

        manager.finish_write([key])
        manager.close()

    def test_get_object_state_read_locked(self, basic_l1_config, basic_layout):
        """Test get_object_state for read-locked objects."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object, then read lock it
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])

        state = manager.get_object_state(key)

        assert state is not None
        # Read-locked is still readable
        assert state.available_for_read() is True
        # But not writable
        assert state.available_for_write() is False

        manager.close()


# =============================================================================
# Tests for L1Manager.close()
# =============================================================================


class TestClose:
    """
    Tests for L1Manager.close() method.

    Per the docstring:
    - Close the L1Manager and free all resources.
    """

    def test_close_empty_manager(self, basic_l1_config):
        """Test that close works on an empty manager."""
        manager = L1Manager(basic_l1_config)

        # Should not raise any exceptions
        manager.close()

    def test_close_with_objects(self, basic_l1_config, basic_layout):
        """Test that close frees all objects in the manager."""
        manager = L1Manager(basic_l1_config)

        # Create multiple objects
        keys = [make_object_key(i) for i in range(5)]
        manager.reserve_write(keys, [False] * 5, basic_layout)

        # Close should free all objects
        manager.close()

    def test_close_clears_objects(self, basic_l1_config, basic_layout):
        """Test that close clears all objects from the manager."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Verify object exists before close
        assert manager.get_object_state(key) is not None

        # Close
        manager.close()

        # After close, get_object_state should return None
        # (objects dict should be cleared)
        assert manager.get_object_state(key) is None


# =============================================================================
# Tests for state machine transitions (integration)
# =============================================================================


class TestStateMachineTransitions:
    """
    Integration tests verifying the state machine transitions as described in the
    L1Manager class docstring.
    """

    def test_full_write_read_cycle(self, basic_l1_config, basic_layout):
        """Test full cycle: None -> write_locked -> ready -> read_locked -> ready."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # None state (key doesn't exist)
        assert manager.get_object_state(key) is None

        # reserve_write: None -> write_locked staging object (not resident)
        result = manager.reserve_write([key], [False], basic_layout)
        assert result[key][0] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None
        assert manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST
        assert manager.report_status()["staging_object_count"] == 1

        # finish_write: write_locked -> ready (admission)
        result = manager.finish_write([key])
        assert result[key] == L1Error.SUCCESS
        state = manager.get_object_state(key)
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        # reserve_read: ready -> read_locked
        result = manager.reserve_read([key])
        assert result[key][0] == L1Error.SUCCESS
        state = manager.get_object_state(key)
        assert state.available_for_read() is True
        assert state.available_for_write() is False

        # finish_read: read_locked -> ready
        result = manager.finish_read([key])
        assert result[key] == L1Error.SUCCESS
        state = manager.get_object_state(key)
        assert state.available_for_read() is True
        assert state.available_for_write() is True

        manager.close()

    def test_full_write_read_with_unsafe_read(self, basic_l1_config, basic_layout):
        """Test cycle with unsafe_read between reserve_read and finish_read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read
        reserve_result = manager.reserve_read([key])
        assert reserve_result[key][0] == L1Error.SUCCESS

        # Multiple unsafe_reads should all succeed
        for _ in range(3):
            unsafe_result = manager.unsafe_read([key])
            assert unsafe_result[key][0] == L1Error.SUCCESS

        # Finish read
        finish_result = manager.finish_read([key])
        assert finish_result[key] == L1Error.SUCCESS

        # Object should still exist (not temporary)
        assert manager.get_object_state(key) is not None

        manager.close()

    def test_delete_from_ready_state(self, basic_l1_config, basic_layout):
        """Test deletion from ready state."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # delete: ready -> None
        result = manager.delete([key])
        assert result[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_temporary_object_lifecycle(self, basic_l1_config, basic_layout):
        """Test temporary object lifecycle: deleted after last read."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Read and release
        manager.reserve_read([key])
        manager.finish_read([key])

        # Object should be deleted
        assert manager.get_object_state(key) is None

        manager.close()

    def test_temporary_object_with_unsafe_read(self, basic_l1_config, basic_layout):
        """Test temporary object with unsafe_read doesn't affect deletion."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create temporary object
        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        # Reserve read
        manager.reserve_read([key])

        # Multiple unsafe_reads
        for _ in range(5):
            result = manager.unsafe_read([key])
            assert result[key][0] == L1Error.SUCCESS

        # Single finish_read should delete the object
        manager.finish_read([key])
        assert manager.get_object_state(key) is None

        manager.close()

    def test_multi_reader_lifecycle_with_shared_key(
        self, basic_l1_config, basic_layout
    ):
        """Full lifecycle of a shared key (MLA TP>1 scenario).

        Simulates multiple workers sharing the same key:
        reserve_read(read_locks=N) acquires N locks,
        finish_read(read_locks=N) releases them all.
        """
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)
        read_locks = 4

        # write -> ready
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        result = manager.reserve_read([key], read_locks=read_locks)
        assert result[key][0] == L1Error.SUCCESS

        # unsafe_read should work while read-locked
        ur = manager.unsafe_read([key])
        assert ur[key][0] == L1Error.SUCCESS

        # finish_read releasing the whole reservation
        fr = manager.finish_read([key], read_locks=read_locks)
        assert fr[key] == L1Error.SUCCESS

        # All locks released -> writable again
        state = manager.get_object_state(key)
        assert state is not None
        assert state.available_for_write() is True

        manager.close()

    def test_temp_object_multi_reader_deletion(self, basic_l1_config, basic_layout):
        """Temporary object deleted after all read locks are released."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)
        read_locks = 3

        manager.reserve_write([key], [True], basic_layout)
        manager.finish_write([key])

        manager.reserve_read([key], read_locks=read_locks)
        assert manager.get_object_state(key) is not None

        manager.finish_read([key], read_locks=read_locks)
        assert manager.get_object_state(key) is None

        manager.close()


# =============================================================================
# Thread safety tests
# =============================================================================


class TestThreadSafety:
    """Tests verifying thread-safety of L1Manager operations."""

    def test_concurrent_reserve_write_different_keys(
        self, basic_l1_config, basic_layout
    ):
        """Test concurrent reserve_write on different keys."""
        manager = L1Manager(basic_l1_config)
        num_threads = 8
        keys_per_thread = 5
        results = []
        lock = threading.Lock()

        def worker(thread_id):
            thread_keys = [
                make_object_key(thread_id * 1000 + i) for i in range(keys_per_thread)
            ]
            is_temporary = [False] * keys_per_thread
            result = manager.reserve_write(thread_keys, is_temporary, basic_layout)
            with lock:
                results.append((thread_keys, result))

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All writes should succeed
        assert len(results) == num_threads
        for keys, result in results:
            for key in keys:
                assert result[key][0] == L1Error.SUCCESS

        manager.close()

    def test_concurrent_read_same_key(self, basic_l1_config, basic_layout):
        """Test concurrent reads on the same key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        num_threads = 10
        results = []
        lock = threading.Lock()

        def worker():
            result = manager.reserve_read([key])
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(results) == num_threads
        for result in results:
            assert result[key][0] == L1Error.SUCCESS

        manager.close()

    def test_concurrent_unsafe_read_same_key(self, basic_l1_config, basic_layout):
        """Test concurrent unsafe_reads on the same read-locked key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(12345)

        # Create ready object and reserve read
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])

        num_threads = 10
        results = []
        lock = threading.Lock()

        def worker():
            result = manager.unsafe_read([key])
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All unsafe_reads should succeed
        assert len(results) == num_threads
        for result in results:
            assert result[key][0] == L1Error.SUCCESS

        manager.close()

    def test_concurrent_read_write_mixed_operations(
        self, basic_l1_config, basic_layout
    ):
        """Test concurrent mixed operations don't cause crashes."""
        manager = L1Manager(basic_l1_config)
        num_threads = 8
        operations_per_thread = 10
        errors = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                for i in range(operations_per_thread):
                    key = make_object_key(thread_id * 1000 + i)

                    # Write cycle
                    write_result = manager.reserve_write([key], [False], basic_layout)
                    if write_result[key][0] == L1Error.SUCCESS:
                        manager.finish_write([key])

                        # Read cycle with unsafe_read
                        read_result = manager.reserve_read([key])
                        if read_result[key][0] == L1Error.SUCCESS:
                            # Do some unsafe_reads
                            manager.unsafe_read([key])
                            manager.unsafe_read([key])
                            manager.finish_read([key])

                        # Delete
                        manager.delete([key])
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        manager.close()


# =============================================================================
# Tests for L1Manager.is_key_evictable()
# =============================================================================


class TestIsKeyEvictable:
    """Tests for L1Manager.is_key_evictable() method."""

    def test_evictable_key_in_ready_state(self, basic_l1_config, basic_layout):
        """A key that has been written and finished should be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write and finish -> key is in ready state
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        assert manager.is_key_evictable(key) is True

        manager.close()

    def test_write_locked_key_is_not_evictable(self, basic_l1_config, basic_layout):
        """A write-locked key should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Reserve write but don't finish -> key is write-locked
        manager.reserve_write([key], [False], basic_layout)

        assert manager.is_key_evictable(key) is False

        # Cleanup
        manager.finish_write([key])
        manager.close()

    def test_read_locked_key_is_not_evictable(self, basic_l1_config, basic_layout):
        """A read-locked key should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write, finish, then reserve read -> key is read-locked
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.reserve_read([key])

        assert manager.is_key_evictable(key) is False

        # Cleanup
        manager.finish_read([key])
        manager.close()

    def test_nonexistent_key_is_not_evictable(self, basic_l1_config):
        """A key that does not exist should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(999)

        assert manager.is_key_evictable(key) is False

        manager.close()

    def test_key_becomes_evictable_after_read_unlock(
        self, basic_l1_config, basic_layout
    ):
        """A key should become evictable after all read locks are released."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write and finish
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read -> not evictable
        manager.reserve_read([key])
        assert manager.is_key_evictable(key) is False

        # Finish read -> evictable again
        manager.finish_read([key])
        assert manager.is_key_evictable(key) is True

        manager.close()

    def test_key_becomes_evictable_after_write_unlock(
        self, basic_l1_config, basic_layout
    ):
        """A key should become evictable after write lock is released."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Reserve write -> not evictable
        manager.reserve_write([key], [False], basic_layout)
        assert manager.is_key_evictable(key) is False

        # Finish write -> evictable
        manager.finish_write([key])
        assert manager.is_key_evictable(key) is True

        manager.close()

    def test_multiple_keys_mixed_evictability(self, basic_l1_config, basic_layout):
        """Test evictability with multiple keys in different states."""
        manager = L1Manager(basic_l1_config)
        key_ready = make_object_key(1)
        key_write_locked = make_object_key(2)
        key_read_locked = make_object_key(3)

        # key_ready: write + finish -> ready state
        manager.reserve_write([key_ready], [False], basic_layout)
        manager.finish_write([key_ready])

        # key_write_locked: write only -> write-locked
        manager.reserve_write([key_write_locked], [False], basic_layout)

        # key_read_locked: write + finish + read -> read-locked
        manager.reserve_write([key_read_locked], [False], basic_layout)
        manager.finish_write([key_read_locked])
        manager.reserve_read([key_read_locked])

        assert manager.is_key_evictable(key_ready) is True
        assert manager.is_key_evictable(key_write_locked) is False
        assert manager.is_key_evictable(key_read_locked) is False

        # Cleanup
        manager.finish_write([key_write_locked])
        manager.finish_read([key_read_locked])
        manager.close()

    def test_deleted_key_is_not_evictable(self, basic_l1_config, basic_layout):
        """A key that has been deleted should not be evictable."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write, finish, then delete
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])
        manager.delete([key])

        assert manager.is_key_evictable(key) is False

        manager.close()

    def test_key_with_multiple_read_locks_not_evictable(
        self, basic_l1_config, basic_layout
    ):
        """A key with multiple read locks should not be evictable
        until all locks are released."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        # Write and finish
        manager.reserve_write([key], [False], basic_layout)
        manager.finish_write([key])

        # Reserve read with read_locks=3
        manager.reserve_read([key], read_locks=3)
        assert manager.is_key_evictable(key) is False

        # Release only 1 read lock -> still locked
        manager.finish_read([key], read_locks=1)
        assert manager.is_key_evictable(key) is False

        # Release the remaining 2 read locks
        manager.finish_read([key], read_locks=2)
        assert manager.is_key_evictable(key) is True

        manager.close()


# =============================================================================
# Tests for staging objects (write tags)
# =============================================================================


def write_ready(manager: L1Manager, keys, layout, is_temporary=False):
    """Stage and admit ``keys`` so they are resident and unlocked."""
    result = manager.reserve_write(keys, [is_temporary] * len(keys), layout)
    for key in keys:
        assert result[key][0] == L1Error.SUCCESS
    result = manager.finish_write(keys)
    for key in keys:
        assert result[key] == L1Error.SUCCESS


class TestStagingReservation:
    """reserve_write on non-resident keys creates tagged staging objects."""

    def test_same_tag_twice_returns_not_writable(self, basic_l1_config, basic_layout):
        """One tag holds at most one staging object per key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        first = manager.reserve_write([key], [False], basic_layout, tag="w")
        second = manager.reserve_write([key], [False], basic_layout, tag="w")

        assert first[key][0] == L1Error.SUCCESS
        assert second[key] == (L1Error.KEY_NOT_WRITABLE, None)

        manager.close()

    def test_different_tags_stage_same_key(self, basic_l1_config, basic_layout):
        """Writers with different tags get independent buffers for one key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)

        result_a = manager.reserve_write([key], [False], basic_layout, tag="a")
        result_b = manager.reserve_write([key], [False], basic_layout, tag="b")
        result_default = manager.reserve_write([key], [False], basic_layout)

        for result in (result_a, result_b, result_default):
            assert result[key][0] == L1Error.SUCCESS
            assert result[key][1] is not None
        assert result_a[key][1] is not result_b[key][1]
        assert result_a[key][1] is not result_default[key][1]
        assert manager.get_object_state(key) is None
        status = manager.report_status()
        assert status["staging_object_count"] == 3
        assert status["total_object_count"] == 3
        assert status["write_locked_count"] == 3

        manager.close()

    def test_new_mode_refuses_resident_key_for_any_tag(
        self, basic_l1_config, basic_layout
    ):
        """Visibility is tag-independent: mode="new" fails for a resident key."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        write_ready(manager, [key], basic_layout)

        result = manager.reserve_write(
            [key], [False], basic_layout, mode="new", tag="other"
        )

        assert result[key] == (L1Error.KEY_NOT_WRITABLE, None)

        manager.close()

    def test_refused_keys_do_not_consume_memory(self, basic_l1_config, basic_layout):
        """Only the newly staged keys are allocated."""
        manager = L1Manager(basic_l1_config)
        resident = make_object_key(1)
        staged = make_object_key(2)
        fresh = make_object_key(3)
        write_ready(manager, [resident], basic_layout)
        manager.reserve_write([staged], [False], basic_layout, tag="w")
        before = manager.get_staging_memory_usage()

        result = manager.reserve_write(
            [resident, staged, fresh], [False] * 3, basic_layout, mode="new", tag="w"
        )

        assert result[resident] == (L1Error.KEY_NOT_WRITABLE, None)
        assert result[staged] == (L1Error.KEY_NOT_WRITABLE, None)
        assert result[fresh][0] == L1Error.SUCCESS
        size = result[fresh][1].get_size()
        assert manager.get_staging_memory_usage() == before + size

        manager.close()

    def test_rejects_mismatched_lengths(self, basic_l1_config, basic_layout):
        """keys and is_temporary must have the same length."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(2)]

        with pytest.raises(ValueError):
            manager.reserve_write(keys, [False], basic_layout)

        manager.close()


class TestStagingAdmission:
    """finish_write variants admit or discard staging objects."""

    def test_finish_write_wrong_tag_returns_key_not_exist(
        self, basic_l1_config, basic_layout
    ):
        """A tag can only admit its own staging objects."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout, tag="a")

        assert manager.finish_write([key], tag="b")[key] == L1Error.KEY_NOT_EXIST
        assert manager.get_object_state(key) is None
        assert manager.finish_write([key], tag="a")[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is not None

        manager.close()

    def test_first_admission_wins_and_later_copy_is_freed(
        self, basic_l1_config, basic_layout
    ):
        """The resident object is kept; the late staging copy is discarded."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        first = manager.reserve_write([key], [False], basic_layout, tag="a")
        second = manager.reserve_write([key], [True], basic_layout, tag="b")
        used_before, _ = manager.get_memory_usage()

        assert manager.finish_write([key], tag="a")[key] == L1Error.SUCCESS
        assert manager.finish_write([key], tag="b")[key] == L1Error.SUCCESS

        state = manager.get_object_state(key)
        assert state is not None
        assert state.memory_obj is first[key][1]
        assert state.is_temporary is False
        assert manager.get_staging_memory_usage() == 0
        used_after, _ = manager.get_memory_usage()
        assert used_after == used_before - second[key][1].get_size()

        manager.close()

    def test_discard_keeps_resident_read_locks(self, basic_l1_config, basic_layout):
        """Discarding a late copy never disturbs readers of the resident object."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        write_ready(manager, [key], basic_layout)
        read = manager.reserve_read([key], read_locks=2)
        # mode="new" refuses resident keys; stage the late copy first.
        manager.delete([key])  # refused: read-locked
        assert manager.get_object_state(key) is not None

        late = manager.reserve_write([key], [False], basic_layout, tag="late")
        # The key is resident, so "all" mode write-locks it in place instead of
        # staging; a read-locked key cannot be write-locked -> NOT_WRITABLE.
        assert late[key] == (L1Error.KEY_NOT_WRITABLE, None)
        assert manager.unsafe_read([key])[key][1] is read[key][1]

        manager.finish_read([key], read_locks=2)
        manager.close()

    def test_live_staging_pins_resident_key(self, basic_l1_config, basic_layout):
        """A key with a live staging object cannot be deleted, even when its
        resident object is unlocked; it can once the reservation is gone."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout, tag="w")
        write_ready(manager, [key], basic_layout)

        assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
        assert manager.get_object_state(key) is not None
        assert manager.report_status()["staging_object_count"] == 1

        assert manager.finish_write_and_delete([key], tag="w")[key] == L1Error.SUCCESS
        assert manager.delete([key])[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_write_and_reserve_read_locks_resident_copy(
        self, basic_l1_config, basic_layout
    ):
        """Two tagged writers: the late one gets read locks on the resident
        object and its own buffer is freed (the concurrent-prefetch case)."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        first = manager.reserve_write([key], [True], basic_layout, tag="p1")
        second = manager.reserve_write([key], [True], basic_layout, tag="p2")
        assert first[key][0] == L1Error.SUCCESS
        assert second[key][0] == L1Error.SUCCESS
        assert manager.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST

        r1 = manager.finish_write_and_reserve_read([key], read_locks=2, tag="p1")
        r2 = manager.finish_write_and_reserve_read([key], read_locks=2, tag="p2")

        assert r1[key][0] == L1Error.SUCCESS
        assert r2[key][0] == L1Error.SUCCESS
        assert r1[key][1] is first[key][1]
        assert r2[key][1] is first[key][1]
        assert manager.get_staging_memory_usage() == 0
        assert manager.unsafe_read([key])[key][0] == L1Error.SUCCESS

        # Four read locks in total; the temporary object lives until all are
        # released.
        assert manager.finish_read([key], read_locks=2)[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is not None
        assert manager.finish_read([key], read_locks=2)[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is None

        manager.close()

    def test_finish_write_and_delete_discards_own_tag_only(
        self, basic_l1_config, basic_layout
    ):
        """Discarding under one tag leaves other tags' staging objects alone."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout, tag="a")
        manager.reserve_write([key], [False], basic_layout, tag="b")

        assert manager.finish_write_and_delete([key], tag="a")[key] == L1Error.SUCCESS
        assert manager.finish_write_and_delete([key], tag="a")[key] == (
            L1Error.KEY_NOT_EXIST
        )
        assert manager.report_status()["staging_object_count"] == 1
        assert manager.finish_write([key], tag="b")[key] == L1Error.SUCCESS
        assert manager.get_object_state(key) is not None

        manager.close()

    def test_partial_admission(self, basic_l1_config, basic_layout):
        """Admitting a subset leaves the rest staged."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(3)]
        manager.reserve_write(keys, [False] * 3, basic_layout)

        result = manager.finish_write(keys[:1])

        assert result[keys[0]] == L1Error.SUCCESS
        assert manager.get_object_state(keys[0]) is not None
        for key in keys[1:]:
            assert manager.get_object_state(key) is None
        status = manager.report_status()
        assert status["total_object_count"] == 3
        assert status["staging_object_count"] == 2

        manager.close()


class TestStagingEviction:
    """Abandoned reservations are reclaimed through delete / clear / eviction."""

    def test_live_staging_is_locked(self, basic_l1_config, basic_layout):
        """delete refuses a key whose only object is a live staging object."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout)

        assert manager.is_key_evictable(key) is False
        assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
        assert manager.report_status()["staging_object_count"] == 1

        manager.close()

    def test_expired_staging_is_evictable_and_reclaimed(
        self, short_write_ttl_l1_config, basic_layout
    ):
        """An expired reservation makes the key evictable; delete reclaims it."""
        manager = L1Manager(short_write_ttl_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout)
        time.sleep(1.2)

        assert manager.is_key_evictable(key) is True
        assert manager.delete([key])[key] == L1Error.SUCCESS
        assert manager.get_staging_memory_usage() == 0
        assert manager.report_status()["staging_object_count"] == 0
        used, _ = manager.get_memory_usage()
        assert used == 0
        assert manager.delete([key])[key] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_delete_reclaims_expired_but_keeps_live_staging(
        self, short_write_ttl_l1_config, basic_layout
    ):
        """Mixed tags: expired ones go, the live one stays and locks the key."""
        manager = L1Manager(short_write_ttl_l1_config)
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout, tag="stale")
        time.sleep(1.2)
        live = manager.reserve_write([key], [False], basic_layout, tag="live")
        assert live[key][0] == L1Error.SUCCESS

        assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED

        assert manager.report_status()["staging_object_count"] == 1
        assert manager.get_staging_memory_usage() == live[key][1].get_size()
        assert manager.finish_write([key], tag="live")[key] == L1Error.SUCCESS

        manager.close()

    def test_force_delete_discards_live_staging(self, basic_l1_config, basic_layout):
        """force=True drops live staging objects as well as the resident one."""
        manager = L1Manager(basic_l1_config)
        key = make_object_key(1)
        write_ready(manager, [key], basic_layout)
        manager.reserve_write([key], [False], basic_layout, mode="update")
        # In-place write lock on the resident object plus a staged copy under
        # another tag for a second key.
        other = make_object_key(2)
        manager.reserve_write([other], [False], basic_layout, tag="a")
        manager.reserve_write([other], [False], basic_layout, tag="b")

        assert manager.delete([key, other], force=True) == {
            key: L1Error.SUCCESS,
            other: L1Error.SUCCESS,
        }

        assert manager.get_object_state(key) is None
        assert manager.get_staging_memory_usage() == 0
        used, _ = manager.get_memory_usage()
        assert used == 0
        assert manager.finish_write([other], tag="a")[other] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_clear_keeps_live_staging_and_reclaims_expired(
        self, short_write_ttl_l1_config, basic_layout
    ):
        """Non-force clear frees expired staging objects only."""
        manager = L1Manager(short_write_ttl_l1_config)
        stale, live = make_object_key(1), make_object_key(2)
        manager.reserve_write([stale], [False], basic_layout)
        time.sleep(1.2)
        manager.reserve_write([live], [False], basic_layout)

        manager.clear()

        status = manager.report_status()
        assert status["staging_object_count"] == 1
        assert manager.finish_write([live])[live] == L1Error.SUCCESS
        assert manager.finish_write([stale])[stale] == L1Error.KEY_NOT_EXIST

        manager.close()

    def test_force_clear_frees_staging(self, basic_l1_config, basic_layout):
        """Force clear drops every staging object."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(3)]
        manager.reserve_write(keys, [False] * 3, basic_layout, tag="w")

        manager.clear(force=True)

        status = manager.report_status()
        assert status["total_object_count"] == 0
        assert status["staging_object_count"] == 0
        assert status["staging_bytes"] == 0
        assert status["memory_used_bytes"] == 0

        manager.close()

    def test_eviction_policy_reclaims_abandoned_reservation(
        self, short_write_ttl_l1_config, basic_layout
    ):
        """End to end through the listener: the LRU policy learns the key at
        reservation time, skips it while the write lock is live, and evicts
        it once the lock expired -- passing the original key to delete."""
        manager = L1Manager(short_write_ttl_l1_config)
        policy = LRUEvictionPolicy()
        manager.register_listener(L1EvictionPolicy(policy))
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout, tag="abandoned")

        # Live reservation: tracked but not eligible.
        actions = policy.get_eviction_actions(
            1.0, key_eligible_filter=manager.is_key_evictable
        )
        assert [k for a in actions for k in a.keys] == []

        time.sleep(1.2)
        actions = policy.get_eviction_actions(
            1.0, key_eligible_filter=manager.is_key_evictable
        )
        evicted = [k for a in actions for k in a.keys]
        assert evicted == [key]
        assert manager.delete(evicted)[key] == L1Error.SUCCESS
        assert manager.get_staging_memory_usage() == 0

        # The policy forgot the key: nothing left to evict.
        actions = policy.get_eviction_actions(
            1.0, key_eligible_filter=manager.is_key_evictable
        )
        assert [k for a in actions for k in a.keys] == []

        manager.close()

    def test_eviction_policy_forgets_discarded_reservation(
        self, basic_l1_config, basic_layout
    ):
        """A key dropped via finish_write_and_delete leaves the policy too."""
        manager = L1Manager(basic_l1_config)
        policy = LRUEvictionPolicy()
        manager.register_listener(L1EvictionPolicy(policy))
        key = make_object_key(1)
        manager.reserve_write([key], [False], basic_layout, tag="p")
        manager.finish_write_and_delete([key], tag="p")

        actions = policy.get_eviction_actions(1.0)

        assert [k for a in actions for k in a.keys] == []
        manager.close()


class TestStagingAccounting:
    """get_staging_memory_usage() and report_status() staging fields."""

    def test_staging_bytes_follow_lifecycle(self, basic_l1_config, basic_layout):
        """Staging bytes grow on reserve and drop on admission or discard,
        while allocator usage only drops on discard."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(4)]
        assert manager.get_staging_memory_usage() == 0

        reserved = manager.reserve_write(keys, [False] * 4, basic_layout, tag="w")
        size = reserved[keys[0]][1].get_size()
        assert manager.get_staging_memory_usage() == 4 * size
        used, _ = manager.get_memory_usage()
        assert used >= 4 * size

        manager.finish_write(keys[:2], tag="w")
        assert manager.get_staging_memory_usage() == 2 * size

        manager.finish_write_and_reserve_read(keys[2:3], tag="w")
        assert manager.get_staging_memory_usage() == size

        manager.finish_write_and_delete(keys[3:], tag="w")
        assert manager.get_staging_memory_usage() == 0
        used_after, _ = manager.get_memory_usage()
        assert used_after == used - size

        manager.finish_read(keys[2:3])
        manager.close()

    def test_report_status_counts(self, basic_l1_config, basic_layout):
        """report_status separates resident, read-locked and staging objects."""
        manager = L1Manager(basic_l1_config)
        ready, locked, temp = (make_object_key(i) for i in range(3))
        staged = [make_object_key(10 + i) for i in range(2)]
        write_ready(manager, [ready, locked], basic_layout)
        write_ready(manager, [temp], basic_layout, is_temporary=True)
        manager.reserve_read([locked])
        manager.reserve_read([temp])
        manager.reserve_write(staged, [False] * 2, basic_layout, tag="a")
        manager.reserve_write(staged[:1], [True], basic_layout, tag="b")

        status = manager.report_status()

        assert status["total_object_count"] == 6
        assert status["read_locked_count"] == 2
        assert status["temporary_count"] == 2
        assert status["staging_object_count"] == 3
        assert status["write_locked_count"] == 3
        assert status["staging_bytes"] == manager.get_staging_memory_usage()
        assert status["staging_bytes"] > 0
        assert status["memory_used_bytes"] >= status["staging_bytes"]

        manager.finish_read([locked])
        manager.finish_read([temp])
        manager.close()

    def test_concurrent_tagged_writers_same_keys(self, basic_l1_config, basic_layout):
        """Many tagged writers race on the same keys: a reservation either
        succeeds or finds the key already resident, every admission
        succeeds, and exactly one copy per key survives."""
        manager = L1Manager(basic_l1_config)
        keys = [make_object_key(i) for i in range(4)]
        num_threads = 8
        errors = []
        lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def worker(thread_id):
            tag = f"writer-{thread_id}"
            try:
                barrier.wait()
                reserved = manager.reserve_write(
                    keys, [False] * len(keys), basic_layout, "new", tag
                )
                mine = [k for k in keys if reserved[k][0] == L1Error.SUCCESS]
                bad = [
                    k
                    for k in keys
                    if reserved[k][0] not in (L1Error.SUCCESS, L1Error.KEY_NOT_WRITABLE)
                ]
                if bad:
                    raise AssertionError(f"{tag}: reserve failed for {bad}")
                # KEY_NOT_WRITABLE means another writer already admitted the
                # key; the rest must admit (or discard) successfully.
                finished = manager.finish_write_and_reserve_read(mine, tag=tag)
                bad = [k for k in mine if finished[k][0] != L1Error.SUCCESS]
                if bad:
                    raise AssertionError(f"{tag}: finish failed for {bad}")
                manager.finish_read(mine)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        status = manager.report_status()
        assert status["total_object_count"] == len(keys)
        assert status["staging_object_count"] == 0
        assert status["read_locked_count"] == 0
        assert manager.get_staging_memory_usage() == 0
        used, _ = manager.get_memory_usage()
        one_copy = manager.get_object_state(keys[0]).memory_obj.get_size()
        assert used == one_copy * len(keys)

        manager.close()

# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for L1MemoryManager.

These tests verify the behavior of L1MemoryManager as described in the
interface docstrings. The tests focus on:

1. allocate() - Thread-safe allocation returning (error_code, memory_objs)
   - Returns SUCCESS and non-empty list on successful allocation
   - Returns OUT_OF_MEMORY and empty list when allocation fails

2. free() - Thread-safe deallocation returning error_code
   - Returns SUCCESS when operation succeeds

3. get_vm_space() - Returns underlying virtual memory as torch.Tensor
"""

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.distributed.config import L1MemoryManagerConfig
from lmcache.v1.multiprocess.distributed.error import L1MemoryManagerError
from lmcache.v1.multiprocess.distributed.memory_manager import (
    L1MemoryManager,
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
def basic_config():
    """Create a basic L1MemoryManagerConfig for testing."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,  # 64MB
        align_bytes=0x1000,  # 4KB
    )


@pytest.fixture
def small_config():
    """Create a small L1MemoryManagerConfig to test memory exhaustion.

    Note: Minimum size is 64MB due to LazyMemoryAllocator's PIN_CHUNK_SIZE.
    """
    return L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,  # 64MB
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=64
        * 1024
        * 1024,  # 64MB (same as final to prevent expansion)
        align_bytes=0x1000,
    )


@pytest.fixture
def non_lazy_config():
    """Create a config that does not use lazy allocation."""
    return L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,  # 128MB
        use_lazy=False,
        init_size_in_bytes=64 * 1024 * 1024,
        align_bytes=0x1000,
    )


@pytest.fixture
def basic_layout():
    """Create a basic MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def multi_tensor_layout():
    """Create a MemoryLayoutDesc with multiple tensor shapes."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512]), torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16, torch.bfloat16],
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


# =============================================================================
# Tests for L1MemoryManager.allocate()
# =============================================================================


class TestAllocate:
    """
    Tests for L1MemoryManager.allocate() method.

    Per the docstring:
    - This function should be thread-safe
    - Returns tuple[L1MemoryManagerError, list[MemoryObj]]
    - Error code is OUT_OF_MEMORY if allocation fails, otherwise SUCCESS
    - If allocation fails, the memory object list will be empty
    """

    def test_allocate_returns_success_and_memory_objs(self, basic_config, basic_layout):
        """Test that allocate returns SUCCESS and valid memory objects."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(basic_layout, count=1)

        assert error == L1MemoryManagerError.SUCCESS
        assert isinstance(mem_objs, list)
        assert len(mem_objs) == 1
        for obj in mem_objs:
            assert obj is not None
            assert obj.is_valid()

        manager.close()

    def test_allocate_returns_correct_count(self, basic_config, basic_layout):
        """Test that allocate returns the requested number of memory objects."""
        manager = L1MemoryManager(basic_config)
        count = 5

        error, mem_objs = manager.allocate(basic_layout, count=count)

        assert error == L1MemoryManagerError.SUCCESS
        assert len(mem_objs) == count

        manager.close()

    def test_allocate_with_multi_tensor_layout(self, basic_config, multi_tensor_layout):
        """Test allocation with multiple tensor shapes in the layout."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(multi_tensor_layout, count=2)

        assert error == L1MemoryManagerError.SUCCESS
        assert len(mem_objs) == 2

        manager.close()

    def test_allocate_returns_out_of_memory_when_exhausted(
        self, small_config, large_layout
    ):
        """
        Test that allocate returns OUT_OF_MEMORY and empty list when memory
        is exhausted.
        """
        manager = L1MemoryManager(small_config)

        # Request more memory than available (8MB * 10 = 80MB > 64MB)
        error, mem_objs = manager.allocate(large_layout, count=10)

        assert error == L1MemoryManagerError.OUT_OF_MEMORY
        assert isinstance(mem_objs, list)
        assert len(mem_objs) == 0

        manager.close()

    def test_allocate_returns_empty_list_on_failure(self, small_config, large_layout):
        """Test that the memory object list is empty when allocation fails."""
        manager = L1MemoryManager(small_config)

        # Request more memory than available (8MB * 10 = 80MB > 64MB)
        error, mem_objs = manager.allocate(large_layout, count=10)

        # Verify the docstring contract: "If the allocation fails, the memory
        # object list will be empty."
        assert error == L1MemoryManagerError.OUT_OF_MEMORY
        assert mem_objs == []

        manager.close()

    def test_allocate_is_thread_safe(self, basic_config, basic_layout):
        """Test that allocate is thread-safe with concurrent allocations."""
        manager = L1MemoryManager(basic_config)
        num_threads = 10
        allocations_per_thread = 5
        results = []
        errors = []

        def allocate_task():
            try:
                for _ in range(allocations_per_thread):
                    error, mem_objs = manager.allocate(basic_layout, count=1)
                    results.append((error, mem_objs))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_task) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have been raised
        assert len(errors) == 0, f"Thread-safety errors: {errors}"

        # All results should have valid error codes
        for error, mem_objs in results:
            assert error in (
                L1MemoryManagerError.SUCCESS,
                L1MemoryManagerError.OUT_OF_MEMORY,
            )
            if error == L1MemoryManagerError.SUCCESS:
                assert len(mem_objs) == 1
            else:
                assert len(mem_objs) == 0

        manager.close()

    def test_allocate_with_non_lazy_config(self, non_lazy_config, basic_layout):
        """Test allocation with non-lazy (MixedMemoryAllocator) configuration."""
        manager = L1MemoryManager(non_lazy_config)

        error, mem_objs = manager.allocate(basic_layout, count=2)

        assert error == L1MemoryManagerError.SUCCESS
        assert len(mem_objs) == 2

        manager.close()


# =============================================================================
# Tests for L1MemoryManager.free()
# =============================================================================


class TestFree:
    """
    Tests for L1MemoryManager.free() method.

    Per the docstring:
    - This function should be thread-safe
    - Returns L1MemoryManagerError indicating the result
    - Returns SUCCESS if operation succeeds
    """

    def test_free_returns_success(self, basic_config, basic_layout):
        """Test that free returns SUCCESS for valid memory objects."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(basic_layout, count=2)
        assert error == L1MemoryManagerError.SUCCESS

        free_error = manager.free(mem_objs)

        assert free_error == L1MemoryManagerError.SUCCESS

        manager.close()

    def test_free_invalidates_memory_objects(self, basic_config, basic_layout):
        """Test that free invalidates the memory objects."""
        manager = L1MemoryManager(basic_config)

        error, mem_objs = manager.allocate(basic_layout, count=1)
        assert error == L1MemoryManagerError.SUCCESS
        assert mem_objs[0].is_valid()

        free_error = manager.free(mem_objs)
        assert free_error == L1MemoryManagerError.SUCCESS

        # After freeing, the memory objects should be invalidated
        for obj in mem_objs:
            assert not obj.is_valid()

        manager.close()

    def test_free_empty_list_returns_success(self, basic_config):
        """Test that freeing an empty list returns SUCCESS."""
        manager = L1MemoryManager(basic_config)

        free_error = manager.free([])

        assert free_error == L1MemoryManagerError.SUCCESS

        manager.close()

    def test_free_allows_reallocation(self, basic_config, basic_layout):
        """Test that freed memory can be reallocated."""
        manager = L1MemoryManager(basic_config)

        # Allocate
        error1, mem_objs1 = manager.allocate(basic_layout, count=5)
        assert error1 == L1MemoryManagerError.SUCCESS

        # Free
        free_error = manager.free(mem_objs1)
        assert free_error == L1MemoryManagerError.SUCCESS

        # Reallocate
        error2, mem_objs2 = manager.allocate(basic_layout, count=5)
        assert error2 == L1MemoryManagerError.SUCCESS
        assert len(mem_objs2) == 5

        manager.close()

    def test_free_is_thread_safe(self, basic_config, basic_layout):
        """Test that free is thread-safe with concurrent operations."""
        manager = L1MemoryManager(basic_config)
        num_threads = 10
        errors_list = []
        lock = threading.Lock()

        def allocate_and_free_task():
            try:
                error, mem_objs = manager.allocate(basic_layout, count=1)
                if error == L1MemoryManagerError.SUCCESS:
                    free_error = manager.free(mem_objs)
                    with lock:
                        errors_list.append(free_error)
            except Exception as e:
                with lock:
                    errors_list.append(e)

        threads = [
            threading.Thread(target=allocate_and_free_task) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All free operations should have succeeded
        for err in errors_list:
            if isinstance(err, Exception):
                pytest.fail(f"Thread-safety error: {err}")
            assert err == L1MemoryManagerError.SUCCESS

        manager.close()


# =============================================================================
# Tests for L1MemoryManager.get_vm_space()
# =============================================================================


class TestGetVmSpace:
    """
    Tests for L1MemoryManager.get_vm_space() method.

    Per the docstring:
    - Returns the underlying virtual memory space as a torch.Tensor
    - Will be used for RDMA communication
    """

    def test_get_vm_space_returns_tensor(self, basic_config):
        """Test that get_vm_space returns a torch.Tensor."""
        manager = L1MemoryManager(basic_config)

        vm_space = manager.get_vm_space()

        assert isinstance(vm_space, torch.Tensor)

        manager.close()

    def test_get_vm_space_returns_consistent_tensor(self, basic_config):
        """Test that get_vm_space returns the same tensor on multiple calls."""
        manager = L1MemoryManager(basic_config)

        vm_space1 = manager.get_vm_space()
        vm_space2 = manager.get_vm_space()

        # Should return the same underlying tensor
        assert vm_space1.data_ptr() == vm_space2.data_ptr()

        manager.close()

    def test_get_vm_space_size_matches_config(self, basic_config):
        """Test that the vm_space size is consistent with the configuration."""
        manager = L1MemoryManager(basic_config)

        vm_space = manager.get_vm_space()

        # The vm_space should have at least the configured size
        # (may be larger due to alignment)
        assert vm_space.numel() >= basic_config.size_in_bytes

        manager.close()


# =============================================================================
# Tests for L1MemoryManager integration
# =============================================================================


class TestL1MemoryManagerIntegration:
    """Integration tests for L1MemoryManager."""

    def test_allocate_free_cycle(self, basic_config, basic_layout):
        """Test a complete allocate-free cycle."""
        manager = L1MemoryManager(basic_config)

        # Allocate
        error, mem_objs = manager.allocate(basic_layout, count=3)
        assert error == L1MemoryManagerError.SUCCESS
        assert len(mem_objs) == 3

        # Verify objects are valid
        for obj in mem_objs:
            assert obj.is_valid()

        # Free
        free_error = manager.free(mem_objs)
        assert free_error == L1MemoryManagerError.SUCCESS

        # Verify objects are invalidated
        for obj in mem_objs:
            assert not obj.is_valid()

        manager.close()

    def test_multiple_allocate_free_cycles(self, basic_config, basic_layout):
        """Test multiple allocate-free cycles."""
        manager = L1MemoryManager(basic_config)

        for i in range(5):
            error, mem_objs = manager.allocate(basic_layout, count=2)
            assert error == L1MemoryManagerError.SUCCESS, f"Cycle {i} failed"
            assert len(mem_objs) == 2

            free_error = manager.free(mem_objs)
            assert free_error == L1MemoryManagerError.SUCCESS

        manager.close()

    def test_interleaved_allocate_and_free(self, basic_config, basic_layout):
        """Test interleaved allocation and free operations."""
        manager = L1MemoryManager(basic_config)

        # Allocate batch 1
        err1, objs1 = manager.allocate(basic_layout, count=2)
        assert err1 == L1MemoryManagerError.SUCCESS

        # Allocate batch 2
        err2, objs2 = manager.allocate(basic_layout, count=2)
        assert err2 == L1MemoryManagerError.SUCCESS

        # Free batch 1
        free_err1 = manager.free(objs1)
        assert free_err1 == L1MemoryManagerError.SUCCESS

        # Allocate batch 3 (should reuse freed memory)
        err3, objs3 = manager.allocate(basic_layout, count=2)
        assert err3 == L1MemoryManagerError.SUCCESS

        # Free remaining
        manager.free(objs2)
        manager.free(objs3)

        manager.close()

    def test_concurrent_allocate_and_free(self, basic_config, basic_layout):
        """
        Test concurrent allocate and free operations from multiple threads.
        This verifies the thread-safety guarantees of both methods.
        """
        manager = L1MemoryManager(basic_config)
        num_threads = 8
        iterations = 20
        exceptions = []

        def worker():
            try:
                for _ in range(iterations):
                    # Allocate
                    error, mem_objs = manager.allocate(basic_layout, count=1)
                    if error == L1MemoryManagerError.SUCCESS:
                        # Briefly hold the memory
                        assert len(mem_objs) == 1
                        # Free
                        free_error = manager.free(mem_objs)
                        assert free_error == L1MemoryManagerError.SUCCESS
            except Exception as e:
                exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raises exception if worker failed

        assert len(exceptions) == 0, f"Concurrency errors: {exceptions}"

        manager.close()


# =============================================================================
# Tests for error code semantics
# =============================================================================


class TestErrorCodeSemantics:
    """Tests verifying error code semantics as documented."""

    def test_success_error_code_value(self):
        """Test that SUCCESS is a valid enum member."""
        assert L1MemoryManagerError.SUCCESS is not None
        assert L1MemoryManagerError.SUCCESS.name == "SUCCESS"

    def test_out_of_memory_error_code_value(self):
        """Test that OUT_OF_MEMORY is a valid enum member."""
        assert L1MemoryManagerError.OUT_OF_MEMORY is not None
        assert L1MemoryManagerError.OUT_OF_MEMORY.name == "OUT_OF_MEMORY"

    def test_error_codes_are_distinct(self):
        """Test that error codes are distinct values."""
        assert L1MemoryManagerError.SUCCESS != L1MemoryManagerError.OUT_OF_MEMORY

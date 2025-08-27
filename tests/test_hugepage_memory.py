# SPDX-License-Identifier: Apache-2.0
"""
Tests for integrated hugepage memory support in HostMemoryAllocator.
"""

# Standard
import ctypes

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import HostMemoryAllocator
import lmcache.c_ops as lmc_ops


class TestIntegratedHugepageSupport:
    """Test integrated hugepage support functionality in HostMemoryAllocator."""

    def test_hugepage_availability_check(self):
        """Test if hugepage availability check works."""
        if lmc_ops.is_hugepage_available():
            size = lmc_ops.get_hugepage_size()
            count = lmc_ops.get_available_hugepage_count()

            assert size > 0
            assert count >= 0
            # Should be either 2MB or 1GB
            assert size in [2 * 1024 * 1024, 1024 * 1024 * 1024]
        else:
            pytest.skip("Hugepages not available on this system")

    def test_hugepage_size_query(self):
        """Test hugepage size query functions."""
        if lmc_ops.is_hugepage_available():
            size = lmc_ops.get_hugepage_size()
            assert size > 0
            # Should be either 2MB or 1GB
            assert size in [2 * 1024 * 1024, 1024 * 1024 * 1024]

    def test_hugepage_count_query(self):
        """Test available hugepage count query."""
        if lmc_ops.is_hugepage_available():
            count = lmc_ops.get_available_hugepage_count()
            assert count >= 0

    def test_basic_hugepage_allocation(self):
        """Test basic hugepage memory allocation using C functions."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")

        # Test with small size that fits in one hugepage
        test_size = 1024 * 1024  # 1MB

        try:
            ptr = lmc_ops.alloc_pinned_hugepage_ptr(test_size)
            assert ptr != 0

            # Test memory access
            array_type = ctypes.c_uint8 * test_size
            buf = array_type.from_address(ptr)
            buf[0] = 42
            assert buf[0] == 42

            # Clean up
            lmc_ops.free_pinned_hugepage_ptr(ptr, test_size)
        except Exception as e:
            pytest.fail(f"Hugepage allocation failed: {e}")

    def test_host_memory_allocator_with_hugepage(self):
        """Test HostMemoryAllocator with hugepage support."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")

        test_size = 1024 * 1024  # 1MB

        try:
            # Test regular memory allocation
            regular_allocator = HostMemoryAllocator(test_size, use_hugepage=False)
            regular_memory = regular_allocator.allocate((100, 100), torch.float32)
            assert regular_memory is not None
            regular_allocator.close()

            # Test hugepage memory allocation
            hugepage_allocator = HostMemoryAllocator(test_size, use_hugepage=True)
            hugepage_memory = hugepage_allocator.allocate((100, 100), torch.float32)
            assert hugepage_memory is not None
            hugepage_allocator.close()

        except Exception as e:
            pytest.fail(f"HostMemoryAllocator hugepage test failed: {e}")

    def test_host_memory_allocator_with_hugepage_and_paging(self):
        """Test HostMemoryAllocator with both hugepage and paging enabled."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")

        test_size = 40000  # Size that fits in one hugepage
        test_shape = torch.Size((100, 100))
        test_dtype = torch.float32

        try:
            # Test with paging enabled
            allocator = HostMemoryAllocator(
                test_size,
                use_paging=True,
                use_hugepage=True,
                shape=test_shape,
                dtype=test_dtype,
                fmt="KV_2LTD",
            )

            memory_obj = allocator.allocate(test_shape, test_dtype)
            assert memory_obj is not None

            # Test memory check
            memcheck_result = allocator.memcheck()
            assert memcheck_result is not None

            allocator.close()

        except Exception as e:
            pytest.fail(f"HostMemoryAllocator hugepage+paging test failed: {e}")

    def test_memory_cleanup(self):
        """Test that hugepage memory is properly cleaned up."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")

        test_size = 1024 * 1024  # 1MB

        try:
            # Get initial hugepage count
            initial_count = lmc_ops.get_available_hugepage_count()

            # Create and destroy allocator
            allocator = HostMemoryAllocator(test_size, use_hugepage=True)
            memory_obj = allocator.allocate((100, 100), torch.float32)
            assert memory_obj is not None

            # Clean up
            allocator.close()

            # Check that hugepage count is restored
            final_count = lmc_ops.get_available_hugepage_count()
            assert final_count >= initial_count

        except Exception as e:
            pytest.fail(f"Memory cleanup test failed: {e}")

    def test_performance_comparison(self):
        """Test performance comparison between regular and hugepage memory."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")

        test_size = 64 * 1024 * 1024  # 64MB
        test_shape = (test_size // 4,)
        test_dtype = torch.float32

        try:
            # Test regular memory allocation time
            # Standard
            import time

            start_time = time.time()
            regular_allocator = HostMemoryAllocator(test_size, use_hugepage=False)
            regular_memory = regular_allocator.allocate(test_shape, test_dtype)
            regular_time = time.time() - start_time

            regular_allocator.close()

            # Test hugepage memory allocation time
            start_time = time.time()
            hugepage_allocator = HostMemoryAllocator(test_size, use_hugepage=True)
            hugepage_memory = hugepage_allocator.allocate(test_shape, test_dtype)
            hugepage_time = time.time() - start_time

            hugepage_allocator.close()

            # Both should complete successfully
            assert regular_memory is not None
            assert hugepage_memory is not None

            # Performance comparison (hugepage should be similar or better)
            # Note: Small differences are expected due to system load
            assert abs(hugepage_time - regular_time) < 1.0  # Within 1 second

        except Exception as e:
            pytest.fail(f"Performance comparison test failed: {e}")

    def test_large_memory_allocation(self):
        """Test large memory allocation with hugepages."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")

        # Test with size that requires multiple hugepages
        test_size = 256 * 1024 * 1024  # 256MB

        try:
            allocator = HostMemoryAllocator(test_size, use_hugepage=True)
            memory_obj = allocator.allocate((test_size // 4,), torch.float32)
            assert memory_obj is not None

            # Test memory access
            assert memory_obj.meta.phy_size >= test_size

            allocator.close()

        except Exception as e:
            pytest.fail(f"Large memory allocation test failed: {e}")

    def test_error_handling(self):
        """Test error handling for invalid parameters."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")

        # Test with invalid size
        try:
            HostMemoryAllocator(0, use_hugepage=True)
            pytest.fail("Should have failed with size 0")
        except Exception:
            # Expected to fail
            pass

        # Test with very large size (should fail gracefully)
        try:
            # Try to allocate more than available hugepages
            huge_size = 1024 * 1024 * 1024 * 1024  # 1TB
            HostMemoryAllocator(huge_size, use_hugepage=True)
            pytest.fail("Should have failed with extremely large size")
        except Exception:
            # Expected to fail
            pass

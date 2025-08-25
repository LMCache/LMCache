"""
Tests for hugepage memory support functionality.
"""

import ctypes
import pytest
import torch
import lmcache.c_ops as lmc_ops
from lmcache.v1.hugepage_memory import (
    HugepageMemoryAllocator,
    NumaHugepageMemoryAllocator,
    get_hugepage_info,
    create_hugepage_allocator,
)


class TestHugepageSupport:
    """Test hugepage support functionality."""
    
    def test_hugepage_availability_check(self):
        """Test if hugepage availability check works."""
        info = get_hugepage_info()
        assert isinstance(info, dict)
        assert "available" in info
        
        if info["available"]:
            assert "hugepage_size" in info
            assert "available_count" in info
            assert info["hugepage_size"] > 0
            assert info["available_count"] >= 0
    
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
        """Test basic hugepage memory allocation."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")
        
        # Test with small size that fits in one hugepage
        test_size = 1024 * 1024  # 1MB
        
        try:
            ptr = lmc_ops.alloc_pinned_hugepage_ptr(test_size, 0)
            assert ptr != 0
            
            # Test memory access
            array_type = ctypes.c_uint8 * test_size
            buf = array_type.from_address(ptr)
            buf[0] = 42
            assert buf[0] == 42
            
            # Clean up
            lmc_ops.free_pinned_hugepage_ptr(ptr)
        except Exception as e:
            pytest.fail(f"Hugepage allocation failed: {e}")
    
    def test_hugepage_memory_allocator(self):
        """Test HugepageMemoryAllocator class."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")
        
        test_size = 1024 * 1024  # 1MB
        
        try:
            allocator = HugepageMemoryAllocator(test_size)
            
            # Check hugepage info
            info = allocator.get_hugepage_info()
            assert info["hugepage_size"] > 0
            assert info["allocated_size"] == test_size
            
            # Test memory allocation
            memory_obj = allocator.allocate((100, 100), torch.float32)
            assert memory_obj is not None
            
            # Clean up
            allocator.close()
        except Exception as e:
            pytest.fail(f"HugepageMemoryAllocator test failed: {e}")
    
    def test_hugepage_memory_allocator_with_paging(self):
        """Test HugepageMemoryAllocator with paging enabled."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")
        
        test_size = 1024 * 1024  # 1MB
        
        try:
            allocator = HugepageMemoryAllocator(
                test_size,
                use_paging=True,
                shape=(100, 100),
                dtype=torch.float32,
                fmt="kv_2ltd"
            )
            
            # Test memory allocation
            memory_obj = allocator.allocate((100, 100), torch.float32)
            assert memory_obj is not None
            
            # Clean up
            allocator.close()
        except Exception as e:
            pytest.fail(f"HugepageMemoryAllocator with paging test failed: {e}")
    
    def test_factory_function(self):
        """Test the factory function for creating hugepage allocators."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")
        
        test_size = 1024 * 1024  # 1MB
        
        try:
            # Test without NUMA mapping
            allocator = create_hugepage_allocator(test_size)
            assert isinstance(allocator, HugepageMemoryAllocator)
            allocator.close()
            
            # Test with NUMA mapping (mock)
            class MockNumaMapping:
                def __init__(self):
                    self.gpu_to_numa_mapping = {0: 0}  # GPU 0 -> NUMA 0
            
            numa_mapping = MockNumaMapping()
            allocator = create_hugepage_allocator(test_size, numa_mapping)
            assert isinstance(allocator, NumaHugepageMemoryAllocator)
            allocator.close()
            
        except Exception as e:
            pytest.fail(f"Factory function test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling for hugepage operations."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")
        
        # Test allocation with size larger than available hugepages
        hugepage_size = lmc_ops.get_hugepage_size()
        available_count = lmc_ops.get_available_hugepage_count()
        
        if available_count > 0:
            oversized = (available_count + 1) * hugepage_size
            
            with pytest.raises(RuntimeError):
                HugepageMemoryAllocator(oversized)
    
    def test_memory_operations(self):
        """Test various memory operations with hugepage allocator."""
        if not lmc_ops.is_hugepage_available():
            pytest.skip("Hugepages not available on this system")
        
        test_size = 1024 * 1024  # 1MB
        
        try:
            allocator = HugepageMemoryAllocator(test_size)
            
            # Test batch allocation
            memory_objs = allocator.batched_allocate(
                (50, 50), torch.float32, batch_size=3
            )
            assert len(memory_objs) == 3
            assert all(obj is not None for obj in memory_objs)
            
            # Test batch free
            allocator.batched_free(memory_objs)
            
            # Test memory check
            result = allocator.memcheck()
            assert result is not None
            
            # Clean up
            allocator.close()
        except Exception as e:
            pytest.fail(f"Memory operations test failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    print("Testing hugepage support...")
    
    info = get_hugepage_info()
    print(f"Hugepage info: {info}")
    
    if info["available"]:
        print("Hugepages are available!")
        print(f"Size: {info['hugepage_size'] / (1024*1024):.1f} MB")
        print(f"Available count: {info['available_count']}")
    else:
        print("Hugepages are not available on this system.")
    
    print("Tests completed.") 
#!/usr/bin/env python3
"""
Example script demonstrating how to use hugepage memory support in LMCache.

This script shows how to:
1. Check hugepage availability
2. Create hugepage memory allocators
3. Use them for memory allocation
4. Monitor hugepage usage
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import lmcache.c_ops as lmc_ops
from lmcache.v1.hugepage_memory import (
    HugepageMemoryAllocator,
    NumaHugepageMemoryAllocator,
    get_hugepage_info,
    create_hugepage_allocator,
)


def check_hugepage_system():
    """Check and display hugepage system information."""
    print("=== Hugepage System Information ===")
    
    info = get_hugepage_info()
    if not info["available"]:
        print("❌ Hugepages are not available on this system")
        print("   To enable hugepages, you may need to:")
        print("   1. Configure hugepages in /etc/default/grub")
        print("   2. Reboot the system")
        print("   3. Mount hugepages filesystem")
        return False
    
    print("✅ Hugepages are available!")
    print(f"   Hugepage size: {info['hugepage_size'] / (1024*1024):.1f} MB")
    print(f"   Available count: {info['available_count']}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"   CUDA devices: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
    else:
        print("   CUDA is not available")
    
    return True


def demonstrate_basic_allocation():
    """Demonstrate basic hugepage memory allocation."""
    print("\n=== Basic Hugepage Allocation ===")
    
    if not lmc_ops.is_hugepage_available():
        print("❌ Skipping - hugepages not available")
        return
    
    try:
        # Allocate 1MB using hugepages
        size = 1024 * 1024
        print(f"Allocating {size / (1024*1024):.1f} MB using hugepages...")
        
        allocator = HugepageMemoryAllocator(size)
        
        # Get allocation info
        info = allocator.get_hugepage_info()
        print(f"   Allocated size: {info['allocated_size'] / (1024*1024):.1f} MB")
        print(f"   Required hugepages: {info['required_hugepages']}")
        
        # Allocate some tensors
        print("   Allocating tensors...")
        tensor1 = allocator.allocate((100, 100), torch.float32)
        tensor2 = allocator.allocate((200, 200), torch.float32)
        
        print(f"   Successfully allocated {tensor1.meta.shape} and {tensor2.meta.shape}")
        
        # Clean up
        allocator.close()
        print("   ✅ Basic allocation test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Basic allocation test failed: {e}")


def demonstrate_numa_allocation():
    """Demonstrate NUMA-aware hugepage allocation."""
    print("\n=== NUMA-Aware Hugepage Allocation ===")
    
    if not lmc_ops.is_hugepage_available():
        print("❌ Skipping - hugepages not available")
        return
    
    if not torch.cuda.is_available():
        print("❌ Skipping - CUDA not available")
        return
    
    try:
        # Mock NUMA mapping for demonstration
        class MockNumaMapping:
            def __init__(self):
                # Map GPU 0 to NUMA node 0
                self.gpu_to_numa_mapping = {0: 0}
        
        numa_mapping = MockNumaMapping()
        
        # Allocate 2MB using NUMA-aware hugepages
        size = 2 * 1024 * 1024
        print(f"Allocating {size / (1024*1024):.1f} MB using NUMA-aware hugepages...")
        
        allocator = NumaHugepageMemoryAllocator(size, numa_mapping)
        
        # Get allocation info
        info = allocator.get_hugepage_info()
        print(f"   Allocated size: {info['allocated_size'] / (1024*1024):.1f} MB")
        print(f"   NUMA node: {info['numa_id']}")
        print(f"   Required hugepages: {info['required_hugepages']}")
        
        # Allocate tensors
        print("   Allocating tensors...")
        tensor = allocator.allocate((500, 500), torch.float32)
        print(f"   Successfully allocated {tensor.meta.shape}")
        
        # Clean up
        allocator.close()
        print("   ✅ NUMA allocation test completed successfully")
        
    except Exception as e:
        print(f"   ❌ NUMA allocation test failed: {e}")


def demonstrate_factory_function():
    """Demonstrate the factory function for creating allocators."""
    print("\n=== Factory Function Usage ===")
    
    if not lmc_ops.is_hugepage_available():
        print("❌ Skipping - hugepages not available")
        return
    
    try:
        size = 1024 * 1024  # 1MB
        
        # Create allocator without NUMA mapping
        print("Creating basic hugepage allocator...")
        allocator1 = create_hugepage_allocator(size)
        print(f"   Type: {type(allocator1).__name__}")
        allocator1.close()
        
        # Create allocator with NUMA mapping
        print("Creating NUMA-aware hugepage allocator...")
        class MockNumaMapping:
            def __init__(self):
                self.gpu_to_numa_mapping = {0: 0}
        
        numa_mapping = MockNumaMapping()
        allocator2 = create_hugepage_allocator(size, numa_mapping)
        print(f"   Type: {type(allocator2).__name__}")
        allocator2.close()
        
        print("   ✅ Factory function test completed successfully")
        
    except Exception as e:
        print(f"   ❌ Factory function test failed: {e}")


def monitor_hugepage_usage():
    """Monitor and display hugepage usage statistics."""
    print("\n=== Hugepage Usage Monitoring ===")
    
    if not lmc_ops.is_hugepage_available():
        print("❌ Skipping - hugepages not available")
        return
    
    try:
        # Get current stats
        hugepage_size = lmc_ops.get_hugepage_size()
        available_count = lmc_ops.get_available_hugepage_count()
        
        print(f"Current hugepage status:")
        print(f"   Hugepage size: {hugepage_size / (1024*1024):.1f} MB")
        print(f"   Available count: {available_count}")
        print(f"   Total available memory: {available_count * hugepage_size / (1024*1024*1024):.2f} GB")
        
        # Try to read from /proc/meminfo for additional info
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'HugePages' in line:
                        print(f"   {line.strip()}")
        except:
            pass
        
        print("   ✅ Usage monitoring completed")
        
    except Exception as e:
        print(f"   ❌ Usage monitoring failed: {e}")


def main():
    """Main function to run all demonstrations."""
    print("LMCache Hugepage Support Demonstration")
    print("=" * 50)
    
    # Check system capabilities
    if not check_hugepage_system():
        print("\n❌ Cannot proceed without hugepage support")
        return
    
    # Run demonstrations
    demonstrate_basic_allocation()
    demonstrate_numa_allocation()
    demonstrate_factory_function()
    monitor_hugepage_usage()
    
    print("\n" + "=" * 50)
    print("✅ All demonstrations completed!")
    print("\nTo use hugepage support in your code:")
    print("1. Import: from lmcache.v1.hugepage_memory import create_hugepage_allocator")
    print("2. Create: allocator = create_hugepage_allocator(size)")
    print("3. Use: memory_obj = allocator.allocate(shape, dtype)")
    print("4. Cleanup: allocator.close()")


if __name__ == "__main__":
    main() 
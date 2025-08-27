#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Example script demonstrating how to use hugepage memory support in LMCache.

This script shows how to:
1. Check hugepage availability
2. Use integrated HostMemoryAllocator with hugepage support
3. Monitor hugepage usage
4. Compare performance between regular and hugepage memory
"""

# Standard
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import HostMemoryAllocator
import lmcache.c_ops as lmc_ops


def check_hugepage_system():
    """Check and display hugepage system information."""
    print("=== Hugepage System Information ===")

    if not lmc_ops.is_hugepage_available():
        print("❌ Hugepages are not available on this system")
        print("   To enable hugepages, you may need to:")
        print("   1. Configure hugepages in /etc/default/grub")
        print("   2. Reboot the system")
        print("   3. Mount hugepages filesystem")
        return False

    print("✅ Hugepages are available!")
    print(f"   Hugepage size: {lmc_ops.get_hugepage_size() / (1024 * 1024):.1f} MB")
    print(f"   Available count: {lmc_ops.get_available_hugepage_count()}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"   CUDA devices: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
    else:
        print("   CUDA is not available")

    return True


def demonstrate_integrated_allocation():
    """Demonstrate integrated HostMemoryAllocator with hugepage support."""
    print("\n=== Integrated HostMemoryAllocator with Hugepage Support ===")

    if not lmc_ops.is_hugepage_available():
        print("❌ Skipping - hugepages not available")
        return

    try:
        # Test different memory sizes
        test_sizes = [
            1024 * 1024,
            64 * 1024 * 1024,
            256 * 1024 * 1024,
        ]  # 1MB, 64MB, 256MB

        for size in test_sizes:
            size_mb = size / (1024 * 1024)
            print(f"\n--- Testing {size_mb:.0f}MB allocation ---")

            # Test regular memory
            start_time = time.time()
            regular_allocator = HostMemoryAllocator(size, use_hugepage=False)
            regular_allocator.allocate((size // 4,), torch.float32)
            regular_time = time.time() - start_time

            print(f"   Regular memory: {regular_time * 1000:.2f}ms")

            # Test hugepage memory
            start_time = time.time()
            hugepage_allocator = HostMemoryAllocator(size, use_hugepage=True)
            hugepage_allocator.allocate((size // 4,), torch.float32)
            hugepage_time = time.time() - start_time

            print(f"   Hugepage memory: {hugepage_time * 1000:.2f}ms")

            # Calculate speedup
            if hugepage_time > 0:
                speedup = regular_time / hugepage_time
                print(f"   Speedup: {speedup:.2f}x")

            # Clean up
            regular_allocator.close()
            hugepage_allocator.close()

        print("\n   ✅ Integrated allocation test completed successfully")

    except Exception as e:
        print(f"   ❌ Integrated allocation test failed: {e}")


def demonstrate_gpu_transfer():
    """Demonstrate GPU transfer performance with hugepage memory."""
    print("\n=== GPU Transfer Performance Test ===")

    if not torch.cuda.is_available():
        print("❌ Skipping - CUDA not available")
        return

    if not lmc_ops.is_hugepage_available():
        print("❌ Skipping - hugepages not available")
        return

    try:
        # Test with 64MB memory
        size = 64 * 1024 * 1024  # 64MB
        print(f"Testing GPU transfer with {size / (1024 * 1024):.0f}MB memory...")

        # Test regular memory transfer
        start_time = time.time()
        regular_allocator = HostMemoryAllocator(size, use_hugepage=False)
        regular_memory = regular_allocator.allocate((size // 4,), torch.float32)

        # Copy to GPU
        gpu_tensor = torch.tensor(
            regular_memory.raw_data, dtype=torch.float32, device="cuda"
        )
        torch.cuda.synchronize()
        regular_to_gpu = time.time() - start_time

        # Copy back from GPU
        start_time = time.time()
        gpu_tensor.cpu()
        torch.cuda.synchronize()
        regular_from_gpu = time.time() - start_time

        regular_total = regular_to_gpu + regular_from_gpu

        print("   Regular memory transfer:")
        print(f"     -> GPU: {regular_to_gpu * 1000:.2f}ms")
        print(f"     <- GPU: {regular_from_gpu * 1000:.2f}ms")
        print(f"     Total: {regular_total * 1000:.2f}ms")

        # Test hugepage memory transfer
        start_time = time.time()
        hugepage_allocator = HostMemoryAllocator(size, use_hugepage=True)
        hugepage_memory = hugepage_allocator.allocate((size // 4,), torch.float32)

        # Copy to GPU
        gpu_tensor = torch.tensor(
            hugepage_memory.raw_data, dtype=torch.float32, device="cuda"
        )
        torch.cuda.synchronize()
        hugepage_to_gpu = time.time() - start_time

        # Copy back from GPU
        start_time = time.time()
        gpu_tensor.cpu()
        torch.cuda.synchronize()
        hugepage_from_gpu = time.time() - start_time

        hugepage_total = hugepage_to_gpu + hugepage_from_gpu

        print("   Hugepage memory transfer:")
        print(f"     -> GPU: {hugepage_to_gpu * 1000:.2f}ms")
        print(f"     <- GPU: {hugepage_from_gpu * 1000:.2f}ms")
        print(f"     Total: {hugepage_total * 1000:.2f}ms")

        # Calculate performance improvement
        if hugepage_total > 0:
            speedup = regular_total / hugepage_total
            print(f"   Performance improvement: {speedup:.2f}x")

        # Clean up
        regular_allocator.close()
        hugepage_allocator.close()

        print("   ✅ GPU transfer test completed successfully")

    except Exception as e:
        print(f"   ❌ GPU transfer test failed: {e}")


def main():
    """Main function to demonstrate hugepage usage."""
    print("🚀 LMCache Hugepage Memory Support Demo")
    print("=" * 60)

    # Check system capabilities
    if not check_hugepage_system():
        return

    # Demonstrate integrated allocation
    demonstrate_integrated_allocation()

    # Demonstrate GPU transfer performance
    demonstrate_gpu_transfer()

    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\n💡 Key Benefits of Integrated Hugepage Support:")
    print("  • Unified interface with HostMemoryAllocator")
    print("  • Easy switching between regular and hugepage memory")
    print("  • Automatic memory cleanup with close() method")
    print("  • Performance improvements for large memory operations")


if __name__ == "__main__":
    main()

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);
uintptr_t alloc_numa_ptr(size_t size, int node);
uintptr_t alloc_pinned_numa_ptr(size_t size, int node);
uintptr_t alloc_shm_pinned_ptr(size_t size, const std::string& shm_name);
void batched_memcpy(const std::vector<uintptr_t>& src_ptrs,
                    const std::vector<uintptr_t>& dst_ptrs,
                    const std::vector<size_t>& sizes);

void free_pinned_ptr(uintptr_t ptr);
void free_numa_ptr(uintptr_t ptr, size_t size);
void free_pinned_numa_ptr(uintptr_t ptr, size_t size);
void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name);

// Hugepage variants (MAP_HUGETLB). Not available for shm: /dev/shm usually
// uses tmpfs, and tmpfs does not support MAP_HUGETLB.
uintptr_t alloc_hugepage_pinned_ptr(size_t size, unsigned int flags);
uintptr_t alloc_hugepage_pinned_numa_ptr(size_t size, int node);

void free_hugepage_pinned_ptr(uintptr_t ptr, size_t size);
void free_hugepage_pinned_numa_ptr(uintptr_t ptr, size_t size);

// PCIe BAR IO-memory allocator.
//
// Opens a sysfs BAR resource file (e.g.
// /sys/bus/pci/devices/0000:XX:XX.X/resourceN), maps it MAP_SHARED with
// mmap(2), and registers the region with CUDA using
// cudaHostRegisterMapped | cudaHostRegisterIoMemory so that
// cudaHostGetDevicePointer() returns a GPU VA that aliases the BAR physical
// address.  Kernel writes to that VA generate PCIe Write TLPs that land
// directly in the BAR (CXL/NVMe CMB / FPGA SRAM) without staging through
// host DRAM.
//
// Constraints the caller must satisfy:
//   - Linux only, IOMMU disabled (intel_iommu=off) or in passthrough mode.
//   - bar_offset must be a multiple of the system page size.
//   - size must not exceed the actual BAR window size.
//   - The CUDA context must be active on the target GPU before calling.
//
// Returns the CPU virtual address of the mapped region.
// A companion fd is stored internally; free_pcie_bar_ptr releases it.
uintptr_t alloc_pcie_bar_ptr(const std::string& bar_path, size_t size,
                              size_t bar_offset);

// Releases a region allocated by alloc_pcie_bar_ptr:
//   cudaHostUnregister → munmap → close(fd).
void free_pcie_bar_ptr(uintptr_t ptr, size_t size);

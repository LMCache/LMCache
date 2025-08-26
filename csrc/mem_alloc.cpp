#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>            // for strerror
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include <fstream>            // for reading /proc/meminfo
#include <sstream>            // for string parsing
#include "mem_alloc.h"

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, size, flags);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(err));
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  cudaError_t err = cudaFreeHost(reinterpret_cast<void*>(ptr));
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost failed: " + std::to_string(err));
  }
}

static void first_touch(void* p, size_t size) {
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = (volatile char*)p + off;
    *c = 0;
  }
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size);

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

// Hugepage support implementation

static int get_hugepage_size_from_sysfs() {
  {
    std::ifstream file(
        "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages");
    if (file.good()) {
      return 2 * 1024 * 1024;  // 2MB
    }
  }

  {
    std::ifstream file(
        "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages");
    if (file.good()) {
      return 1024 * 1024 * 1024;  // 1GB
    }
  }

  return 0;  // No hugepages available
}

int get_hugepage_size() {
  static int hugepage_size = get_hugepage_size_from_sysfs();
  return hugepage_size;
}

bool is_hugepage_available() { return get_hugepage_size() > 0; }

int get_available_hugepage_count() {
  int hugepage_size = get_hugepage_size();
  if (hugepage_size == 0) return 0;

  std::string filename;
  if (hugepage_size == 2 * 1024 * 1024) {  // 2MB
    filename = "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages";
  } else if (hugepage_size == 1024 * 1024 * 1024) {  // 1GB
    filename = "/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages";
  } else {
    return 0;
  }

  std::ifstream file(filename);
  if (!file.good()) return 0;

  int count;
  file >> count;
  if (file.fail()) {
    return 0;
  }
  return count;
}

uintptr_t alloc_pinned_hugepage_ptr(size_t size) {
  int hugepage_size = get_hugepage_size();
  if (hugepage_size == 0) {
    throw std::runtime_error("Hugepages are not available on this system");
  }

  // Round up size to hugepage boundary
  size_t aligned_size = (size + hugepage_size - 1) & ~(hugepage_size - 1);

  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap with hugepage failed: ") +
                             strerror(errno));
  }

  // Register with CUDA for pinned memory
  cudaError_t st = cudaHostRegister(ptr, aligned_size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, aligned_size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_hugepage_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);

  // Round up size to hugepage boundary
  int hugepage_size = get_hugepage_size();
  size_t aligned_size = (size + hugepage_size - 1) & ~(hugepage_size - 1);

  // Unpin first, then unmap
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, aligned_size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }

  if (munmap(p, aligned_size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

uintptr_t alloc_pinned_numa_hugepage_ptr(size_t size, int node) {
  int hugepage_size = get_hugepage_size();
  if (hugepage_size == 0) {
    throw std::runtime_error("Hugepages are not available on this system");
  }

  // Round up size to hugepage boundary
  size_t aligned_size = (size + hugepage_size - 1) & ~(hugepage_size - 1);

  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap with hugepage failed: ") +
                             strerror(errno));
  }

  // Bind to specific NUMA node
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, aligned_size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, aligned_size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  // First touch to ensure memory is allocated on the correct node
  first_touch(ptr, aligned_size);

  // Register with CUDA for pinned memory
  cudaError_t st = cudaHostRegister(ptr, aligned_size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, aligned_size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_hugepage_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);

  // Round up size to hugepage boundary
  int hugepage_size = get_hugepage_size();
  size_t aligned_size = (size + hugepage_size - 1) & ~(hugepage_size - 1);

  // Unpin first, then unmap
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, aligned_size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }

  if (munmap(p, aligned_size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}
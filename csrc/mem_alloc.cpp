#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>            // for strerror
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
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
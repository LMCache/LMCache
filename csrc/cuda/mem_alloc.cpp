// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cassert>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mman.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <cstring>            // for strerror
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include "mem_alloc.h"

static constexpr size_t HUGEPAGE_SIZE = 2UL * 1024 * 1024;  // MAP_HUGE_2MB

static inline size_t _align_hugepage(size_t size) {
  return (size + HUGEPAGE_SIZE - 1) & ~(HUGEPAGE_SIZE - 1);
}

// NVIDIA r580+ kernel drivers build one flat 16-byte-per-4KiB-page
// bookkeeping table per host registration via kvzalloc(), which Linux caps
// at INT_MAX bytes.  A single cudaHostRegister (or cudaHostAlloc) of
// >= 2^39 bytes (512 GiB) therefore fails with cudaErrorMemoryAllocation
// ("NVRM: failed to allocate page table").  The cap is per call, not
// cumulative, so large ranges are registered in chunks below it.
// https://github.com/NVIDIA/open-gpu-kernel-modules/pull/1234
static constexpr size_t PIN_WALL = 1ULL << 39;       // 512 GiB
static constexpr size_t PIN_CHUNK_MAX = 1ULL << 36;  // 64 GiB per call

// A failed CUDA runtime call latches a per-thread sticky error that the
// next error-checked CUDA call (typically an unrelated kernel launch)
// would report as its own.  Consume it before throwing.
static void _clear_cuda_error() { (void)cudaGetLastError(); }

// Register [ptr, ptr+size) in <= PIN_CHUNK_MAX chunks.  On failure, rolls
// back the chunks registered so far and throws.
static void _host_register_chunked(void* ptr, size_t size, unsigned int flags) {
  char* base = static_cast<char*>(ptr);
  for (size_t off = 0; off < size;) {
    size_t n = std::min(PIN_CHUNK_MAX, size - off);
    cudaError_t st = cudaHostRegister(base + off, n, flags);
    if (st != cudaSuccess) {
      _clear_cuda_error();
      for (size_t undo = off; undo > 0;) {
        undo -= PIN_CHUNK_MAX;
        cudaHostUnregister(base + undo);
      }
      _clear_cuda_error();
      throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                               cudaGetErrorString(st));
    }
    off += n;
  }
}

// Unregister every chunk of [ptr, ptr+size).  Returns the first error but
// keeps going so remaining chunks are still released.
static cudaError_t _host_unregister_chunked(void* ptr, size_t size) {
  cudaError_t first = cudaSuccess;
  char* base = static_cast<char*>(ptr);
  size_t remaining = size;
  while (remaining > 0) {
    size_t off = ((remaining - 1) / PIN_CHUNK_MAX) * PIN_CHUNK_MAX;
    cudaError_t st = cudaHostUnregister(base + off);
    if (st != cudaSuccess) {
      _clear_cuda_error();
      if (first == cudaSuccess) {
        first = st;
      }
    }
    remaining = off;
  }
  return first;
}

// mmap-backed allocations made by alloc_pinned_ptr for sizes >= PIN_WALL.
// free_pinned_ptr only receives the pointer, so remember which pointers
// must be unregistered+munmap'ed instead of cudaFreeHost'ed.
static std::mutex g_mmap_pinned_mu;
static std::unordered_map<uintptr_t, size_t> g_mmap_pinned;

static void* _mmap_anon(size_t size, bool hugepages) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (hugepages) {
    flags |= MAP_HUGETLB | MAP_HUGE_2MB;
  }
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }
  return ptr;
}

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  constexpr unsigned int registerable_flags =
      cudaHostAllocPortable | cudaHostAllocMapped;
  // WriteCombined has no cudaHostRegister equivalent. Preserve CUDA's
  // semantics (including validation of unknown flags) on that path.
  if (size < PIN_WALL || (flags & ~registerable_flags) != 0) {
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size, flags);
    if (err != cudaSuccess) {
      _clear_cuda_error();
      throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(err));
    }
    return reinterpret_cast<uintptr_t>(ptr);
  }

  // >= 512 GiB: one cudaHostAlloc would hit the driver's per-registration
  // page-table cap.  mmap the range and register it in chunks instead.
  unsigned int register_flags = 0;
  if (flags & cudaHostAllocPortable) register_flags |= cudaHostRegisterPortable;
  if (flags & cudaHostAllocMapped) register_flags |= cudaHostRegisterMapped;
  void* ptr = _mmap_anon(size, false);
  // Record ownership before pinning so a map allocation failure cannot leak
  // registered memory. The pointer is not published to the caller yet.
  try {
    std::lock_guard<std::mutex> lk(g_mmap_pinned_mu);
    g_mmap_pinned.emplace(reinterpret_cast<uintptr_t>(ptr), size);
  } catch (...) {
    munmap(ptr, size);
    throw;
  }
  try {
    _host_register_chunked(ptr, size, register_flags);
  } catch (...) {
    std::lock_guard<std::mutex> lk(g_mmap_pinned_mu);
    g_mmap_pinned.erase(reinterpret_cast<uintptr_t>(ptr));
    munmap(ptr, size);
    throw;
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  size_t mmap_size = 0;
  {
    std::lock_guard<std::mutex> lk(g_mmap_pinned_mu);
    auto it = g_mmap_pinned.find(ptr);
    if (it != g_mmap_pinned.end()) {
      mmap_size = it->second;
      g_mmap_pinned.erase(it);
    }
  }
  if (mmap_size > 0) {
    // mmap-backed (>= PIN_WALL) allocation: unregister chunks, then unmap.
    void* p = reinterpret_cast<void*>(ptr);
    cudaError_t st = _host_unregister_chunked(p, mmap_size);
    if (munmap(p, mmap_size) != 0) {
      throw std::runtime_error(std::string("munmap failed: ") +
                               strerror(errno));
    }
    if (st != cudaSuccess) {
      throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                               cudaGetErrorString(st));
    }
    return;
  }
  cudaError_t err = cudaFreeHost(reinterpret_cast<void*>(ptr));
  if (err != cudaSuccess) {
    _clear_cuda_error();
    throw std::runtime_error("cudaFreeHost failed: " + std::to_string(err));
  }
}

uintptr_t alloc_hugepage_pinned_ptr(size_t size, unsigned int flags) {
  size = _align_hugepage(size);
  void* ptr = _mmap_anon(size, true);

  try {
    _host_register_chunked(ptr, size, flags);
  } catch (...) {
    munmap(ptr, size);
    throw;
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_hugepage_pinned_ptr(uintptr_t ptr, size_t size) {
  size = _align_hugepage(size);
  void* p = reinterpret_cast<void*>(ptr);

  // Unpin first, then unmap.
  cudaError_t st = _host_unregister_chunked(p, size);
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
}

void batched_memcpy(const std::vector<uintptr_t>& src_ptrs,
                    const std::vector<uintptr_t>& dst_ptrs,
                    const std::vector<size_t>& sizes) {
  if (src_ptrs.size() != dst_ptrs.size() || src_ptrs.size() != sizes.size()) {
    throw std::invalid_argument(
        "batched_memcpy expects equally sized src_ptrs, dst_ptrs, and sizes");
  }

  for (size_t i = 0; i < src_ptrs.size(); ++i) {
    if (sizes[i] == 0) {
      continue;
    }
    std::memmove(reinterpret_cast<void*>(dst_ptrs[i]),
                 reinterpret_cast<const void*>(src_ptrs[i]), sizes[i]);
  }
}

static void first_touch(void* p, size_t size, bool hugepages) {
  const size_t ps =
      hugepages ? HUGEPAGE_SIZE : static_cast<size_t>(sysconf(_SC_PAGESIZE));
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = static_cast<volatile char*>(p) + off;
    *c = 0;
  }
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

static uintptr_t _alloc_numa_impl(size_t size, int node, bool hugepages) {
  if (hugepages) {
    assert(size % HUGEPAGE_SIZE == 0);
  }

  void* ptr = _mmap_anon(size, hugepages);

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size, hugepages);

  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t alloc_numa_ptr(size_t size, int node) {
  return _alloc_numa_impl(size, node, false);
}

void free_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

static uintptr_t _alloc_pinned_numa_impl(size_t size, int node,
                                         bool hugepages) {
  void* ptr = reinterpret_cast<void*>(_alloc_numa_impl(size, node, hugepages));

  try {
    _host_register_chunked(ptr, size, 0);
  } catch (...) {
    munmap(ptr, size);
    throw;
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  return _alloc_pinned_numa_impl(size, node, false);
}

uintptr_t alloc_hugepage_pinned_numa_ptr(size_t size, int node) {
  size = _align_hugepage(size);
  return _alloc_pinned_numa_impl(size, node, true);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  cudaError_t st = _host_unregister_chunked(p, size);
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
}

void free_hugepage_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  size = _align_hugepage(size);
  free_pinned_numa_ptr(ptr, size);
}

uintptr_t alloc_shm_pinned_ptr(size_t size, const std::string& shm_name) {
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0)
    throw std::runtime_error(std::string("shm_open failed: ") +
                             strerror(errno));

  if (ftruncate(fd, size) != 0) {
    int err = errno;
    close(fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("ftruncate failed: ") + strerror(err));
  }

  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }

  first_touch(ptr, size, false);

  try {
    _host_register_chunked(ptr, size, 0);
  } catch (...) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw;
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name) {
  void* p = reinterpret_cast<void*>(ptr);
  cudaError_t st = _host_unregister_chunked(p, size);
  if (munmap(p, size) != 0) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
  shm_unlink(shm_name.c_str());
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
}

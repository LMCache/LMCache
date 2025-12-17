#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>  // for strerror
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include <pybind11/pybind11.h>
#include "mem_alloc.h"

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  cudaError_t err;
  {
    pybind11::gil_scoped_release release;
    err = cudaHostAlloc(&ptr, size, flags);
  }
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(err));
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  cudaError_t err;
  {
    pybind11::gil_scoped_release release;
    err = cudaFreeHost(reinterpret_cast<void*>(ptr));
  }
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost failed: " + std::to_string(err));
  }
}

static void first_touch_range(void* p, size_t start, size_t end) {
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = start; off < end; off += ps) {
    volatile char* c = (volatile char*)p + off;
    *c = 0;
  }
}

static void first_touch(void* p, size_t size) {
  const long ps = sysconf(_SC_PAGESIZE);
  const size_t num_pages = (size + ps - 1) / ps;

  auto start_time = std::chrono::steady_clock::now();

  // Use multiple threads for large allocations (>1GB)
  const size_t threshold = 1UL << 30;  // 1GB
  if (size < threshold) {
    // Small allocation: single-threaded
    std::cout << "[first_touch] Size: " << std::fixed << std::setprecision(2)
              << (size / (1024.0 * 1024.0 * 1024.0))
              << " GB, Mode: single-threaded" << std::endl;
    first_touch_range(p, 0, size);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    double throughput =
        (size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000.0);
    std::cout << "[first_touch] Completed in " << duration << " ms, "
              << "Throughput: " << std::fixed << std::setprecision(2)
              << throughput << " GB/s" << std::endl;
    return;
  }

  // Large allocation: multi-threaded
  // Use at most 8 threads to avoid overwhelming the system
  const unsigned int hw_threads = std::thread::hardware_concurrency();
  const unsigned int num_threads = std::min(8u, std::max(1u, hw_threads / 2));

  std::cout << "[first_touch] Size: " << std::fixed << std::setprecision(2)
            << (size / (1024.0 * 1024.0 * 1024.0)) << " GB, "
            << "Mode: multi-threaded, Threads: " << num_threads
            << " (hw_threads: " << hw_threads << ")" << std::endl;

  const size_t chunk_size = (size + num_threads - 1) / num_threads;
  const size_t aligned_chunk = ((chunk_size + ps - 1) / ps) * ps;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (unsigned int i = 0; i < num_threads; ++i) {
    size_t start = i * aligned_chunk;
    size_t end = std::min(start + aligned_chunk, size);
    if (start >= size) break;

    threads.emplace_back(first_touch_range, p, start, end);
  }

  for (auto& t : threads) {
    t.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_time - start_time)
                      .count();
  double throughput = (size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000.0);
  std::cout << "[first_touch] Completed in " << duration << " ms, "
            << "Throughput: " << std::fixed << std::setprecision(2)
            << throughput << " GB/s" << std::endl;
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  auto total_start = std::chrono::steady_clock::now();

  std::cout << "[alloc_pinned_numa_ptr] Starting allocation: " << std::fixed
            << std::setprecision(2) << (size / (1024.0 * 1024.0 * 1024.0))
            << " GB on NUMA node " << node << std::endl;

  auto step_start = std::chrono::steady_clock::now();
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  auto step_end = std::chrono::steady_clock::now();
  auto mmap_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                           step_end - step_start)
                           .count();
  std::cout << "[alloc_pinned_numa_ptr] mmap completed in " << mmap_duration
            << " ms" << std::endl;

  // Maximum of 64 numa nodes
  step_start = std::chrono::steady_clock::now();
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  step_end = std::chrono::steady_clock::now();
  auto mbind_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            step_end - step_start)
                            .count();
  std::cout << "[alloc_pinned_numa_ptr] mbind completed in " << mbind_duration
            << " ms" << std::endl;

  // Release GIL during time-consuming operations to avoid blocking Python
  // interpreter
  std::cout << "[alloc_pinned_numa_ptr] Releasing GIL for first_touch and "
               "cudaHostRegister"
            << std::endl;
  {
    pybind11::gil_scoped_release release;

    step_start = std::chrono::steady_clock::now();
    first_touch(ptr, size);
    step_end = std::chrono::steady_clock::now();
    auto first_touch_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(step_end -
                                                              step_start)
            .count();
    std::cout << "[alloc_pinned_numa_ptr] first_touch total: "
              << first_touch_duration << " ms" << std::endl;

    step_start = std::chrono::steady_clock::now();
    cudaError_t st = cudaHostRegister(ptr, size, 0);
    if (st != cudaSuccess) {
      munmap(ptr, size);
      throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                               cudaGetErrorString(st));
    }
    step_end = std::chrono::steady_clock::now();
    auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                             step_end - step_start)
                             .count();
    std::cout << "[alloc_pinned_numa_ptr] cudaHostRegister completed in "
              << cuda_duration << " ms" << std::endl;
  }
  std::cout << "[alloc_pinned_numa_ptr] GIL reacquired" << std::endl;

  auto total_end = std::chrono::steady_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            total_end - total_start)
                            .count();
  std::cout << "[alloc_pinned_numa_ptr] Total allocation time: "
            << total_duration << " ms (" << std::fixed << std::setprecision(2)
            << (total_duration / 1000.0) << " s)" << std::endl;

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Release GIL during cleanup operations
  pybind11::gil_scoped_release release;
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
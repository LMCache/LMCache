#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>  // for strerror
#include <cstdio>   // for fprintf, stderr
#include <cstdlib>  // for getenv
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include <pybind11/pybind11.h>
#include "mem_alloc.h"

// Global log file handle and mutex for thread-safe logging
static FILE* g_log_file = nullptr;
static std::mutex g_log_mutex;

// Get log file handle (lazy initialization)
static FILE* get_log_file() {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  if (g_log_file == nullptr) {
    const char* log_path = std::getenv("LMCACHE_MEM_ALLOC_LOG");
    if (log_path == nullptr || log_path[0] == '\0') {
      // Default: /tmp/lmcache_mem_alloc_<pid>.log
      char default_path[256];
      snprintf(default_path, sizeof(default_path),
               "/tmp/lmcache_mem_alloc_%d.log", getpid());
      g_log_file = fopen(default_path, "a");
      if (g_log_file != nullptr) {
        fprintf(stderr, "[LMCache] Memory allocation logs: %s\n", default_path);
        fflush(stderr);
      }
    } else {
      g_log_file = fopen(log_path, "a");
      if (g_log_file != nullptr) {
        fprintf(stderr, "[LMCache] Memory allocation logs: %s\n", log_path);
        fflush(stderr);
      }
    }
    if (g_log_file == nullptr) {
      // Fallback to stderr
      g_log_file = stderr;
    }
  }
  return g_log_file;
}

// Thread-safe logging function
static void log_msg(const char* format, ...) {
  FILE* log_file = get_log_file();
  std::lock_guard<std::mutex> lock(g_log_mutex);

  // Add timestamp and PID
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) %
                1000;
  struct tm tm_buf;
  localtime_r(&now_time_t, &tm_buf);
  fprintf(log_file, "[%04d-%02d-%02d %02d:%02d:%02d.%03ld][PID:%d] ",
          tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
          tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, now_ms.count(),
          getpid());

  va_list args;
  va_start(args, format);
  vfprintf(log_file, format, args);
  va_end(args);

  fflush(log_file);
}

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  auto total_start = std::chrono::steady_clock::now();

  log_msg("[alloc_pinned_ptr] Starting allocation: %.2f GB, flags=%u\n",
          size / (1024.0 * 1024.0 * 1024.0), flags);

  void* ptr = nullptr;
  cudaError_t err;

  log_msg("[alloc_pinned_ptr] Releasing GIL for cudaHostAlloc\n");
  {
    pybind11::gil_scoped_release release;
    auto step_start = std::chrono::steady_clock::now();
    err = cudaHostAlloc(&ptr, size, flags);
    auto step_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        step_end - step_start)
                        .count();
    log_msg("[alloc_pinned_ptr] cudaHostAlloc completed in %ld ms\n", duration);
  }
  log_msg("[alloc_pinned_ptr] GIL reacquired\n");

  if (err != cudaSuccess) {
    log_msg("[alloc_pinned_ptr] ERROR: cudaHostAlloc failed: %s\n",
            cudaGetErrorString(err));
    throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(err));
  }

  auto total_end = std::chrono::steady_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            total_end - total_start)
                            .count();
  log_msg("[alloc_pinned_ptr] Total allocation time: %ld ms (%.2f s)\n",
          total_duration, total_duration / 1000.0);
  log_msg("[alloc_pinned_ptr] Allocation successful: ptr=0x%lx\n",
          reinterpret_cast<uintptr_t>(ptr));

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  log_msg("[free_pinned_ptr] Freeing ptr=0x%lx\n", ptr);

  cudaError_t err;
  log_msg("[free_pinned_ptr] Releasing GIL for cudaFreeHost\n");
  {
    pybind11::gil_scoped_release release;
    auto start = std::chrono::steady_clock::now();
    err = cudaFreeHost(reinterpret_cast<void*>(ptr));
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    log_msg("[free_pinned_ptr] cudaFreeHost completed in %ld ms\n", duration);
  }
  log_msg("[free_pinned_ptr] GIL reacquired\n");

  if (err != cudaSuccess) {
    log_msg("[free_pinned_ptr] ERROR: cudaFreeHost failed: %s\n",
            cudaGetErrorString(err));
    throw std::runtime_error("cudaFreeHost failed: " + std::to_string(err));
  }

  log_msg("[free_pinned_ptr] Free successful\n");
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
    log_msg("[first_touch] Size: %.2f GB, Mode: single-threaded\n",
            size / (1024.0 * 1024.0 * 1024.0));
    first_touch_range(p, 0, size);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    double throughput =
        (size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000.0);
    log_msg("[first_touch] Completed in %ld ms, Throughput: %.2f GB/s\n",
            duration, throughput);
    return;
  }

  // Large allocation: multi-threaded
  // Use at most 8 threads to avoid overwhelming the system
  const unsigned int hw_threads = std::thread::hardware_concurrency();
  const unsigned int num_threads = std::min(8u, std::max(1u, hw_threads / 2));

  log_msg(
      "[first_touch] Size: %.2f GB, Mode: multi-threaded, Threads: %u "
      "(hw_threads: %u)\n",
      size / (1024.0 * 1024.0 * 1024.0), num_threads, hw_threads);

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
  log_msg("[first_touch] Completed in %ld ms, Throughput: %.2f GB/s\n",
          duration, throughput);
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  auto total_start = std::chrono::steady_clock::now();

  log_msg(
      "[alloc_pinned_numa_ptr] Starting allocation: %.2f GB on NUMA node "
      "%d\n",
      size / (1024.0 * 1024.0 * 1024.0), node);

  auto step_start = std::chrono::steady_clock::now();
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  auto step_end = std::chrono::steady_clock::now();
  auto mmap_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                           step_end - step_start)
                           .count();
  log_msg("[alloc_pinned_numa_ptr] mmap completed in %ld ms\n", mmap_duration);

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
  log_msg("[alloc_pinned_numa_ptr] mbind completed in %ld ms\n",
          mbind_duration);

  // Release GIL during time-consuming operations to avoid blocking Python
  // interpreter
  log_msg(
      "[alloc_pinned_numa_ptr] Releasing GIL for first_touch and "
      "cudaHostRegister\n");
  {
    pybind11::gil_scoped_release release;

    step_start = std::chrono::steady_clock::now();
    first_touch(ptr, size);
    step_end = std::chrono::steady_clock::now();
    auto first_touch_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(step_end -
                                                              step_start)
            .count();
    log_msg("[alloc_pinned_numa_ptr] first_touch total: %ld ms\n",
            first_touch_duration);

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
    log_msg("[alloc_pinned_numa_ptr] cudaHostRegister completed in %ld ms\n",
            cuda_duration);
  }
  log_msg("[alloc_pinned_numa_ptr] GIL reacquired\n");

  auto total_end = std::chrono::steady_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            total_end - total_start)
                            .count();
  log_msg("[alloc_pinned_numa_ptr] Total allocation time: %ld ms (%.2f s)\n",
          total_duration, total_duration / 1000.0);

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
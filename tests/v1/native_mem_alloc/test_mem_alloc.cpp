// SPDX-License-Identifier: Apache-2.0
// Link the production allocator against a fake CUDA runtime. Large mmap calls
// reserve virtual addresses only; no CUDA driver or large physical RAM is used.
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mem_alloc.h"

constexpr size_t GiB = 1ULL << 30;
struct Registration {
  uintptr_t ptr;
  size_t size;
  unsigned int flags;
};
std::vector<Registration> registrations;
std::vector<uintptr_t> unregistrations;
std::map<void*, size_t> mappings;
std::map<void*, size_t> pinned;
int fail_register = -1;
int fail_unregister = -1;
bool fail_alloc = false;
cudaError_t pending_error = cudaSuccess;
size_t alloc_size = 0;
unsigned int alloc_flags = 0;
int free_calls = 0;

extern "C" void* __real_mmap(void*, size_t, int, int, int, off_t);
extern "C" int __real_munmap(void*, size_t);

extern "C" void* __wrap_mmap(void* addr, size_t size, int prot, int flags,
                             int fd, off_t offset) {
  // Exercise the hugepage API without requiring a configured hugepage pool.
  flags &= ~MAP_HUGETLB;
  void* ptr = __real_mmap(addr, size, prot, flags | MAP_NORESERVE, fd, offset);
  assert(ptr != MAP_FAILED);
  mappings.emplace(ptr, size);
  return ptr;
}

extern "C" int __wrap_munmap(void* ptr, size_t size) {
  assert(mappings.at(ptr) == size);
  mappings.erase(ptr);
  return __real_munmap(ptr, size);
}

extern "C" long __wrap_syscall(long number, ...) {
  // NUMA placement is outside this test's contract and may be denied in CI.
  assert(number == SYS_mbind);
  return 0;
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  alloc_size = size;
  alloc_flags = flags;
  if (fail_alloc) return pending_error = cudaErrorMemoryAllocation;
  *ptr = std::malloc(1);
  assert(*ptr != nullptr);
  return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
  ++free_calls;
  std::free(ptr);
  return cudaSuccess;
}

cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
  assert(pending_error == cudaSuccess);
  registrations.push_back({reinterpret_cast<uintptr_t>(ptr), size, flags});
  if (static_cast<int>(registrations.size()) - 1 == fail_register)
    return pending_error = cudaErrorMemoryAllocation;
  assert(size > 0 && size <= 64 * GiB);
  assert(reinterpret_cast<uintptr_t>(ptr) % 4096 == 0);
  assert(pinned.emplace(ptr, size).second);
  return cudaSuccess;
}

cudaError_t cudaHostUnregister(void* ptr) {
  unregistrations.push_back(reinterpret_cast<uintptr_t>(ptr));
  assert(pinned.erase(ptr) == 1);
  if (static_cast<int>(unregistrations.size()) - 1 == fail_unregister)
    return pending_error = cudaErrorMemoryAllocation;
  return cudaSuccess;
}

cudaError_t cudaGetLastError() {
  cudaError_t result = pending_error;
  pending_error = cudaSuccess;
  return result;
}

const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }

void reset() {
  assert(mappings.empty());
  assert(pinned.empty());
  assert(pending_error == cudaSuccess);
  registrations.clear();
  unregistrations.clear();
  fail_register = fail_unregister = -1;
  fail_alloc = false;
  alloc_size = 0;
  free_calls = 0;
}

template <typename F>
void expect_failure(F action, const std::string& message) {
  bool caught = false;
  try {
    action();
  } catch (const std::runtime_error& error) {
    caught = true;
    assert(std::string(error.what()).find(message) != std::string::npos);
  }
  assert(caught);
  assert(pending_error == cudaSuccess);
}

void check_reverse_cleanup(size_t count) {
  assert(unregistrations.size() == count);
  for (size_t i = 0; i < count; ++i)
    assert(unregistrations[i] == registrations[count - i - 1].ptr);
}

int main() {
  // Below the wall, including zero and all allocation flags, retain CUDA's API.
  for (size_t size : {size_t(0), size_t(4096), 512 * GiB - 1}) {
    for (unsigned int flags : {0u, 1u, 2u, 3u, 4u, 7u}) {
      reset();
      auto ptr = alloc_pinned_ptr(size, flags);
      assert(alloc_size == size && alloc_flags == flags);
      assert(registrations.empty());
      free_pinned_ptr(ptr);
      assert(free_calls == 1);
    }
  }
  // Exact boundary, full chunks, and a tail that is not page-sized.
  for (size_t size : {512 * GiB, 520 * GiB, 512 * GiB + 17}) {
    for (unsigned int flags : {0u, 1u, 2u, 3u}) {
      reset();
      auto ptr = alloc_pinned_ptr(size, flags);
      assert(alloc_size == 0);
      size_t covered = 0;
      for (const auto& registration : registrations) {
        assert(registration.ptr == ptr + covered);
        assert(registration.flags == flags);
        covered += registration.size;
      }
      assert(covered == size);
      free_pinned_ptr(ptr);
      assert(free_calls == 0);
      check_reverse_cleanup(registrations.size());
    }
  }
  // Do not silently discard WriteCombined or unknown flags at the boundary.
  for (unsigned int flags : {4u, 7u, 8u}) {
    reset();
    fail_alloc = true;
    expect_failure([&] { alloc_pinned_ptr(512 * GiB, flags); },
                   "cudaHostAlloc");
    assert(alloc_flags == flags && registrations.empty());
  }
  // Fail each of nine registrations, then verify rollback and retry.
  for (int failure = 0; failure < 9; ++failure) {
    reset();
    fail_register = failure;
    expect_failure([] { alloc_pinned_ptr(520 * GiB, 0); }, "cudaHostRegister");
    check_reverse_cleanup(failure);
    reset();
    auto ptr = alloc_pinned_ptr(520 * GiB, 0);
    free_pinned_ptr(ptr);
  }
  // An unregister error must not prevent cleanup of the other chunks.
  reset();
  auto ptr = alloc_pinned_ptr(520 * GiB, 0);
  fail_unregister = 0;
  expect_failure([&] { free_pinned_ptr(ptr); }, "cudaHostUnregister");
  check_reverse_cleanup(9);
  reset();
  fail_register = 3;
  fail_unregister = 0;
  expect_failure([] { alloc_pinned_ptr(520 * GiB, 0); }, "cudaHostRegister");
  check_reverse_cleanup(3);

  // Small NUMA, hugepage and shm pools use matching register/unregister paths.
  for (bool fail : {false, true}) {
    for (int kind = 0; kind < 4; ++kind) {
      reset();
      fail_register = fail ? 0 : -1;
      const std::string name = "/lmcache-pin-test-" + std::to_string(getpid());
      auto allocate = [&]() {
        switch (kind) {
          case 0:
            return alloc_pinned_numa_ptr(4096, 0);
          case 1:
            return alloc_hugepage_pinned_ptr(4096, 3);
          case 2:
            return alloc_hugepage_pinned_numa_ptr(4096, 0);
          default:
            return alloc_shm_pinned_ptr(4096, name);
        }
      };
      if (fail) {
        expect_failure(allocate, "cudaHostRegister");
      } else {
        auto small = allocate();
        assert(registrations.size() == 1);
        switch (kind) {
          case 0:
            free_pinned_numa_ptr(small, 4096);
            break;
          case 1:
            free_hugepage_pinned_ptr(small, 4096);
            break;
          case 2:
            free_hugepage_pinned_numa_ptr(small, 4096);
            break;
          default:
            free_shm_pinned_ptr(small, 4096, name);
            break;
        }
        check_reverse_cleanup(1);
      }
    }
  }
  reset();
  std::cout
      << "Host allocator boundary, flags, rollback and cleanup checks passed\n";
}

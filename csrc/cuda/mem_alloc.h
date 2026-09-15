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

// --- DMA-completion-safe deferred free ---------------------------------------
// The eviction path (LRU evictor -> PinnedAllocFree.free) historically called
// the free_*_ptr routines above SYNCHRONOUSLY. Those routines do
// cudaHostUnregister()+munmap() immediately, with NO ordering against an
// in-flight lmcache-driven D2H staging copy that is still reading the 64 KiB
// buffer on the transfer stream. Result: a use-after-free read of the freed
// staging region (a segfault in the evictor or the transfer worker).
//
// defer_free_pinned() records a lightweight CUDA event on the buffer's transfer
// stream and enqueues the buffer on a background reaper deque. A detached
// reaper thread polls cudaEventQuery (NON-blocking; NOT a global
// cudaDeviceSynchronize) and performs the real unregister/munmap/cudaFreeHost
// only AFTER the event has completed, i.e. after the DMA that last touched the
// buffer has drained.
//
// FreeKind mirrors the five free_*_ptr entry points so the reaper can dispatch
// to the correct teardown for each allocation shape.
enum class FreeKind : int {
  PINNED = 0,           // cudaFreeHost
  NUMA = 1,             // munmap
  PINNED_NUMA = 2,      // cudaHostUnregister + munmap
  HUGEPAGE_PINNED = 3,  // cudaHostUnregister + munmap (hugepage-aligned)
  HUGEPAGE_PINNED_NUMA = 4,
  SHM_PINNED = 5,  // cudaHostUnregister + munmap + shm_unlink
};

// Enqueue a DMA-completion-safe free. stream_ptr is the raw cudaStream_t
// (as int64) that the buffer's last D2H staging copy was enqueued on
// (torch.cuda.current_stream().cuda_stream on the LMCache worker). Returns
// immediately; the actual free happens on the reaper thread once the recorded
// event reports cudaSuccess.
void defer_free_pinned(int kind, uintptr_t ptr, size_t size,
                       const std::string& shm_name, int64_t stream_ptr);

// Diagnostics: number of buffers still awaiting DMA completion in the reaper
// deque. 0 once all deferred frees have drained.
size_t deferred_free_pending_count();


// Hugepage variants (MAP_HUGETLB). Not available for shm: /dev/shm usually
// uses tmpfs, and tmpfs does not support MAP_HUGETLB.
uintptr_t alloc_hugepage_pinned_ptr(size_t size, unsigned int flags);
uintptr_t alloc_hugepage_pinned_numa_ptr(size_t size, int node);

void free_hugepage_pinned_ptr(uintptr_t ptr, size_t size);
void free_hugepage_pinned_numa_ptr(uintptr_t ptr, size_t size);

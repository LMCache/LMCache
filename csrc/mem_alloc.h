#include <cstdint>

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);

void free_pinned_ptr(uintptr_t ptr);

uintptr_t alloc_pinned_numa_ptr(size_t size, int node);

void free_pinned_numa_ptr(uintptr_t ptr, size_t size);

// Reserve (but do NOT pin/register) a contiguous host memory range.
// This is useful for progressively pinning sub-ranges via cudaHostRegister.
uintptr_t mmap_host_ptr(size_t size);
uintptr_t mmap_host_numa_ptr(size_t size, int node);
void munmap_host_ptr(uintptr_t ptr, size_t size);

// Pin/unpin a host memory range for GPU access (progressive pinning).
void cuda_host_register(uintptr_t ptr, size_t size, unsigned int flags);
void cuda_host_unregister(uintptr_t ptr);
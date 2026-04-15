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

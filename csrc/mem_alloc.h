#include <cstdint>

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);

void free_pinned_ptr(uintptr_t ptr);

uintptr_t alloc_pinned_numa_ptr(size_t size, int node);

void free_pinned_numa_ptr(uintptr_t ptr, size_t size);

// CPU NUMA functions
int set_cpu_affinity(const int* cpu_list, int cpu_count);
int set_memory_policy(int policy, const int* nodes, int node_count);
int get_numa_node_count();
int get_cpu_count();

// Hugepage support functions
uintptr_t alloc_pinned_hugepage_ptr(size_t size);

void free_pinned_hugepage_ptr(uintptr_t ptr, size_t size);

uintptr_t alloc_pinned_numa_hugepage_ptr(size_t size, int node);

void free_pinned_numa_hugepage_ptr(uintptr_t ptr, size_t size);

// Hugepage configuration functions
int get_hugepage_size();
bool is_hugepage_available();
int get_available_hugepage_count();

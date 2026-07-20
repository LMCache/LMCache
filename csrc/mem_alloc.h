// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
//
// 2026/6/5 add c_memcpy function
//   Wenwen Chen <wenwen.chen@samsung.com>

#include <cstdint>
#include <string>
#include <vector>

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);
uintptr_t alloc_numa_ptr(size_t size, int node);
uintptr_t alloc_pinned_numa_ptr(size_t size, int node);
uintptr_t alloc_shm_pinned_ptr(size_t size, const std::string& shm_name);
bool c_memcpy(void* dst_ptr, const void* src_ptr, size_t length);
void batched_memcpy(const std::vector<uintptr_t>& src_ptrs,
                    const std::vector<uintptr_t>& dst_ptrs,
                    const std::vector<size_t>& sizes);

void free_pinned_ptr(uintptr_t ptr);
void free_numa_ptr(uintptr_t ptr, size_t size);
void free_pinned_numa_ptr(uintptr_t ptr, size_t size);
void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name);

// Hugepage variants (MAP_HUGETLB). Not available for shm: /dev/shm usually
// uses tmpfs, and tmpfs does not support MAP_HUGETLB.
uintptr_t alloc_hugepage_pinned_ptr(size_t size, unsigned int flags);
uintptr_t alloc_hugepage_pinned_numa_ptr(size_t size, int node);

void free_hugepage_pinned_ptr(uintptr_t ptr, size_t size);
void free_hugepage_pinned_numa_ptr(uintptr_t ptr, size_t size);

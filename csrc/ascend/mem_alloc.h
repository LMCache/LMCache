// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

/*
 * These following APIs are called directly in LMCache,
 * therefore we assume the ptrs management are done by the python program for
 * now.
 */

uintptr_t alloc_pinned_ptr(std::size_t size, unsigned int flags);

void free_pinned_ptr(uintptr_t ptr);

uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node);

void free_pinned_numa_ptr(uintptr_t ptr, std::size_t size);
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the lazy memory allocator."""

# First Party
from lmcache.v1.memory_allocators.lazy_memory_allocator import (
    LazyMemoryAllocator,
    align_to,
    get_numa_id,
)

__all__ = ["LazyMemoryAllocator", "align_to", "get_numa_id"]

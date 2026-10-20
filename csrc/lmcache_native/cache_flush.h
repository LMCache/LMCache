// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace lmcache {
namespace lmcache_native {

/**
 * @brief Whether this build can write back and invalidate CPU cache lines.
 *
 * True on x86-64, where CLFLUSHOPT is used when the CPU reports it and CLFLUSH
 * otherwise. False on other architectures; cache_flush_range() then throws.
 */
bool cache_flush_supported();

/**
 * @brief Write back and invalidate every CPU cache line overlapping a range.
 *
 * Used on memory that several hosts map without hardware coherence (CXL 2.0
 * multi-host Device-DAX). A writer calls it after its DMA completes so the
 * bytes reach the device; a reader calls it before its DMA so no stale line
 * from an earlier access survives on the reader's host.
 *
 * The flushes are followed by a store fence: Intel orders CLFLUSHOPT only
 * with SFENCE/MFENCE (LFENCE waits for local completion only), so the fence
 * makes the flushes globally performed before any later store, including the
 * doorbell write that starts the next GPU copy.
 *
 * @param ptr Start address of the range in this process.
 * @param size Length of the range in bytes; 0 is a no-op.
 *
 * @throws std::runtime_error if cache_flush_supported() is false.
 */
void cache_flush_range(uintptr_t ptr, size_t size);

}  // namespace lmcache_native
}  // namespace lmcache

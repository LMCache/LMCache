// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace lmcache {
namespace lmcache_native {

/** @brief Whether cache_flush_range() works: x86-64 with CLFLUSHOPT. */
bool cache_flush_supported();

/**
 * @brief Write back and invalidate every CPU cache line overlapping a range.
 *
 * For memory several hosts map without hardware coherence: a writer flushes
 * after its DMA so the bytes reach the device, a reader before its DMA so no
 * stale line survives. The trailing SFENCE (only SFENCE/MFENCE order
 * CLFLUSHOPT) completes the flushes before any later store, such as the
 * doorbell that starts the next GPU copy.
 *
 * @param ptr Start address of the range in this process.
 * @param size Length of the range in bytes; 0 is a no-op.
 * @throws std::runtime_error if cache_flush_supported() is false.
 */
void cache_flush_range(uintptr_t ptr, size_t size);

}  // namespace lmcache_native
}  // namespace lmcache

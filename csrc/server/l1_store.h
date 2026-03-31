// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — L1 slab storage interface
//
// Abstract interface for the in-memory (L1) KV cache store.
// The slab is mmap'd + cudaHostRegister'd for zero-copy DMA.
//
// State machine per slot: Free → Writing → Ready → Reading → Ready
//                                                  ↘ Free (evict/delete)
//
// Uses the existing C++ TTLLock for lock state management.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.h"

// Forward-declare existing C++ types
namespace lmcache {
namespace storage_manager {
class Bitmap;
}
}  // namespace lmcache

namespace lmcache {
namespace server {

// ============================================================================
// L1Store configuration
// ============================================================================

struct L1StoreConfig {
  size_t capacity_bytes;    // Total slab capacity
  bool use_hugepages;       // mmap with MAP_HUGETLB
  bool cuda_host_register;  // cudaHostRegister for zero-copy DMA
  uint32_t ttl_seconds;     // TTL for read locks (default 300)
};

// ============================================================================
// L1Store — abstract interface
// ============================================================================

class L1Store {
 public:
  virtual ~L1Store() = default;

  // ---- Write path ----

  /// Reserve write slots for the given keys.
  /// Transitions matching free slots to Writing state.
  /// Returns a map of successfully reserved keys to their slab references.
  virtual std::unordered_map<ObjectKey, MemorySlabRef, ObjectKeyHash>
  reserve_write(const std::vector<ObjectKey>& keys,
                const MemoryLayoutDesc& layout, const std::string& mode) = 0;

  /// Mark writing complete, transition to Ready state.
  /// Triggers L2 store via callback if configured.
  virtual void finish_write(const std::vector<ObjectKey>& keys) = 0;

  // ---- Read path ----

  /// Reserve read slots for the given keys.
  /// Returns slab references for keys in Ready state.
  /// extra_count: additional reader count for MLA multi-reader locking.
  virtual std::unordered_map<ObjectKey, MemorySlabRef, ObjectKeyHash>
  reserve_read(const std::vector<ObjectKey>& keys, int extra_count = 0) = 0;

  /// Release read locks.
  virtual void finish_read(const std::vector<ObjectKey>& keys,
                           int extra_count = 0) = 0;

  // ---- Lookup ----

  /// Find the longest contiguous prefix of keys that exist in L1.
  /// Returns the number of prefix hits.
  virtual int64_t prefix_lookup(const std::vector<ObjectKey>& keys) = 0;

  // ---- Deletion / eviction ----

  /// Delete a specific key from L1.
  virtual L1Error delete_key(const ObjectKey& key) = 0;

  /// Evict entries to free at least `bytes_needed` bytes.
  /// Returns the number of bytes actually freed.
  virtual size_t evict(size_t bytes_needed) = 0;

  // ---- Capacity / diagnostics ----

  virtual size_t total_capacity_bytes() const = 0;
  virtual size_t used_bytes() const = 0;

  /// Clear all entries (optionally forced even if locked).
  virtual void clear(bool force = false) = 0;

  /// Memory consistency check (debug).
  virtual bool memcheck() const = 0;

  // ---- Factory ----

  /// Create an L1Store instance.
  static std::unique_ptr<L1Store> create(const L1StoreConfig& config);
};

}  // namespace server
}  // namespace lmcache

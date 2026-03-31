// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — L2 async I/O adapter interface
//
// Abstract interface mirroring Python L2AdapterInterface.
// Non-blocking submit/query model with eventfd signalling for
// integration with the ZMQ poll loop.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "types.h"

// Forward-declare Bitmap from existing C++ code
namespace lmcache {
namespace storage_manager {
class Bitmap;
}
}  // namespace lmcache

namespace lmcache {
namespace server {

/// Opaque task identifier for L2 async operations
using L2TaskId = int64_t;

// ============================================================================
// L2Adapter — abstract interface for L2 storage backends
// ============================================================================

class L2Adapter {
 public:
  virtual ~L2Adapter() = default;

  // ---- Event file descriptors ----
  //
  // Each of the three methods MUST return a distinct fd.
  // The main loop polls these to know when results are available.

  /// eventfd for completed store tasks
  virtual int get_store_event_fd() const = 0;

  /// eventfd for completed lookup-and-lock tasks
  virtual int get_lookup_and_lock_event_fd() const = 0;

  /// eventfd for completed load tasks
  virtual int get_load_event_fd() const = 0;

  // ---- Store path ----

  /// Submit a store task (fire-and-forget with task tracking).
  /// @param keys     Object keys to store
  /// @param data     Corresponding slab references with data
  /// @return         Task ID for tracking completion
  virtual L2TaskId submit_store_task(
      const std::vector<ObjectKey>& keys,
      const std::vector<MemorySlabRef>& data) = 0;

  /// Pop all completed store tasks.
  /// @return Map of task_id → success/failure
  virtual std::unordered_map<L2TaskId, bool> pop_completed_store_tasks() = 0;

  // ---- Lookup and lock path ----

  /// Submit a lookup-and-lock task (checks existence, pins objects).
  /// @param keys  Keys to look up
  /// @return      Task ID for querying results
  virtual L2TaskId submit_lookup_and_lock_task(
      const std::vector<ObjectKey>& keys) = 0;

  /// Query lookup-and-lock result.
  /// @param task_id  Task ID from submit_lookup_and_lock_task
  /// @return         Bitmap of hits (1=found), or nullptr if not ready
  ///                 Non-idempotent: result is consumed on read.
  virtual storage_manager::Bitmap* query_lookup_and_lock_result(
      L2TaskId task_id) = 0;

  /// Release pins acquired during lookup (fire-and-forget).
  virtual void submit_unlock(const std::vector<ObjectKey>& keys) = 0;

  // ---- Load path ----

  /// Submit an L2-to-L1 load task.
  /// @param keys     Keys to load
  /// @param targets  L1 slab references to write into
  /// @return         Task ID for querying results
  virtual L2TaskId submit_load_task(
      const std::vector<ObjectKey>& keys,
      const std::vector<MemorySlabRef>& targets) = 0;

  /// Query load result.
  /// @param task_id  Task ID from submit_load_task
  /// @return         Bitmap of per-key success, or nullptr if not ready
  ///                 Non-idempotent: result is consumed on read.
  virtual storage_manager::Bitmap* query_load_result(L2TaskId task_id) = 0;

  // ---- Lifecycle ----

  /// Close the adapter and release resources (including eventfds).
  virtual void close() = 0;

  /// Health check.
  virtual bool is_healthy() const = 0;
};

}  // namespace server
}  // namespace lmcache

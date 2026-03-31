// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — Main cache engine orchestrator
//
// Mirrors Python MPCacheEngine: ties together GPUContext, L1Store,
// L2Adapter, TokenHasher, SessionManager, and the async prefetch pool.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_context.h"
#include "l1_store.h"
#include "l2_adapter.h"
#include "session_manager.h"
#include "token_hasher.h"
#include "types.h"

// Forward-declare Bitmap
namespace lmcache {
namespace storage_manager {
class Bitmap;
}
}  // namespace lmcache

namespace lmcache {
namespace server {

// ============================================================================
// CacheEngine — main orchestrator
// ============================================================================

class CacheEngine {
 public:
  /// @param chunk_size       Tokens per chunk (e.g. 256)
  /// @param l1_config        L1 slab store configuration
  /// @param l2_adapter       Optional L2 adapter (nullptr = L1 only)
  CacheEngine(int chunk_size, const L1StoreConfig& l1_config,
              std::unique_ptr<L2Adapter> l2_adapter = nullptr);

  ~CacheEngine();

  // Non-copyable
  CacheEngine(const CacheEngine&) = delete;
  CacheEngine& operator=(const CacheEngine&) = delete;

  // ---- GPU registration ----

  void register_kv_cache(int instance_id,
                         const std::vector<CudaIpcTensorDesc>& kv_caches,
                         const std::string& model_name, int world_size);

  void unregister_kv_cache(int instance_id);

  // ---- Store / Retrieve ----

  /// Store GPU KV cache blocks to L1 (+ async L2).
  /// Returns (event_ipc_handle, success).
  std::pair<std::vector<uint8_t>, bool> store(
      const IPCCacheEngineKey& key, int instance_id,
      const std::vector<int32_t>& gpu_block_ids,
      const std::vector<uint8_t>& event_ipc_handle);

  /// Retrieve L1 KV cache into GPU blocks.
  /// Returns (event_ipc_handle, success).
  std::pair<std::vector<uint8_t>, bool> retrieve(
      const IPCCacheEngineKey& key, int instance_id,
      const std::vector<int32_t>& gpu_block_ids,
      const std::vector<uint8_t>& event_ipc_handle,
      int skip_first_n_tokens = 0);

  // ---- Lookup paths ----

  /// Async lookup: returns a prefetch job ID.
  int lookup(const IPCCacheEngineKey& key, int tp_size);

  /// Query the number of hits for a prefetch request before it's finished.
  /// Returns the hit count if lookup is done, -1 if still in progress or
  /// invalid.
  int query_prefetch_lookup_hits(int prefetch_job_id);

  /// Poll async lookup status.  Returns chunk count or -1 if still in progress.
  int query_prefetch_status(int prefetch_job_id);

  /// Release read locks acquired during lookup.
  void free_lookup_locks(const IPCCacheEngineKey& key, int tp_size);

  // ---- Utility ----

  bool ping() const { return true; }
  int get_chunk_size() const { return chunk_size_; }
  void end_session(const std::string& request_id);
  void clear();
  void close();

 private:
  // L2-to-L1 prefetch helper (runs on prefetch pool)
  int run_prefetch_load(
      const std::string& request_id,
      const std::vector<ObjectKey>& remaining_keys,
      const MemoryLayoutDesc& layout_desc,
      std::unordered_map<int, storage_manager::Bitmap*> l2_lookup_results,
      int extra_count);

  // Find layout desc from a matching GPU context
  MemoryLayoutDesc find_layout_desc(const std::string& model_name,
                                    int world_size) const;

  int chunk_size_;

  // GPU contexts: instance_id → GPUContext
  std::unordered_map<int, std::unique_ptr<GPUContext>> gpu_contexts_;
  // GPU context metadata: instance_id → (model_name, world_size)
  std::unordered_map<int, std::pair<std::string, int>> gpu_context_meta_;

  // Storage
  std::unique_ptr<L1Store> l1_store_;
  std::unique_ptr<L2Adapter> l2_adapter_;

  // Hashing and sessions
  TokenHasher token_hasher_;
  SessionManager session_manager_;

  // Prefetch job tracking
  struct PrefetchJob {
    PrefetchHandle handle;
    int world_size;
    std::string request_id;
  };
  std::mutex prefetch_job_lock_;
  std::unordered_map<int, PrefetchJob> prefetch_jobs_;
  int next_prefetch_job_id_ = 0;

  // Thread pool for async L2-to-L1 prefetch (8 workers)
  struct ThreadPool;
  std::unique_ptr<ThreadPool> prefetch_pool_;

  // Global engine lock (legacy, for clear())
  std::mutex lock_;
};

}  // namespace server
}  // namespace lmcache

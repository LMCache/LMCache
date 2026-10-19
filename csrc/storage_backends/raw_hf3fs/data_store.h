// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.
#pragma once

// Single-file + offset management for LMCache 3FS backend.
//
// Each key gets a stride-aligned region (stride=1MB) in a single 3FS file.
// The 3FS chunk_size equals the LMCache data_size, so each key maps to exactly
// 1 3FS chunk with no internal waste.
//
// When data_size changes, new store files are created (old files untouched).
// The offset_map is persisted to chunk 0 (meta region) of the .store file via
// StorageClient RDMA (not the FUSE mount), so it can be recovered after
// restart.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Include full 3FS headers — needed because DataStore internally creates
// MetaClient objects, accesses Layout, SessionId, InodeId, etc.
#include "client/meta/MetaClient.h"
#include "fbs/meta/Common.h"
#include "fbs/meta/Schema.h"
#include "fbs/mgmtd/RoutingInfo.h"

namespace hf3fs {
namespace storage {
namespace client {
class StorageClient;
}
}  // namespace storage
}  // namespace hf3fs

namespace lmcache {
namespace connector {

/**
 * Manages a single 3FS file as a key-value store.
 *
 * - Creates `<base_dir>/data_<index>.store` on init_file().
 * - Each key gets a stride-aligned region with stride=1MB per key.
 * - 3FS chunk_size = data_size: each key maps to 1 chunk, no waste.
 * - Thread-safe offset allocation per file.
 * - offset_map persisted to .meta file for recovery on restart.
 */
class DataStore {
 public:
  // Internal stride for buffer layout: 1MB for tight packing.
  static constexpr uint32_t INTERNAL_STRIDE = 1024 * 1024;  // 1 MB
  // 3FS layout chunk cap for large keys. Keys over STRIPE_THRESHOLD are
  // stored as multiple STRIPE_CHUNK_SIZE chunks striped across chains, so a
  // big key is read from several chains in parallel (~8.5GB/s) instead of
  // one large chunk on one chain (~5GB/s).
  static constexpr uint32_t STRIPE_THRESHOLD = 16 * 1024 * 1024;  // 16 MB
  static constexpr uint32_t STRIPE_CHUNK_SIZE = 4 * 1024 * 1024;  // 4 MB

  // Runtime override for STRIPE_CHUNK_SIZE (bytes) via env
  // LMCACHE_3FS_STRIPE_CHUNK_SIZE, read once. Lets us tune the layout chunk
  // used for large keys (>= STRIPE_THRESHOLD) without recompiling. 0 = unset.
  static uint32_t override_stripe_chunk_size() {
    static uint32_t v = [] {
      const char* e = std::getenv("LMCACHE_3FS_STRIPE_CHUNK_SIZE");
      return (e && *e) ? static_cast<uint32_t>(std::max(0, std::atoi(e))) : 0u;
    }();
    return v;
  }
  // Effective STRIPE_CHUNK_SIZE (env override wins).
  static uint32_t effective_stripe_chunk_size() {
    uint32_t ov = override_stripe_chunk_size();
    return ov ? ov : STRIPE_CHUNK_SIZE;
  }

  DataStore() = default;
  DataStore(uint32_t file_index, uint32_t data_size,
            uint32_t chain_table_id = 1)
      : file_index_(file_index),
        stride_(INTERNAL_STRIDE),
        chunk_size_(data_size),
        layout_chunk_size_(data_size > STRIPE_THRESHOLD
                               ? effective_stripe_chunk_size()
                               : data_size) {
    // Set in the body (not the init list) to keep declaration order simple.
    chain_table_id_ = chain_table_id;
  }

  // Create the data_<index>.store file and cache chain/chunk ID info.
  // When data_size differs from existing store, creates a new store file.
  // base_dir: 3FS root-relative directory holding this file (default
  // "lmcache"). num_files_: total number of data files (for distributing chains
  // across files).
  void init_file(hf3fs::meta::client::MetaClient* mc,
                 hf3fs::storage::client::StorageClient* sc,
                 const hf3fs::flat::UserInfo& ui,
                 const hf3fs::flat::RoutingInfo& routing_info,
                 uint32_t chunk_size, const std::string& base_dir = "lmcache",
                 uint32_t num_files = 1, uint32_t file_permission = 0644);

  // Look up offset for key only (no allocation). Returns (offset, size).
  // Size is 0 if the key is not found.
  std::pair<uint64_t, uint32_t> lookup_offset(const std::string& key) const;

  // Look up offset for key, or allocate a new one with given data_len.
  // Thread-safe. Uses key_stride for per-key region allocation (matches
  // RDMA buffer layout). Returns (offset, size). Size is 0 if newly allocated.
  std::pair<uint64_t, uint32_t> get_or_allocate_offset(const std::string& key,
                                                       size_t data_len,
                                                       size_t key_stride = 0);

  // Remove key from offset map (for delete). Returns true if key was found.
  bool remove_offset(const std::string& key);

  // Check if key exists in map.
  bool contains(const std::string& key) const;

  // Extend the data file via MetaClient sync to accommodate new_size.
  void extend_file_size(uint64_t new_size);

  // Persist offset_map to chunk 0 of the .store file via StorageClient RDMA.
  void save_meta();

  // Load offset_map from .meta file (call on init, before any operations).
  // Returns true if meta file exists and was loaded successfully.
  bool load_meta();

  // Get the cached chain ID raw value for chunks in this file.
  uint32_t get_chain_id_raw() const { return chain_id_; }

  // Get the file index (0..num_files-1).
  uint32_t get_file_index() const { return file_index_; }

  // Get the meta chunk ID (16-byte packed string) for the given file offset.
  std::string get_meta_chunk_id(uint64_t offset) const;

  // Get the file size after extension.
  uint64_t get_file_size() const { return current_file_size_; }

  // Get the layout for computing chain/chunk IDs.
  const hf3fs::meta::Layout& get_layout() const { return layout_; }

  // Get the cached inode ID.
  const hf3fs::meta::InodeId& get_inode_id() const { return cached_inode_id_; }

  // Get the cached routing info.
  const hf3fs::flat::RoutingInfo& get_routing_info() const {
    return routing_info_;
  }

  // Refresh routing info from a fresh RoutingInfo snapshot.
  // Called after MgmtdClient session reset to update cached stripe size,
  // layout, chain IDs, and chain index list without re-opening the file.
  void refresh_routing_info(const hf3fs::flat::RoutingInfo& new_routing_info);

  // Build chainIndex->chainId mapping from the ordered chain table in
  // routing_info_.chainTables.  Falls back to iterating routing_info_.chains
  // (unordered) if chainTables is not available.
  void rebuild_chain_id_map_from_tables();

  // Get chain ID for a specific chunk index (file offset / stride).
  uint32_t get_chain_id_for_chunk(uint64_t chunk_index) const;

  // Sync file metadata to persist the current file size.
  void sync_file_size(hf3fs::meta::client::MetaClient* mc,
                      const hf3fs::flat::UserInfo& ui, bool written);

  // Sync file metadata only if dirty_count_ >= threshold.
  void sync_if_dirty(hf3fs::meta::client::MetaClient* mc,
                     const hf3fs::flat::UserInfo& ui, bool written,
                     int threshold = 128);

  // Mark meta as dirty (call when offset_map changes).
  void mark_meta_dirty();

  // Start the periodic meta saver background thread.
  void start_meta_saver();

  // Stop the periodic meta saver background thread.
  void stop_meta_saver();

  // Batch look up or allocate offsets for multiple keys in a single lock
  // acquisition.
  std::vector<std::pair<uint64_t, uint32_t>> batch_lookup_or_allocate(
      const std::vector<std::string>& keys,
      const std::vector<size_t>& data_lens);

  uint32_t get_chunk_size() const { return chunk_size_; }
  uint32_t get_layout_chunk_size() const { return layout_chunk_size_; }
  uint32_t get_stride() const { return stride_; }
  uint32_t get_stripe_size() const { return stripe_size_; }

  void inc_dirty_count();
  int get_and_reset_dirty_count();

  // Lock and lookup/allocate a single key's offset.
  // allocatedRegion uses key_stride for per-key offset spacing (must match
  // the RDMA buffer layout stride, which is key_size).
  std::pair<uint64_t, uint32_t> lookup_or_alloc(const std::string& key,
                                                size_t data_len,
                                                size_t key_stride = 0);

  // Lock and lookup/allocate a batch of keys' offsets.
  std::vector<std::pair<uint64_t, uint32_t>> batch_lookup_or_alloc(
      const std::vector<std::string>& keys,
      const std::vector<size_t>& data_lens, size_t key_stride = 0);

  const std::vector<uint32_t>& get_chain_index_list() const {
    return chain_index_list_;
  }
  const std::unordered_map<uint32_t, uint32_t>& get_chain_id_map();

  // Get the legacy FUSE meta path (unused for I/O).
  std::string get_meta_path() const;

  // Clear internal state: offset_map, next_offset, file size, dirty state.
  // Used for cleanup between tests. Thread-safe (locks offset_mu_).
  void clear_state();

 private:
  // Background thread for periodic meta saving.
  void meta_saver_loop();

  mutable std::mutex offset_mu_;
  std::unordered_map<std::string, std::pair<uint64_t, uint32_t>> offset_map_;
  uint64_t next_offset_{0};
  uint64_t current_file_size_{0};
  uint32_t stride_{0};
  uint32_t chunk_size_{0};
  uint32_t layout_chunk_size_{0};
  uint32_t stripe_size_{0};
  uint32_t num_files_{1};

  int dirty_count_{0};
  uint32_t file_index_{0};
  uint32_t chain_table_id_{1};
  uint32_t chain_id_{0};
  hf3fs::meta::InodeId cached_inode_id_;
  hf3fs::meta::Layout layout_;
  hf3fs::flat::RoutingInfo routing_info_;
  std::string base_dir_{"lmcache"};
  std::string file_path_;  // "<base_dir>/data_<index>.store"
  std::string meta_path_;  // legacy FUSE meta path (no longer used for I/O)
  std::vector<uint32_t> chain_index_list_;
  mutable std::unordered_map<uint32_t, uint32_t> chain_id_map_;

  // Clients for RDMA-based offset_map persistence (set in init_file, read-only
  // by the meta_saver thread; owners keep them alive until after
  // stop_meta_saver).
  hf3fs::storage::client::StorageClient* storage_client_{nullptr};
  hf3fs::flat::UserInfo user_info_;

  // Lazy meta saving fields.
  std::mutex meta_save_mu_;
  std::condition_variable meta_save_cv_;
  bool meta_dirty_{false};
  bool meta_save_stop_{false};
  std::thread meta_saver_thread_;
  static constexpr std::chrono::milliseconds META_SAVE_INTERVAL =
      std::chrono::milliseconds(5000);
};

}  // namespace connector
}  // namespace lmcache

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.
#pragma once

// Single-file + offset management for LMCache 3FS backend.
//
// Every key is allocated ceil(size / cs) COMPLETE layout chunks
// (cs = layout_chunk_size_, default 4MB), aligned to a chunk boundary, and a
// key's chunks are always a CONTIGUOUS chunk-index run (the write/read paths
// derive chunk ids by walking indices from offset, and the 28B meta entry
// stores only (offset, size)). The 3FS server serializes writes that share a
// chunk, so co-locating concurrent small keys into one chunk collapses Store
// throughput (measured 2026-10-22: 4MB keys 8-per-chunk ~907 MB/s vs
// 1-per-chunk ~2870 MB/s). A key smaller than cs wastes its chunk tail.
//
// Chunk 0 is the meta region: it carries the v4 header (see data_store.cc),
// a 64B owner lease and the first batch of persisted offset_map entries.
// When entries outgrow chunk 0, additional meta chunks are allocated on
// demand (their indices are recorded in the chunk 0 header, up to
// META_LIST_MAX), so the persisted map scales with key count instead of
// being capped at one chunk. Data allocation starts at offset = cs.
//
// Deletion: remove_offset() erases the key and queues its chunks for
// PHYSICAL release (removeChunks) by the background meta_saver thread.
// Released chunks enter the free list only after removeChunks succeeds, so a
// reallocated chunk can never be physically removed underneath a live key.
// New keys prefer a best-fit contiguous free range (contiguity keeps the
// key's chunks on distinct chains via the chunk->chain stripe); if no range
// fits, allocation falls back to the bump pointer.
//
// Multi-instance protection: the 64B owner lease in chunk 0 records a random
// instance id, a heartbeat timestamp and the hostname. init_file() refuses
// to start when a DIFFERENT live instance's lease is fresh (heartbeat within
// LEASE_TTL_SEC): two instances sharing one data_*_store silently corrupt
// data (independent offset maps double-allocate the same chunks and each
// save_meta overwrites the other's map). The heartbeat is refreshed by the
// meta_saver thread; close() releases the lease so a clean restart takes
// over immediately; a crashed instance's lease expires after the TTL.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <map>
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
 * - Creates `<base_dir>/data_<chunk_size>_store` on init_file().
 * - Each key occupies ceil(size/cs) whole layout chunks (chunk-aligned).
 * - Thread-safe offset allocation.
 * - offset_map persisted to chunk 0 of the store file for recovery on restart.
 */
class DataStore {
 public:
  // cs = the 3FS layout chunk / per-WriteIO transfer block (already normalized
  // to a multiple of 4MB by the connector). Keys larger than cs span multiple
  // chunks; the connector splits each key's byte range at cs boundaries so a
  // single WriteIO never crosses a chunk (keys need not be cs-aligned).
  DataStore(uint32_t file_index, uint32_t cs, uint32_t chain_table_id = 1)
      : file_index_(file_index), chunk_size_(cs), layout_chunk_size_(cs) {
    // Set in the body (not the init list) to keep declaration order simple.
    chain_table_id_ = chain_table_id;
  }

  // Create the data_<chunk_size>_store file and cache chain/chunk ID info.
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

  // Canonical in-map form of a key: any wire-format key string (e.g.
  // "model@rank@group@hash[@salt]") is folded deterministically into
  // "<hi>:<lo>" (SpookyHashV2 128-bit, fixed seeds). All map operations
  // go through fold_key(), so lookup/insert/delete agree on one key
  // identity and the persisted snapshot round-trips after restart.
  // The fold is one-way: the original key string cannot be recovered
  // from the persisted meta (the snapshot is not self-describing).
  static std::string fold_key(const std::string& raw_key);

  // Look up offset for key, or allocate a new one with given data_len.
  // Thread-safe. New keys occupy ceil(size/cs) whole chunks (see file
  // header). Returns (offset, size). Size is 0 if newly allocated.
  std::pair<uint64_t, uint32_t> get_or_allocate_offset(const std::string& key,
                                                       size_t data_len,
                                                       size_t key_stride = 0);

  // Remove key from offset map (for delete). Returns true if key was found.
  bool remove_offset(const std::string& key);

  // Check if key exists in map.
  bool contains(const std::string& key) const;

  // Extend the data file via MetaClient sync to accommodate new_size.
  void extend_file_size(uint64_t new_size);

  // Persist offset_map to chunk 0 of the store file via StorageClient RDMA.
  void save_meta();

  // Load offset_map from chunk 0 of the store file (call on init, before
  // any operations). Returns true if a meta blob exists and loaded OK.
  bool load_meta();

  // Get the cached chain ID raw value for chunks in this file.
  uint32_t get_chain_id_raw() const { return chain_id_; }

  // Get the file index (0..num_files-1).
  uint32_t get_file_index() const { return file_index_; }

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

  // Get chain ID for a specific chunk index (file offset / cs).
  uint32_t get_chain_id_for_chunk(uint64_t chunk_index) const;

  // Whether init_file() succeeded (false when refused by a foreign owner
  // lease — the caller must not publish the store).
  bool init_ok() const { return !init_refused_; }

  // Zero the owner lease on clean shutdown so the next instance can start
  // without waiting for the TTL. No-op if this store never wrote a lease.
  void release_lease();

  // Re-read the lease from chunk 0 and report whether a DIFFERENT live
  // instance owns the file (used to guard cleanup_data_files).
  bool lease_blocked_by_foreign();

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
  uint32_t get_stripe_size() const { return stripe_size_; }

  void inc_dirty_count();
  int get_and_reset_dirty_count();

  // Lock and lookup/allocate a single key's offset.
  // New keys get ceil(size/cs) whole chunks (chunk-aligned); the WriteIO
  // boundary split at cs happens in the connector.
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

  // Clear internal state: offset_map, next_offset, file size, dirty state.
  // Used for cleanup between tests. Thread-safe (locks offset_mu_).
  void clear_state();

 private:
  // 64B owner lease record stored at LEASE_OFFSET inside chunk 0.
  struct LeaseData {
    uint64_t id_hi;
    uint64_t id_lo;
    int64_t heartbeat;  // unix epoch seconds, writer's clock
    char host[40];
  };

  uint64_t alloc_new_locked(uint32_t size);

  // Allocate one chunk for the meta region (free list first, bump fallback).
  // offset_mu_ must be held.
  uint64_t alloc_meta_chunk_locked();

  // RDMA read/write of the 64B lease region in chunk 0.
  bool read_lease(LeaseData& out);
  bool write_lease(const LeaseData& in);

  // Physically remove queued chunks (removeChunks) and move the succeeded
  // ones into the free list. Runs on the meta_saver thread.
  void release_pending_chunks();

  // Merge physically-released chunk indices into free_ranges_ (sorted,
  // adjacency-merged). offset_mu_ must be held.
  void merge_into_free_list_locked(std::vector<uint64_t> chunk_indices);

  // Whether a lease record belongs to a different, live instance.
  bool lease_is_foreign_live(const LeaseData& ld) const;

  // Resolve the chain for a chunk index via chain_id_map_ (same lookup the
  // data path uses; get_chain_id_for_chunk is unreliable under mgmtd
  // routing — see .cc).
  uint32_t resolve_chain_for_chunk(uint64_t chunk_idx);

  // Background thread for periodic meta saving.
  void meta_saver_loop();

  mutable std::mutex offset_mu_;
  std::unordered_map<std::string, std::pair<uint64_t, uint32_t>> offset_map_;
  uint64_t next_offset_{0};
  uint64_t current_file_size_{0};
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
  std::string file_path_;  // "<base_dir>/data_<chunk_size>_store"
  std::vector<uint32_t> chain_index_list_;
  mutable std::unordered_map<uint32_t, uint32_t> chain_id_map_;

  // Clients for RDMA-based offset_map persistence (set in init_file, read-only
  // by the meta_saver thread; owners keep them alive until after
  // stop_meta_saver).
  hf3fs::storage::client::StorageClient* storage_client_{nullptr};
  hf3fs::flat::UserInfo user_info_;

  // Whole-chunk free list: sorted chunk-index range map (start -> length in
  // chunks). Populated ONLY by the background physical-release path, so
  // every chunk in here is already removed from the storage targets.
  std::map<uint64_t, uint64_t> free_ranges_;

  // Chunk indices queued for physical removeChunks release.
  std::vector<uint64_t> pending_release_;

  // Meta extension chunk indices (chunk 0 is implicit and never listed).
  // Monotonic high-water list, persisted in the chunk 0 header.
  std::vector<uint32_t> meta_chunk_list_;

  // Owner lease state (see file header).
  uint64_t lease_id_hi_{0};
  uint64_t lease_id_lo_{0};
  bool lease_owned_{false};
  bool init_refused_{false};
  std::string lease_host_;

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

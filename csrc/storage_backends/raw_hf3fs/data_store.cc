// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include "data_store.h"

#include <atomic>

// DBG macro for runtime-controlled debug logging.
// Controlled by LMCACHE_DBG=1 env var (default off).
// get_dbg_flag() is defined in hf3fs_connector.h and set in hf3fs_connector.cc.
#include "hf3fs_connector.h"
#define DBG(fmt, ...)                                            \
  do {                                                           \
    if (get_dbg_flag())                                          \
      fprintf(stderr, "[LMCache 3FS] " fmt "\n", ##__VA_ARGS__); \
  } while (0)

#include <folly/executors/InlineExecutor.h>
#include <folly/hash/SpookyHashV2.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <future>
#include <random>
#include <span>
#include <string_view>
#include <unistd.h>

#include "client/meta/MetaClient.h"
#include "client/mgmtd/RoutingInfo.h"
#include "client/storage/StorageClient.h"
#include "fbs/meta/Common.h"
#include "fbs/meta/Schema.h"
#include "fbs/mgmtd/RoutingInfo.h"

namespace lmcache {
namespace connector {

// Binary offset_map meta format persisted to the meta region of the store
// file (RDMA, not FUSE). Keeps atomicity on the storage side without the
// FUSE mount.
//
// v4 chunk 0 layout (offsets in bytes within chunk 0):
//   0     magic[4]            'LM3M'
//   4     version u8          4
//   5     chunk_size u32
//   9     total_entries u32   entries across all meta chunks
//   13    next_offset u64
//   21    file_size u64
//   29    num_chunk0_entries u32
//   33    num_meta_chunks u32 (>= 1, includes chunk 0)
//   37    owner lease 64B: instance_id_hi u64, instance_id_lo u64,
//         heartbeat i64 (unix epoch sec), host char[40]
//   101   meta_chunk_list u32[META_LIST_MAX]  (extension chunk indices;
//         entries [0, num_meta_chunks-1) are valid)
//   4197  entries (28B each) belonging to chunk 0
//
// Extension meta chunks (indices from meta_chunk_list) hold META_EXT_CAP
// dense entries each. Header v4 total size is META_HEADER_SIZE = 4197.
constexpr char META_MAGIC[4] = {'L', 'M', '3', 'M'};
constexpr uint8_t META_VERSION = 5;
constexpr size_t META_ENTRY_SIZE = 8 + 8 + 8 + 4;
constexpr size_t META_HEADER_SIZE = 37 + 64 + 4096;
constexpr size_t LEASE_OFFSET = 37;
constexpr size_t LEASE_SIZE = 64;
constexpr size_t META_LIST_OFFSET = 101;
constexpr size_t META_LIST_MAX = 1024;
constexpr int64_t LEASE_TTL_SEC = 60;

// Page-aligned buffer for RDMA registration (mirrors alloc_rdma).
static uint8_t* meta_buf_alloc(size_t sz) {
  if (sz == 0) sz = 1;
  long ps = sysconf(_SC_PAGESIZE);
  size_t as = (static_cast<size_t>(sz) + ps - 1) / ps * ps;
  void* p = nullptr;
  if (posix_memalign(&p, static_cast<size_t>(ps), as) != 0) return nullptr;
  return static_cast<uint8_t*>(p);
}

void DataStore::init_file(hf3fs::meta::client::MetaClient* mc,
                          hf3fs::storage::client::StorageClient* sc,
                          const hf3fs::flat::UserInfo& ui,
                          const hf3fs::flat::RoutingInfo& routing_info,
                          uint32_t chunk_size, const std::string& base_dir,
                          uint32_t num_files, uint32_t file_permission) {
  if (!mc) return;

  // Keep clients for RDMA-based offset_map persistence (save_meta/load_meta).
  storage_client_ = sc;
  user_info_ = ui;

  chunk_size_ = chunk_size;
  num_files_ = num_files > 0 ? num_files : 1;
  base_dir_ = base_dir.empty() ? "lmcache" : base_dir;
  DBG("init_file: file_index=%u chunk_size=%u layout_chunk_size_=%u "
      "base_dir=%s",
      file_index_, chunk_size_, layout_chunk_size_, base_dir_.c_str());

  // Build file path relative to 3FS root (MetaClient path resolution requires
  // relative paths). The chunk_size is embedded in the name so that changing
  // cs starts a fresh file instead of colliding with a store written under a
  // different layout (old-cs files are left orphaned for the operator).
  file_path_ = base_dir_ + "/data_" + std::to_string(chunk_size_) + "_store";

  // Number of chains in routing_info determines stripe size.
  size_t num_chains = routing_info.chains.size();
  stripe_size_ = static_cast<uint32_t>(num_chains);
  DBG("init_file: num_chains=%zu, stripe_size=%u, chainTables.size=%zu",
      num_chains, stripe_size_, routing_info.chainTables.size());
  // Build ClusterRange layout with all chains for this file.
  // Each file stripes across ALL nodes, enabling full parallel bandwidth.
  // Keys larger than cs span several layout chunks striped across chains;
  // the connector splits each key's byte range at cs boundaries.
  layout_ = hf3fs::meta::Layout::newChainRange(
      hf3fs::flat::ChainTableId(chain_table_id_),
      hf3fs::flat::ChainTableVersion(routing_info.routingInfoVersion),
      layout_chunk_size_, stripe_size_, 0);

  // Cache routing info for chain ID computation.
  routing_info_ = routing_info;

  hf3fs::Path path(file_path_);
  hf3fs::meta::client::SessionId session =
      hf3fs::meta::client::SessionId::random();

  // Try to open first.
  DBG("init_file: try open %s", file_path_.c_str());
  auto openRes = folly::coro::blockingWait(
      mc->open(ui, hf3fs::meta::InodeId::root(), path, session,
               static_cast<int>(hf3fs::meta::OpenFlags(O_RDWR))));
  DBG("init_file: open %s done, hasError=%d", file_path_.c_str(),
      (int)openRes.hasError());

  hf3fs::meta::Inode fileInode;
  if (openRes.hasError()) {
    // File doesn't exist, create it.
    DBG("init_file: try create %s", file_path_.c_str());
    auto createRes = folly::coro::blockingWait(
        mc->create(ui, hf3fs::meta::InodeId::root(), path, session,
                   hf3fs::flat::Permission(file_permission),
                   static_cast<int>(hf3fs::meta::OpenFlags(O_RDWR | O_CREAT)),
                   std::nullopt));
    DBG("init_file: create %s done, hasError=%d", file_path_.c_str(),
        (int)createRes.hasError());
    if (createRes.hasError()) {
      init_refused_ = true;
      fprintf(stderr,
              "[LMCache 3FS] ERROR: cannot open or create data store %s: "
              "%s. Check that base_dir exists or can be created. All "
              "store/load operations on this connector will fail.\n",
              file_path_.c_str(), createRes.error().describe().c_str());
      return;
    }
    fileInode = *createRes;

    // Sync the newly created file to persist it.
    hf3fs::meta::VersionedLength vlen{0, 0};
    auto syncRes = folly::coro::blockingWait(
        mc->sync(ui, fileInode.id, /*read=*/false, /*written=*/false, vlen));
    if (syncRes.hasError()) {
      DBG("Failed to sync new file %s: %s", file_path_.c_str(),
          syncRes.error().describe().c_str());
    }
  } else {
    fileInode = *openRes;
  }

  // Cache the inode ID.
  cached_inode_id_ = fileInode.id;

  // Cache the chain index list for get_chain_id_for_chunk().
  auto chainSpan = layout_.getChainIndexList();
  chain_index_list_.assign(chainSpan.begin(), chainSpan.end());

  // Build chainIndex -> chainId mapping from the ordered chain table.
  rebuild_chain_id_map_from_tables();
  DBG("init_file: %s inode=%lu chain=%u map_size=%zu", file_path_.c_str(),
      (unsigned long)cached_inode_id_.u64(), chain_id_, chain_id_map_.size());

  // Owner lease: two instances sharing one data_*_store silently corrupt
  // data (independent offset maps double-allocate the same chunks; each
  // save_meta overwrites the other's map). Refuse to start when a
  // different live instance holds a fresh lease.
  {
    std::random_device rd;
    lease_id_hi_ = (static_cast<uint64_t>(rd()) << 32) | rd();
    lease_id_lo_ = (static_cast<uint64_t>(rd()) << 32) | rd();
    char host[64] = {0};
    gethostname(host, sizeof(host) - 1);
    lease_host_ = host;

    LeaseData cur{};
    bool have_lease = read_lease(cur);
    if (have_lease && lease_is_foreign_live(cur)) {
      init_refused_ = true;
      fprintf(stderr,
              "[LMCache 3FS] ERROR: %s is owned by another live LMCache "
              "instance (host=%s, last heartbeat %llds ago). Sharing one "
              "data_*_store across instances corrupts data. Use a separate "
              "base_dir per instance, wait %llds for the lease to expire, or "
              "delete the file.\n",
              file_path_.c_str(), cur.host,
              (long long)(time(nullptr) - cur.heartbeat),
              (long long)LEASE_TTL_SEC);
      return;
    }

    LeaseData mine{};
    mine.id_hi = lease_id_hi_;
    mine.id_lo = lease_id_lo_;
    mine.heartbeat = time(nullptr);
    std::snprintf(mine.host, sizeof(mine.host), "%s", lease_host_.c_str());
    lease_owned_ = write_lease(mine);
    if (!lease_owned_) {
      DBG("init_file: write_lease failed; multi-instance protection "
          "degraded (continuing)");
    }
  }

  // Load existing offset_map from the meta region (via RDMA) if any.
  load_meta();

  // Reserve chunk 0 as the meta region: data allocation starts at chunk_size_
  // so key data never collides with the persisted offset_map.
  {
    std::lock_guard<std::mutex> lock(offset_mu_);
    if (next_offset_ < chunk_size_) next_offset_ = chunk_size_;
    if (current_file_size_ < chunk_size_) current_file_size_ = chunk_size_;
  }
}

void DataStore::rebuild_chain_id_map_from_tables() {
  // Build chainIndex -> chainId mapping from the ordered chain table.
  // routing_info_.chains is an unordered_map<ChainId, ChainInfo> — its
  // iteration order is non-deterministic and does NOT correspond to
  // chain indices 0, 1, 2, ... Therefore we must NOT iterate it to build
  // an index→id mapping.
  //
  // Instead we look up the ChainTable from routing_info_.chainTables using
  // the tableId/tableVersion from our layout.  ChainTable.chains is a
  // vector<ChainId> ordered by chain index, so chains[i] is the ChainId
  // for chain index i.
  chain_id_map_.clear();
  uint32_t fallbackChainId = 0;

  // Try chainTables first (ordered, correct mapping).
  bool chains_from_table = false;
  auto ctIt = routing_info_.chainTables.find(
      hf3fs::flat::ChainTableId(layout_.tableId.toUnderType()));
  if (ctIt != routing_info_.chainTables.end()) {
    auto& versionMap = ctIt->second;
    auto vtIt = versionMap.find(
        hf3fs::flat::ChainTableVersion(layout_.tableVersion.toUnderType()));
    if (vtIt != versionMap.end()) {
      const auto& chainVec = vtIt->second.chains;
      size_t n = std::min(chain_index_list_.size(), chainVec.size());
      for (size_t i = 0; i < n; ++i) {
        uint32_t chainIdx = chain_index_list_[i];
        chain_id_map_[chainIdx] = chainVec[i].toUnderType();
        if (fallbackChainId == 0) {
          fallbackChainId = chainVec[i].toUnderType();
        }
      }
      chains_from_table = true;
    }
  }

  // Fallback: if chainTables didn't have the mapping, iterate chains map
  // (unordered — only safe when all chains map to the same node, which is
  // not ideal but at least gives us *some* mapping).
  if (!chains_from_table && !routing_info_.chains.empty()) {
    size_t chainIdx = 0;
    for (auto& [chainId, chainInfo] : routing_info_.chains) {
      if (chainIdx < chain_index_list_.size()) {
        chain_id_map_[chain_index_list_[chainIdx]] = chainId.toUnderType();
        if (fallbackChainId == 0) {
          fallbackChainId = chainId.toUnderType();
        }
      }
      chainIdx++;
      if (chainIdx >= chain_index_list_.size()) break;
    }
  }

  chain_id_ = fallbackChainId;
}

void DataStore::refresh_routing_info(
    const hf3fs::flat::RoutingInfo& new_routing_info) {
  // Only update if routing version actually changed.
  if (routing_info_.routingInfoVersion == new_routing_info.routingInfoVersion) {
    return;
  }

  DBG("refresh_routing_info: %s version %u -> %u", file_path_.c_str(),
      routing_info_.routingInfoVersion.toUnderType(),
      new_routing_info.routingInfoVersion.toUnderType());

  // Update stripe size and layout with new routing info.
  stripe_size_ = static_cast<uint32_t>(new_routing_info.chains.size());
  layout_ = hf3fs::meta::Layout::newChainRange(
      hf3fs::flat::ChainTableId(chain_table_id_),
      hf3fs::flat::ChainTableVersion(new_routing_info.routingInfoVersion),
      layout_chunk_size_, stripe_size_, 0);

  routing_info_ = new_routing_info;

  // Recompute chain index list.
  auto chainSpan = layout_.getChainIndexList();
  chain_index_list_.assign(chainSpan.begin(), chainSpan.end());

  // Rebuild chain_id_map_ from the ordered chain table (same logic as
  // init_file), since the previous version only invalidated the cache.
  rebuild_chain_id_map_from_tables();

  DBG("refresh_routing_info: %s stripe=%u chain=%u map_size=%zu",
      file_path_.c_str(), stripe_size_, chain_id_, chain_id_map_.size());
}

// Allocate a fresh region for a NEW key (offset_mu_ held; caller has already
// verified the key is absent from offset_map_). The region is a CONTIGUOUS
// run of ceil(size/cs) complete chunks: the write/read paths derive chunk
// ids by walking indices from the returned offset, so scattered placement is
// not representable. Contiguity also implies distinct chains per chunk
// (chunk i stripes to chain i % stripe), preserving chain parallelism.
// Policy: best-fit free range first (released chunks, physically removed
// from the storage targets), bump pointer as fallback.
uint64_t DataStore::alloc_new_locked(uint32_t size) {
  const uint64_t cs = layout_chunk_size_ ? layout_chunk_size_ : chunk_size_;
  uint64_t nchunks = (static_cast<uint64_t>(size) + cs - 1) / cs;

  if (nchunks > 0 && !free_ranges_.empty()) {
    auto best = free_ranges_.end();
    for (auto it = free_ranges_.begin(); it != free_ranges_.end(); ++it) {
      if (it->second >= nchunks &&
          (best == free_ranges_.end() || it->second < best->second)) {
        best = it;
        if (it->second == nchunks) break;  // exact fit
      }
    }
    if (best != free_ranges_.end()) {
      uint64_t start = best->first;
      if (best->second == nchunks) {
        free_ranges_.erase(best);
      } else {
        uint64_t rest = best->second - nchunks;
        free_ranges_.erase(best);
        free_ranges_[start + nchunks] = rest;
      }
      return start * cs;
    }
  }

  uint64_t first = (next_offset_ + cs - 1) / cs;
  next_offset_ = (first + nchunks) * cs;
  if (next_offset_ > current_file_size_) {
    current_file_size_ = next_offset_;
  }
  return first * cs;
}

// Allocate one chunk for the meta region (offset_mu_ held). Free list first
// so released data chunks are recycled; bump pointer as fallback.
uint64_t DataStore::alloc_meta_chunk_locked() {
  if (!free_ranges_.empty()) {
    auto it = free_ranges_.begin();
    uint64_t start = it->first;
    if (it->second > 1) {
      uint64_t rest = it->second - 1;
      free_ranges_.erase(it);
      free_ranges_[start + 1] = rest;
    } else {
      free_ranges_.erase(it);
    }
    return start;
  }
  const uint64_t cs = layout_chunk_size_ ? layout_chunk_size_ : chunk_size_;
  uint64_t idx = (next_offset_ + cs - 1) / cs;
  next_offset_ = (idx + 1) * cs;
  if (next_offset_ > current_file_size_) {
    current_file_size_ = next_offset_;
  }
  return idx;
}

// The fold seeds are part of the persisted metadata format: changing them
// silently orphans every previously-written entry.
constexpr uint64_t FOLD_SEED_HI = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t FOLD_SEED_LO = 0xBF58476D1CE4E5B9ULL;

std::string DataStore::fold_key(const std::string& raw_key) {
  uint64_t hi = FOLD_SEED_HI;
  uint64_t lo = FOLD_SEED_LO;
  folly::hash::SpookyHashV2::Hash128(raw_key.data(), raw_key.size(), &hi, &lo);
  char buf[44];
  int n = std::snprintf(buf, sizeof(buf), "%llu:%llu", (unsigned long long)hi,
                        (unsigned long long)lo);
  return std::string(buf, n);
}

std::pair<uint64_t, uint32_t> DataStore::get_or_allocate_offset(
    const std::string& key, size_t data_len, size_t key_stride) {
  const std::string fk = fold_key(key);
  std::lock_guard<std::mutex> lock(offset_mu_);

  auto it = offset_map_.find(fk);
  if (it != offset_map_.end()) {
    return {it->second.first, it->second.second};
  }

  uint32_t size = static_cast<uint32_t>(data_len);
  uint64_t offset = alloc_new_locked(size);

  offset_map_[fk] = {offset, size};

  return {offset, size};
}

std::pair<uint64_t, uint32_t> DataStore::lookup_or_alloc(const std::string& key,
                                                         size_t data_len,
                                                         size_t key_stride) {
  const std::string fk = fold_key(key);
  std::lock_guard<std::mutex> lock(offset_mu_);
  auto it = offset_map_.find(fk);
  if (it != offset_map_.end()) {
    uint32_t stored_size = it->second.second;
    uint32_t new_size = static_cast<uint32_t>(data_len);
    if (new_size != stored_size) {
      it->second.second = new_size;
      DBG("lookup_or_alloc HIT(size-mismatch key=%s): updating stored_size %u "
          "-> %u",
          key.c_str(), stored_size, new_size);
    }
    return it->second;
  }
  uint32_t size = static_cast<uint32_t>(data_len);
  uint64_t offset = alloc_new_locked(size);
  offset_map_[fk] = {offset, size};
  {
    std::string tail = key.size() > 16 ? key.substr(key.size() - 16) : key;
    DBG("lookup_or_alloc FIRST-ALLOC key_tail=%s off=%lu", tail.c_str(),
        (unsigned long)offset);
  }
  return {offset, size};
}

std::pair<uint64_t, uint32_t> DataStore::lookup_offset(
    const std::string& key) const {
  std::lock_guard<std::mutex> lock(offset_mu_);
  auto it = offset_map_.find(fold_key(key));
  if (it != offset_map_.end()) {
    return it->second;
  }
  return {0, 0};
}

std::vector<std::pair<uint64_t, uint32_t>> DataStore::batch_lookup_or_alloc(
    const std::vector<std::string>& keys, const std::vector<size_t>& data_lens,
    size_t key_stride) {
  std::vector<std::pair<uint64_t, uint32_t>> results;
  results.reserve(keys.size());

  std::lock_guard<std::mutex> lock(offset_mu_);

  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string fk = fold_key(keys[i]);
    auto it = offset_map_.find(fk);
    if (it != offset_map_.end()) {
      results.emplace_back(it->second.first, it->second.second);
    } else {
      uint32_t size = static_cast<uint32_t>(data_lens[i]);
      uint64_t offset = alloc_new_locked(size);
      offset_map_[fk] = {offset, size};
      results.emplace_back(offset, size);
    }
  }

  return results;
}

std::vector<std::pair<uint64_t, uint32_t>> DataStore::batch_lookup_or_allocate(
    const std::vector<std::string>& keys,
    const std::vector<size_t>& data_lens) {
  std::vector<std::pair<uint64_t, uint32_t>> results;
  results.reserve(keys.size());

  std::lock_guard<std::mutex> lock(offset_mu_);

  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string fk = fold_key(keys[i]);
    auto it = offset_map_.find(fk);
    if (it != offset_map_.end()) {
      results.emplace_back(it->second.first, it->second.second);
    } else {
      uint32_t size = static_cast<uint32_t>(data_lens[i]);
      uint64_t offset = alloc_new_locked(size);
      offset_map_[fk] = {offset, size};
      results.emplace_back(offset, size);
    }
  }

  return results;
}

void DataStore::inc_dirty_count() {
  std::lock_guard<std::mutex> lock(offset_mu_);
  dirty_count_++;
}

bool DataStore::remove_offset(const std::string& key) {
  std::lock_guard<std::mutex> lock(offset_mu_);
  auto it = offset_map_.find(fold_key(key));
  if (it == offset_map_.end()) {
    return false;
  }
  // Queue the key's chunks for physical release. They enter free_ranges_
  // only after removeChunks succeeds (release_pending_chunks, meta_saver
  // thread), so a chunk can never be physically removed after being handed
  // to a new key.
  const uint64_t cs = layout_chunk_size_ ? layout_chunk_size_ : chunk_size_;
  if (cs > 0 && it->second.second > 0) {
    uint64_t first = it->second.first / cs;
    uint64_t nchunks = (static_cast<uint64_t>(it->second.second) + cs - 1) / cs;
    for (uint64_t i = 0; i < nchunks; ++i) {
      pending_release_.push_back(first + i);
    }
  }
  offset_map_.erase(it);
  return true;
}

bool DataStore::contains(const std::string& key) const {
  std::lock_guard<std::mutex> lock(offset_mu_);
  auto found = offset_map_.find(fold_key(key)) != offset_map_.end();
  if (!found) {
    std::string tail = key.size() > 16 ? key.substr(key.size() - 16) : key;
    DBG("contains MISS key_tail=%s offset_map_size=%zu", tail.c_str(),
        offset_map_.size());
  }
  return found;
}

void DataStore::extend_file_size(uint64_t new_size) {
  std::lock_guard<std::mutex> lock(offset_mu_);
  if (new_size > current_file_size_) {
    current_file_size_ = new_size;
  }
}

void DataStore::sync_file_size(hf3fs::meta::client::MetaClient* mc,
                               const hf3fs::flat::UserInfo& ui, bool written) {
  if (!mc) return;

  // Snapshot the file size under the offset lock, then issue the blocking
  // MetaClient sync RPC OUTSIDE the lock. Holding offset_mu_ across the
  // blocking RPC serialized store (lookup_or_alloc) against all other ops
  // (including load's lookup_offset), which starved concurrent loads even
  // with lane-separated worker threads.
  uint64_t file_size = 0;
  {
    std::lock_guard<std::mutex> lock(offset_mu_);
    if (current_file_size_ == 0) return;
    file_size = current_file_size_;
  }

  hf3fs::meta::VersionedLength vlen{file_size, 0};
  auto syncRes = folly::coro::blockingWait(mc->sync(
      ui, cached_inode_id_, /*read=*/!written, /*written=*/written, vlen));
  if (syncRes.hasError()) {
    DBG("Failed to sync file size: %s", syncRes.error().describe().c_str());
  }
}

int DataStore::get_and_reset_dirty_count() {
  std::lock_guard<std::mutex> lock(offset_mu_);
  int count = dirty_count_;
  dirty_count_ = 0;
  return count;
}

void DataStore::sync_if_dirty(hf3fs::meta::client::MetaClient* mc,
                              const hf3fs::flat::UserInfo& ui, bool written,
                              int threshold) {
  if (!mc) return;
  uint64_t file_size = 0;
  {
    std::lock_guard<std::mutex> lock(offset_mu_);
    if (dirty_count_ < threshold) return;
    dirty_count_ = 0;
    file_size = current_file_size_;
  }
  if (file_size == 0) return;
  hf3fs::meta::VersionedLength vlen{file_size, 0};
  auto syncRes = folly::coro::blockingWait(mc->sync(
      ui, cached_inode_id_, /*read=*/!written, /*written=*/written, vlen));
  if (syncRes.hasError()) {
    DBG("Failed to sync file size: %s", syncRes.error().describe().c_str());
  }
}

uint32_t DataStore::get_chain_id_for_chunk(uint64_t chunk_index) const {
  if (chain_index_list_.empty()) return chain_id_;
  auto stripe = static_cast<size_t>(chunk_index % stripe_size_);
  uint32_t chainIndex = chain_index_list_[stripe];
  // Construct a ChainRef and look up the ChainId.
  hf3fs::flat::ChainRef ref(layout_.tableId, layout_.tableVersion,
                            static_cast<uint64_t>(chainIndex));
  auto chainIdOpt = routing_info_.getChainId(ref);
  if (chainIdOpt) {
    return chainIdOpt->toUnderType();
  }
  return chain_id_;
}

const std::unordered_map<uint32_t, uint32_t>& DataStore::get_chain_id_map() {
  // chain_id_map_ is built once in init_file() from the ordered chain
  // table (chainTables.chains vector).  The old lazy rebuild path used
  // routing_info_.getChainId(ref) which looks up chainTables — but
  // chainTables is empty in mgmtd routing info, so every chain fell back
  // to the same chain_id_.  Since init_file now populates the map
  // correctly on first call, we only rebuild if something cleared it.
  if (chain_id_map_.empty() && !chain_index_list_.empty()) {
    // Last resort: iterate routing_info_.chains (unordered — only for
    // single-node clusters where all chains share the same node).
    size_t i = 0;
    for (auto& [chainId, chainInfo] : routing_info_.chains) {
      if (i < chain_index_list_.size()) {
        chain_id_map_[chain_index_list_[i]] = chainId.toUnderType();
      }
      i++;
      if (i >= chain_index_list_.size()) break;
    }
  }
  return chain_id_map_;
}

// Physically remove queued chunks from the storage targets (removeChunks,
// one op per chunk following the fuse PUNCH_HOLE pattern) and move succeeded
// chunks into the free list. Runs on the meta_saver thread every cycle.
void DataStore::release_pending_chunks() {
  if (!storage_client_ || layout_chunk_size_ == 0) return;
  if (chain_index_list_.empty()) return;

  std::vector<uint64_t> chunks;
  {
    std::lock_guard<std::mutex> lock(offset_mu_);
    if (pending_release_.empty()) return;
    chunks.swap(pending_release_);
  }

  std::vector<hf3fs::storage::client::RemoveChunksOp> ops;
  ops.reserve(chunks.size());
  for (uint64_t idx : chunks) {
    uint32_t chainIdRaw = resolve_chain_for_chunk(idx);
    hf3fs::meta::ChunkId mcId(cached_inode_id_, 0, static_cast<uint32_t>(idx));
    hf3fs::storage::ChunkId cid(std::string_view(mcId.pack().data(), 16));
    ops.push_back(
        storage_client_->createRemoveOp(hf3fs::flat::ChainId(chainIdRaw), cid,
                                        hf3fs::storage::ChunkId(cid, 1)));
  }

  std::vector<uint64_t> succeeded;
  std::vector<uint64_t> failed;
  try {
    auto res = folly::coro::blockingWait(storage_client_->removeChunks(
        std::span<hf3fs::storage::client::RemoveChunksOp>(ops.data(),
                                                          ops.size()),
        user_info_));
    if (res.hasError()) {
      failed = std::move(chunks);
      DBG("release_pending_chunks: removeChunks failed: %s",
          std::string(res.error().message()).c_str());
    } else {
      for (size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].status().isOK()) {
          succeeded.push_back(chunks[i]);
        } else {
          failed.push_back(chunks[i]);
        }
      }
    }
  } catch (...) {
    failed = std::move(chunks);
  }

  std::lock_guard<std::mutex> lock(offset_mu_);
  merge_into_free_list_locked(std::move(succeeded));
  if (!failed.empty()) {
    pending_release_.insert(pending_release_.end(), failed.begin(),
                            failed.end());
  }
}

// offset_mu_ held. Merge sorted chunk indices into free_ranges_, coalescing
// index-adjacent ranges.
void DataStore::merge_into_free_list_locked(
    std::vector<uint64_t> chunk_indices) {
  if (chunk_indices.empty()) return;
  std::sort(chunk_indices.begin(), chunk_indices.end());
  size_t i = 0;
  while (i < chunk_indices.size()) {
    uint64_t start = chunk_indices[i];
    uint64_t len = 1;
    ++i;
    while (i < chunk_indices.size() && chunk_indices[i] == start + len) {
      ++len;
      ++i;
    }
    // Coalesce with the predecessor range.
    auto it = free_ranges_.lower_bound(start);
    if (it != free_ranges_.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second == start) {
        start = prev->first;
        len += prev->second;
        free_ranges_.erase(prev);
      }
    }
    // Coalesce with the successor range.
    it = free_ranges_.lower_bound(start);
    if (it != free_ranges_.end() && it->first == start + len) {
      len += it->second;
      free_ranges_.erase(it);
    }
    free_ranges_[start] = len;
  }
}

bool DataStore::lease_is_foreign_live(const LeaseData& ld) const {
  if (ld.id_hi == 0 && ld.id_lo == 0) return false;
  if (ld.id_hi == lease_id_hi_ && ld.id_lo == lease_id_lo_) return false;
  return (time(nullptr) - ld.heartbeat) < LEASE_TTL_SEC;
}

// Resolve the chain for a chunk index the same way the data path does
// (chain_id_map_ built from the ordered chain table; NOT
// get_chain_id_for_chunk, whose routing_info_.getChainId falls back to a
// single chain because chainTables is empty in mgmtd routing info).
uint32_t DataStore::resolve_chain_for_chunk(uint64_t chunk_idx) {
  if (chain_index_list_.empty()) return chain_id_;
  size_t stripe = stripe_size_ ? stripe_size_ : 1;
  uint32_t chainIndex = chain_index_list_[chunk_idx % stripe];
  const auto& cmap = get_chain_id_map();
  auto cit = cmap.find(chainIndex);
  return cit != cmap.end() ? cit->second : chain_id_;
}

bool DataStore::read_lease(LeaseData& out) {
  if (!storage_client_ || layout_chunk_size_ == 0) return false;
  if (chain_index_list_.empty()) return false;

  size_t stripe = stripe_size_ ? stripe_size_ : 1;
  uint32_t chainIndex = chain_index_list_[0 % stripe];
  uint32_t chainIdRaw = get_chain_id_raw();
  const auto& cmap = get_chain_id_map();
  auto cit = cmap.find(chainIndex);
  if (cit != cmap.end()) chainIdRaw = cit->second;

  hf3fs::meta::ChunkId mcId(cached_inode_id_, 0, 0);
  hf3fs::storage::ChunkId chunkId(std::string_view(mcId.pack().data(), 16));

  uint8_t* buf = meta_buf_alloc(LEASE_SIZE);
  if (!buf) return false;
  std::memset(buf, 0, LEASE_SIZE);
  auto reg = storage_client_->registerIOBuffer(buf, LEASE_SIZE);
  if (reg.hasError()) {
    std::free(buf);
    DBG("read_lease: registerIOBuffer failed: %s",
        reg.error().describe().c_str());
    return false;
  }
  hf3fs::storage::client::IOBuffer ioBuf(std::move(*reg));
  auto rio = storage_client_->createReadIO(
      hf3fs::flat::ChainId(chainIdRaw), chunkId, LEASE_OFFSET,
      static_cast<uint32_t>(LEASE_SIZE), buf, &ioBuf);
  hf3fs::flat::UserInfo ui = user_info_;
  hf3fs::storage::client::ReadOptions ro;
  ro.set_enableChecksum(false);
  auto res = folly::coro::blockingWait(storage_client_->batchRead(
      std::span<hf3fs::storage::client::ReadIO>(&rio, 1), ui, ro));
  if (res.hasError()) {
    std::free(buf);
    // Fresh file: chunk 0 does not exist yet — no lease.
    return false;
  }
  std::memcpy(&out, buf, LEASE_SIZE);
  std::free(buf);
  return true;
}

bool DataStore::write_lease(const LeaseData& in) {
  if (!storage_client_ || layout_chunk_size_ == 0) return false;
  if (chain_index_list_.empty()) return false;

  size_t stripe = stripe_size_ ? stripe_size_ : 1;
  uint32_t chainIndex = chain_index_list_[0 % stripe];
  uint32_t chainIdRaw = get_chain_id_raw();
  const auto& cmap = get_chain_id_map();
  auto cit = cmap.find(chainIndex);
  if (cit != cmap.end()) chainIdRaw = cit->second;

  hf3fs::meta::ChunkId mcId(cached_inode_id_, 0, 0);
  hf3fs::storage::ChunkId chunkId(std::string_view(mcId.pack().data(), 16));

  uint8_t* buf = meta_buf_alloc(LEASE_SIZE);
  if (!buf) return false;
  std::memcpy(buf, &in, LEASE_SIZE);
  auto reg = storage_client_->registerIOBuffer(buf, LEASE_SIZE);
  if (reg.hasError()) {
    std::free(buf);
    DBG("write_lease: registerIOBuffer failed: %s",
        reg.error().describe().c_str());
    return false;
  }
  hf3fs::storage::client::IOBuffer ioBuf(std::move(*reg));
  auto wio = storage_client_->createWriteIO(
      hf3fs::flat::ChainId(chainIdRaw), chunkId, LEASE_OFFSET,
      static_cast<uint32_t>(LEASE_SIZE), layout_chunk_size_, buf, &ioBuf);
  hf3fs::flat::UserInfo ui = user_info_;
  hf3fs::storage::client::WriteOptions wo;
  auto res = folly::coro::blockingWait(storage_client_->batchWrite(
      std::span<hf3fs::storage::client::WriteIO>(&wio, 1), ui, wo));
  if (res.hasError()) {
    std::free(buf);
    DBG("write_lease: batchWrite failed: %s", res.error().describe().c_str());
    return false;
  }
  std::free(buf);
  return true;
}

void DataStore::release_lease() {
  if (!lease_owned_) return;
  LeaseData zero{};
  if (write_lease(zero)) {
    lease_owned_ = false;
    DBG("release_lease: cleared owner lease on %s", file_path_.c_str());
  }
}

bool DataStore::lease_blocked_by_foreign() {
  LeaseData cur{};
  if (!read_lease(cur)) return false;
  return lease_is_foreign_live(cur);
}

// Persist the full offset_map snapshot to the meta region (chunk 0 + any
// extension chunks recorded in meta_chunk_list_). Full-snapshot semantics:
// write volume scales with total key count (design decision 2026-10-23 —
// no journal). Runs on the meta_saver thread.
void DataStore::save_meta() {
  if (!storage_client_ || chunk_size_ == 0) return;
  if (chain_index_list_.empty()) return;

  const size_t cs = layout_chunk_size_;
  const size_t chunk0_cap =
      cs > META_HEADER_SIZE ? (cs - META_HEADER_SIZE) / META_ENTRY_SIZE : 0;
  const size_t ext_cap = cs / META_ENTRY_SIZE;

  std::string entBlob;
  uint32_t chunk_size = 0;
  uint64_t next_offset = 0;
  uint64_t file_size = 0;
  {
    std::lock_guard<std::mutex> lock(offset_mu_);
    chunk_size = chunk_size_;
    next_offset = next_offset_;
    file_size = current_file_size_;
    entBlob.reserve(offset_map_.size() * META_ENTRY_SIZE);
    for (const auto& [key, val] : offset_map_) {
      size_t colon = key.find(':');
      if (colon == std::string::npos || colon == 0 || colon + 1 >= key.size()) {
        continue;  // key not in "high:low" form — cannot represent
      }
      uint64_t hi = std::strtoull(key.substr(0, colon).c_str(), nullptr, 0);
      uint64_t lo = std::strtoull(key.substr(colon + 1).c_str(), nullptr, 0);
      auto p32 = [&entBlob](uint32_t v) {
        entBlob.append(reinterpret_cast<const char*>(&v), 4);
      };
      auto p64 = [&entBlob](uint64_t v) {
        entBlob.append(reinterpret_cast<const char*>(&v), 8);
      };
      p64(hi);
      p64(lo);
      p64(val.first);
      p32(val.second);
    }
    size_t num = entBlob.size() / META_ENTRY_SIZE;
    size_t need_ext =
        num > chunk0_cap ? (num - chunk0_cap + ext_cap - 1) / ext_cap : 0;
    if (need_ext > META_LIST_MAX) {
      // Hard cap ~149M entries (META_LIST_MAX * ext_cap + chunk0_cap): far
      // beyond feasible memory, kept only as a format guard.
      entBlob.resize((chunk0_cap + META_LIST_MAX * ext_cap) * META_ENTRY_SIZE);
      fprintf(stderr,
              "[LMCache 3FS] WARNING: offset_map exceeds meta region hard "
              "cap; persisted snapshot truncated to %zu entries\n",
              chunk0_cap + META_LIST_MAX * ext_cap);
      need_ext = META_LIST_MAX;
    }
    while (meta_chunk_list_.size() < need_ext) {
      meta_chunk_list_.push_back(
          static_cast<uint32_t>(alloc_meta_chunk_locked()));
    }
  }

  size_t num = entBlob.size() / META_ENTRY_SIZE;
  size_t n0 = std::min(num, chunk0_cap);
  size_t k_ext = meta_chunk_list_.size();

  // Build the v4 chunk 0 header (exactly META_HEADER_SIZE bytes).
  std::string h;
  h.reserve(META_HEADER_SIZE);
  auto put32 = [&h](uint32_t v) {
    h.append(reinterpret_cast<const char*>(&v), 4);
  };
  auto put64 = [&h](uint64_t v) {
    h.append(reinterpret_cast<const char*>(&v), 8);
  };
  h.append(META_MAGIC, 4);
  h.push_back(static_cast<char>(META_VERSION));
  put32(chunk_size);
  put32(static_cast<uint32_t>(num));
  put64(next_offset);
  put64(file_size);
  put32(static_cast<uint32_t>(n0));
  put32(static_cast<uint32_t>(k_ext + 1));
  LeaseData lease{};
  // Write a live lease only while this instance owns the file; a zero lease
  // must never be resurrected by a late save (defense in depth on top of
  // the stop-before-release ordering in the connector).
  if (lease_owned_) {
    lease.id_hi = lease_id_hi_;
    lease.id_lo = lease_id_lo_;
    lease.heartbeat = time(nullptr);
    std::snprintf(lease.host, sizeof(lease.host), "%s", lease_host_.c_str());
  }
  h.append(reinterpret_cast<const char*>(&lease), LEASE_SIZE);
  for (size_t i = 0; i < META_LIST_MAX; ++i) {
    put32(i < k_ext ? meta_chunk_list_[i] : 0);
  }

  size_t total = (1 + k_ext) * cs;
  uint8_t* buf = meta_buf_alloc(total);
  if (!buf) {
    DBG("save_meta: buffer alloc failed (%zu bytes)", total);
    return;
  }
  std::memset(buf, 0, total);
  std::memcpy(buf, h.data(), h.size());
  std::memcpy(buf + META_HEADER_SIZE, entBlob.data(), n0 * META_ENTRY_SIZE);
  size_t done = n0;
  for (size_t k = 0; k < k_ext && done < num; ++k) {
    size_t take = std::min(num - done, ext_cap);
    std::memcpy(buf + (1 + k) * cs, entBlob.data() + done * META_ENTRY_SIZE,
                take * META_ENTRY_SIZE);
    done += take;
  }

  auto reg = storage_client_->registerIOBuffer(buf, total);
  if (reg.hasError()) {
    std::free(buf);
    DBG("save_meta: registerIOBuffer failed: %s",
        reg.error().describe().c_str());
    return;
  }
  hf3fs::storage::client::IOBuffer ioBuf(std::move(*reg));

  // One WriteIO per meta chunk, grouped by chain (mirrors the data path:
  // IOs share one registered buffer; batchWrite is issued per chain).
  std::map<uint32_t, std::vector<hf3fs::storage::client::WriteIO>> by_chain;
  {
    char basePacked[16];
    std::memcpy(basePacked,
                hf3fs::meta::ChunkId(cached_inode_id_, 0, 0).pack().data(), 16);
    auto add_io = [&](uint32_t chunk_idx, uint8_t* ptr) {
      uint32_t ci = chunk_idx;
      std::memcpy(basePacked + 12, &ci, 4);
      hf3fs::storage::ChunkId cid(std::string_view(basePacked, 16));
      uint32_t chain = resolve_chain_for_chunk(chunk_idx);
      by_chain[chain].push_back(storage_client_->createWriteIO(
          hf3fs::flat::ChainId(chain), cid, 0, static_cast<uint32_t>(cs),
          layout_chunk_size_, ptr, &ioBuf));
    };
    add_io(0, buf);
    for (size_t k = 0; k < k_ext; ++k) {
      add_io(meta_chunk_list_[k], buf + (1 + k) * cs);
    }
  }

  hf3fs::flat::UserInfo ui = user_info_;
  hf3fs::storage::client::WriteOptions wo;
  try {
    std::vector<folly::coro::Task<hf3fs::Result<hf3fs::Void>>> tasks;
    for (auto& [chain, wios] : by_chain) {
      tasks.push_back(storage_client_->batchWrite(
          std::span<hf3fs::storage::client::WriteIO>(wios.data(), wios.size()),
          ui, wo));
    }
    auto results = folly::coro::blockingWait(
        folly::coro::collectAllRange(std::move(tasks)));
    for (auto& r : results) {
      if (r.hasError()) {
        DBG("save_meta: batchWrite meta failed: %s",
            r.error().describe().c_str());
        break;
      }
    }
    DBG("save_meta: wrote %zu entries (%zu meta chunks, chunk0=%zu) to %s", num,
        k_ext + 1, n0, file_path_.c_str());
  } catch (...) {
    DBG("save_meta: batchWrite meta threw exception");
  }
  std::free(buf);
}

// ---------- Lazy meta saving ----------

void DataStore::mark_meta_dirty() {
  {
    std::lock_guard<std::mutex> lock(meta_save_mu_);
    meta_dirty_ = true;
  }
  meta_save_cv_.notify_one();
}

void DataStore::start_meta_saver() {
  if (meta_saver_thread_.joinable()) {
    return;
  }
  meta_save_stop_ = false;
  meta_saver_thread_ = std::thread(&DataStore::meta_saver_loop, this);
}

void DataStore::stop_meta_saver() {
  if (!meta_saver_thread_.joinable()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(meta_save_mu_);
    meta_save_stop_ = true;
  }
  meta_save_cv_.notify_one();
  meta_saver_thread_.join();
}

void DataStore::meta_saver_loop() {
  while (true) {
    bool should_save = false;
    bool stop = false;
    {
      std::unique_lock<std::mutex> lock(meta_save_mu_);
      meta_save_cv_.wait_for(lock, META_SAVE_INTERVAL,
                             [this] { return meta_dirty_ || meta_save_stop_; });
      stop = meta_save_stop_;
      should_save = meta_dirty_;
      meta_dirty_ = false;
    }
    // Refresh the owner lease heartbeat so a second instance refuses to
    // start while this one is idle (64B RDMA write per cycle).
    if (lease_owned_) {
      LeaseData mine{};
      mine.id_hi = lease_id_hi_;
      mine.id_lo = lease_id_lo_;
      mine.heartbeat = time(nullptr);
      std::snprintf(mine.host, sizeof(mine.host), "%s", lease_host_.c_str());
      if (!write_lease(mine)) {
        DBG("meta_saver_loop: lease heartbeat write failed");
      }
    }
    release_pending_chunks();
    // save_meta() does blocking 3FS RDMA/MetaClient storage I/O (not FUSE —
    // raw_hf3fs bypasses the FUSE mount entirely); never hold meta_save_mu_
    // across it, or concurrent mark_meta_dirty() (called on every store)
    // deadlocks on the same lock like the offset_mu_ variant did.
    if (should_save) save_meta();
    if (stop) return;
  }
}

// Read chunk 0, parse the v4 header, then read any extension meta chunks
// and rebuild the offset_map. Called once from init_file.
bool DataStore::load_meta() {
  if (!storage_client_ || chunk_size_ == 0) return false;
  if (chain_index_list_.empty()) return false;

  const size_t cs = layout_chunk_size_;
  const size_t chunk0_cap =
      cs > META_HEADER_SIZE ? (cs - META_HEADER_SIZE) / META_ENTRY_SIZE : 0;
  const size_t ext_cap = cs / META_ENTRY_SIZE;

  hf3fs::flat::UserInfo ui = user_info_;
  hf3fs::storage::client::ReadOptions ro;
  ro.set_enableChecksum(false);

  uint8_t* buf = meta_buf_alloc(cs);
  if (!buf) return false;
  std::memset(buf, 0, cs);
  auto reg = storage_client_->registerIOBuffer(buf, cs);
  if (reg.hasError()) {
    std::free(buf);
    return false;
  }
  hf3fs::storage::client::IOBuffer ioBuf(std::move(*reg));
  {
    hf3fs::meta::ChunkId mcId(cached_inode_id_, 0, 0);
    hf3fs::storage::ChunkId chunkId(std::string_view(mcId.pack().data(), 16));
    auto rio = storage_client_->createReadIO(
        hf3fs::flat::ChainId(resolve_chain_for_chunk(0)), chunkId, 0,
        static_cast<uint32_t>(cs), buf, &ioBuf);
    auto res = folly::coro::blockingWait(storage_client_->batchRead(
        std::span<hf3fs::storage::client::ReadIO>(&rio, 1), ui, ro));
    if (res.hasError()) {
      std::free(buf);
      // No meta written yet (chunk empty) — start fresh.
      return false;
    }
  }

  uint32_t total = 0, n0 = 0, nmeta = 0;
  uint64_t no = 0, fsz = 0;
  uint32_t file_cs = 0;
  std::vector<uint32_t> ext_list;
  bool parsed = false;
  {
    std::string_view blob(reinterpret_cast<const char*>(buf), cs);
    if (blob.size() >= META_HEADER_SIZE &&
        std::memcmp(blob.data(), META_MAGIC, 4) == 0 &&
        static_cast<uint8_t>(blob[4]) == META_VERSION) {
      size_t off = 5;
      auto read32 = [&](uint32_t& v) -> bool {
        if (off + 4 > blob.size()) return false;
        std::memcpy(&v, blob.data() + off, 4);
        off += 4;
        return true;
      };
      auto read64 = [&](uint64_t& v) -> bool {
        if (off + 8 > blob.size()) return false;
        std::memcpy(&v, blob.data() + off, 8);
        off += 8;
        return true;
      };
      if (read32(file_cs) && read32(total) && read64(no) && read64(fsz) &&
          read32(n0) && read32(nmeta)) {
        off = META_LIST_OFFSET;
        for (uint32_t i = 0; i + 1 < nmeta && i < META_LIST_MAX; ++i) {
          uint32_t idx = 0;
          if (off + 4 > blob.size()) break;
          std::memcpy(&idx, blob.data() + off, 4);
          off += 4;
          if (idx != 0) ext_list.push_back(idx);
        }
        parsed = true;
      }
    }
  }

  if (!parsed) {
    std::free(buf);
    DBG("load_meta: no valid meta in chunk 0 (fresh or legacy file)");
    return false;
  }
  if (file_cs != chunk_size_) {
    std::free(buf);
    DBG("load_meta: chunk_size mismatch %u != %u, discarding meta", file_cs,
        chunk_size_);
    return false;
  }

  // Entries from extension chunks: one batchRead per chain group, sharing a
  // single registered buffer.
  std::string entries;
  entries.reserve(static_cast<size_t>(total) * META_ENTRY_SIZE);
  entries.append(reinterpret_cast<const char*>(buf + META_HEADER_SIZE),
                 std::min<size_t>(n0, chunk0_cap) * META_ENTRY_SIZE);
  std::free(buf);

  if (!ext_list.empty()) {
    uint8_t* ebuf = meta_buf_alloc(ext_list.size() * cs);
    if (!ebuf) {
      DBG("load_meta: ext buffer alloc failed");
      return false;
    }
    std::memset(ebuf, 0, ext_list.size() * cs);
    auto ereg = storage_client_->registerIOBuffer(ebuf, ext_list.size() * cs);
    if (ereg.hasError()) {
      std::free(ebuf);
      DBG("load_meta: ext registerIOBuffer failed: %s",
          ereg.error().describe().c_str());
      return false;
    }
    hf3fs::storage::client::IOBuffer eioBuf(std::move(*ereg));
    std::map<uint32_t, std::vector<hf3fs::storage::client::ReadIO>> by_chain;
    char basePacked[16];
    std::memcpy(basePacked,
                hf3fs::meta::ChunkId(cached_inode_id_, 0, 0).pack().data(), 16);
    for (size_t k = 0; k < ext_list.size(); ++k) {
      uint32_t ci = ext_list[k];
      std::memcpy(basePacked + 12, &ci, 4);
      hf3fs::storage::ChunkId cid(std::string_view(basePacked, 16));
      by_chain[resolve_chain_for_chunk(ci)].push_back(
          storage_client_->createReadIO(
              hf3fs::flat::ChainId(resolve_chain_for_chunk(ci)), cid, 0,
              static_cast<uint32_t>(cs), ebuf + k * cs, &eioBuf));
    }
    bool ext_ok = true;
    try {
      std::vector<folly::coro::Task<hf3fs::Result<hf3fs::Void>>> tasks;
      for (auto& [chain, rios] : by_chain) {
        tasks.push_back(storage_client_->batchRead(
            std::span<hf3fs::storage::client::ReadIO>(rios.data(), rios.size()),
            ui, ro));
      }
      auto results = folly::coro::blockingWait(
          folly::coro::collectAllRange(std::move(tasks)));
      for (auto& r : results) {
        if (r.hasError()) {
          ext_ok = false;
          DBG("load_meta: ext batchRead failed: %s",
              r.error().describe().c_str());
          break;
        }
      }
    } catch (...) {
      ext_ok = false;
    }
    if (ext_ok) {
      size_t remaining = total > entries.size() / META_ENTRY_SIZE
                             ? total - entries.size() / META_ENTRY_SIZE
                             : 0;
      for (size_t k = 0; k < ext_list.size() && remaining > 0; ++k) {
        size_t take = std::min(remaining, ext_cap);
        entries.append(reinterpret_cast<const char*>(ebuf + k * cs),
                       take * META_ENTRY_SIZE);
        remaining -= take;
      }
    }
    std::free(ebuf);
  }

  {
    std::lock_guard<std::mutex> lock(offset_mu_);
    offset_map_.clear();
    next_offset_ = no;
    current_file_size_ = fsz;
    meta_chunk_list_ = ext_list;
    free_ranges_.clear();
    pending_release_.clear();
    size_t count = entries.size() / META_ENTRY_SIZE;
    for (size_t i = 0; i < count; ++i) {
      const char* e = entries.data() + i * META_ENTRY_SIZE;
      uint64_t hi = 0, lo = 0, ofs = 0;
      uint32_t sz = 0;
      std::memcpy(&hi, e, 8);
      std::memcpy(&lo, e + 8, 8);
      std::memcpy(&ofs, e + 16, 8);
      std::memcpy(&sz, e + 24, 4);
      char kb[48];
      int kn = std::snprintf(kb, sizeof(kb), "%llu:%llu",
                             (unsigned long long)hi, (unsigned long long)lo);
      offset_map_[std::string(kb, kn)] = {ofs, sz};
    }
    DBG("load_meta: loaded %zu keys (n0=%u ext=%zu) next_off=%lu", count, n0,
        ext_list.size(), (unsigned long)no);
  }
  return true;
}

void DataStore::clear_state() {
  std::lock_guard<std::mutex> lock(offset_mu_);
  offset_map_.clear();
  next_offset_ = 0;
  current_file_size_ = 0;
  dirty_count_ = 0;
  free_ranges_.clear();
  pending_release_.clear();
  meta_chunk_list_.clear();
  lease_owned_ = false;
  init_refused_ = false;
  // Reset layout to initial state (no inode, no chain mapping yet).
  cached_inode_id_ = hf3fs::meta::InodeId();
  chain_id_ = 0;
  chain_id_map_.clear();
  chain_index_list_.clear();
  layout_ = hf3fs::meta::Layout();
}

}  // namespace connector
}  // namespace lmcache

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include "data_store.h"

#include <atomic>

// DBG macro for runtime-controlled debug logging.
// Controlled by LMCACHE_DBG=1 env var (default off).
// get_dbg_flag() is defined in hf3fs_connector.h and set in hf3fs_connector.cc.
#include "hf3fs_connector.h"
#define DBG(fmt, ...) \
  do { if (get_dbg_flag()) fprintf(stderr, "[LMCache 3FS] " fmt "\n", ##__VA_ARGS__); } while(0)

#include <folly/executors/InlineExecutor.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
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

// Binary offset_map meta format persisted to chunk 0 of the .store file
// (RDMA, not FUSE). Keeps atomicity on the storage side without the FUSE mount.
constexpr char META_MAGIC[4] = {'L', 'M', '3', 'M'};
constexpr uint8_t META_VERSION = 3;
constexpr size_t META_HEADER_SIZE = 4 + 1 + 4 + 4 + 8 + 8 + 4;
constexpr size_t META_ENTRY_SIZE = 8 + 8 + 8 + 4;

// Page-aligned buffer for RDMA registration (mirrors alloc_rdma).
static uint8_t* meta_buf_alloc(size_t sz) {
  if (sz == 0) sz = 1;
  long ps = sysconf(_SC_PAGESIZE);
  size_t as = (static_cast<size_t>(sz) + ps - 1) / ps * ps;
  void* p = nullptr;
  if (posix_memalign(&p, static_cast<size_t>(ps), as) != 0) return nullptr;
  return static_cast<uint8_t*>(p);
}

void DataStore::init_file(
    hf3fs::meta::client::MetaClient* mc,
    hf3fs::storage::client::StorageClient* sc,
    const hf3fs::flat::UserInfo& ui,
    const hf3fs::flat::RoutingInfo& routing_info,
    uint32_t chunk_size,
    const std::string& base_dir,
    uint32_t num_files,
    uint32_t file_permission) {
  if (!mc) return;

  // Keep clients for RDMA-based offset_map persistence (save_meta/load_meta).
  storage_client_ = sc;
  user_info_ = ui;

  stride_ = INTERNAL_STRIDE;
  chunk_size_ = chunk_size;
  num_files_ = num_files > 0 ? num_files : 1;
  base_dir_ = base_dir.empty() ? "lmcache" : base_dir;
  DBG("init_file: file_index=%u chunk_size=%u layout_chunk_size_=%u (env override=%u) base_dir=%s",
      file_index_, chunk_size_, layout_chunk_size_, override_stripe_chunk_size(),
      base_dir_.c_str());

  // Build file path relative to 3FS root (MetaClient path resolution requires relative paths)
  file_path_ = base_dir_ + "/data_" + std::to_string(file_index_) + ".store";

  // Number of chains in routing_info determines stripe size.
  size_t num_chains = routing_info.chains.size();
  stripe_size_ = static_cast<uint32_t>(num_chains);
  DBG("init_file: num_chains=%zu, stripe_size=%u, chainTables.size=%zu",
      num_chains, stripe_size_, routing_info.chainTables.size());
  // Build ClusterRange layout with all chains for this file.
  // Each file stripes across ALL nodes, enabling full parallel bandwidth.
  // Use layout_chunk_size_ (capped for large keys) so big keys span
  // multiple chunks across chains for parallel reads.
  layout_ = hf3fs::meta::Layout::newChainRange(
      hf3fs::flat::ChainTableId(chain_table_id_),
      hf3fs::flat::ChainTableVersion(routing_info.routingInfoVersion),
      layout_chunk_size_,
      stripe_size_,
      0);

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
  DBG("init_file: open %s done, hasError=%d", file_path_.c_str(), (int)openRes.hasError());

  hf3fs::meta::Inode fileInode;
  if (openRes.hasError()) {
    // File doesn't exist, create it.
    DBG("init_file: try create %s", file_path_.c_str());
    auto createRes = folly::coro::blockingWait(
        mc->create(ui, hf3fs::meta::InodeId::root(), path, session,
                   hf3fs::flat::Permission(file_permission),
                   static_cast<int>(hf3fs::meta::OpenFlags(O_RDWR | O_CREAT)),
                   std::nullopt));
    DBG("init_file: create %s done, hasError=%d", file_path_.c_str(), (int)createRes.hasError());
    if (createRes.hasError()) {
      DBG("Failed to create data store %s: %s",
          file_path_.c_str(),
          createRes.error().describe().c_str());
      return;
    }
    fileInode = *createRes;

    // Sync the newly created file to persist it.
    hf3fs::meta::VersionedLength vlen{0, 0};
    auto syncRes = folly::coro::blockingWait(
        mc->sync(ui, fileInode.id, /*read=*/ false, /*written=*/ false, vlen));
    if (syncRes.hasError()) {
      DBG("Failed to sync new file %s: %s",
          file_path_.c_str(),
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
  
  // Load existing offset_map from chunk 0 (via RDMA) if any.
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

void DataStore::refresh_routing_info(const hf3fs::flat::RoutingInfo& new_routing_info) {
  // Only update if routing version actually changed.
  if (routing_info_.routingInfoVersion == new_routing_info.routingInfoVersion) {
    return;
  }

  DBG("refresh_routing_info: %s version %u -> %u",
      file_path_.c_str(),
      routing_info_.routingInfoVersion.toUnderType(),
      new_routing_info.routingInfoVersion.toUnderType());

  // Update stripe size and layout with new routing info.
  stripe_size_ = static_cast<uint32_t>(new_routing_info.chains.size());
  layout_ = hf3fs::meta::Layout::newChainRange(
      hf3fs::flat::ChainTableId(chain_table_id_),
      hf3fs::flat::ChainTableVersion(new_routing_info.routingInfoVersion),
      layout_chunk_size_,
      stripe_size_,
      0);

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

std::pair<uint64_t, uint32_t> DataStore::get_or_allocate_offset(
    const std::string& key, size_t data_len, size_t key_stride) {
  std::lock_guard<std::mutex> lock(offset_mu_);

  auto it = offset_map_.find(key);
  if (it != offset_map_.end()) {
    return {it->second.first, it->second.second};
  }

  // Allocate new offset: each key gets a region aligned to chunk_size_.
  // CRITICAL: 3FS file stride must match RDMA buffer stride (chunk_size_)
  // so that buf[i*chunk_size_ + c*cs] maps to file offset offset+c*cs exactly.
  uint64_t offset = next_offset_;
  uint32_t size = static_cast<uint32_t>(data_len);
  uint64_t allocatedRegion =
      ((static_cast<uint64_t>(size) + chunk_size_ - 1) / chunk_size_) * chunk_size_;

  offset_map_[key] = {offset, size};
  next_offset_ += allocatedRegion;

  if (next_offset_ > current_file_size_) {
    current_file_size_ = next_offset_;
  }

  return {offset, size};
}

std::pair<uint64_t, uint32_t> DataStore::lookup_or_alloc(
    const std::string& key, size_t data_len, size_t key_stride) {
  std::lock_guard<std::mutex> lock(offset_mu_);
  auto it = offset_map_.find(key);
  if (it != offset_map_.end()) {
    uint32_t stored_size = it->second.second;
    uint32_t new_size = static_cast<uint32_t>(data_len);
    if (new_size != stored_size) {
      it->second.second = new_size;
      DBG("lookup_or_alloc HIT(size-mismatch key=%s): updating stored_size %u -> %u", key.c_str(), stored_size, new_size);
    }
    return it->second;
  }
  uint64_t offset = next_offset_;
  uint32_t size = static_cast<uint32_t>(data_len);
  uint64_t allocatedRegion =
      ((static_cast<uint64_t>(size) + chunk_size_ - 1) / chunk_size_) * chunk_size_;
  offset_map_[key] = {offset, size};
  next_offset_ += allocatedRegion;
  if (next_offset_ > current_file_size_) {
    current_file_size_ = next_offset_;
  }
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
  auto it = offset_map_.find(key);
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
    auto it = offset_map_.find(keys[i]);
    if (it != offset_map_.end()) {
      results.emplace_back(it->second.first, it->second.second);
    } else {
      uint64_t offset = next_offset_;
      uint32_t size = static_cast<uint32_t>(data_lens[i]);
      uint64_t allocatedRegion =
          ((static_cast<uint64_t>(size) + chunk_size_ - 1) / chunk_size_) * chunk_size_;
      offset_map_[keys[i]] = {offset, size};
      next_offset_ += allocatedRegion;
      if (next_offset_ > current_file_size_) {
        current_file_size_ = next_offset_;
      }
      results.emplace_back(offset, size);
    }
  }

  return results;
}

std::vector<std::pair<uint64_t, uint32_t>> DataStore::batch_lookup_or_allocate(
    const std::vector<std::string>& keys, const std::vector<size_t>& data_lens) {
  std::vector<std::pair<uint64_t, uint32_t>> results;
  results.reserve(keys.size());

  std::lock_guard<std::mutex> lock(offset_mu_);

  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = offset_map_.find(keys[i]);
    if (it != offset_map_.end()) {
      results.emplace_back(it->second.first, it->second.second);
    } else {
      // Allocate new offset: each key gets a chunk-aligned region.
      uint64_t offset = next_offset_;
      uint32_t size = static_cast<uint32_t>(data_lens[i]);
      uint64_t allocatedRegion =
          ((static_cast<uint64_t>(size) + chunk_size_ - 1) / chunk_size_) * chunk_size_;

      offset_map_[keys[i]] = {offset, size};
      next_offset_ += allocatedRegion;

      if (next_offset_ > current_file_size_) {
        current_file_size_ = next_offset_;
      }

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
  auto it = offset_map_.find(key);
  if (it != offset_map_.end()) {
    offset_map_.erase(it);
    return true;
  }
  return false;
}

bool DataStore::contains(const std::string& key) const {
  std::lock_guard<std::mutex> lock(offset_mu_);
  auto found = offset_map_.find(key) != offset_map_.end();
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

std::string DataStore::get_meta_chunk_id(uint64_t offset) const {
  // Compute chunk index from file offset using chunk_size (3FS chunk mapping).
  // stride_ controls buffer layout (1MB), chunk_size_ controls 3FS chunk_id.
  uint64_t chunk_index = offset / chunk_size_;

  // Construct ChunkId directly from inode_id and chunk_index.
  hf3fs::meta::ChunkId chunkId(cached_inode_id_, 0,
                                static_cast<uint32_t>(chunk_index));
  return chunkId.pack();
}

void DataStore::sync_file_size(hf3fs::meta::client::MetaClient* mc,
                               const hf3fs::flat::UserInfo& ui,
                               bool written) {
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
  auto syncRes = folly::coro::blockingWait(
      mc->sync(ui, cached_inode_id_, /*read=*/ !written, /*written=*/ written,
               vlen));
  if (syncRes.hasError()) {
      DBG("Failed to sync file size: %s",
          syncRes.error().describe().c_str());
  }
}

int DataStore::get_and_reset_dirty_count() {
  std::lock_guard<std::mutex> lock(offset_mu_);
  int count = dirty_count_;
  dirty_count_ = 0;
  return count;
}

void DataStore::sync_if_dirty(hf3fs::meta::client::MetaClient* mc,
                              const hf3fs::flat::UserInfo& ui,
                              bool written,
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
  auto syncRes = folly::coro::blockingWait(
      mc->sync(ui, cached_inode_id_, /*read=*/ !written, /*written=*/ written,
               vlen));
  if (syncRes.hasError()) {
    DBG("Failed to sync file size: %s",
        syncRes.error().describe().c_str());
  }
}

uint32_t DataStore::get_chain_id_for_chunk(uint64_t chunk_index) const {
  if (chain_index_list_.empty()) return chain_id_;
  auto stripe = static_cast<size_t>(chunk_index % stripe_size_);
  uint32_t chainIndex = chain_index_list_[stripe];
  // Construct a ChainRef and look up the ChainId.
  hf3fs::flat::ChainRef ref(
      layout_.tableId, layout_.tableVersion, static_cast<uint64_t>(chainIndex));
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

void DataStore::save_meta() {
  if (!storage_client_ || chunk_size_ == 0) return;
  if (chain_index_list_.empty()) return;

  // Snapshot under lock, do RDMA I/O outside it: never hold offset_mu_ across
  // blocking I/O, which previously deadlocked store workers in sync_if_dirty.
  std::unordered_map<std::string, std::pair<uint64_t, uint32_t>> snapshot;
  uint32_t stride = 0;
  uint32_t chunk_size = 0;
  uint64_t next_offset = 0;
  uint64_t file_size = 0;
  {
    std::lock_guard<std::mutex> lock(offset_mu_);
    stride = stride_;
    chunk_size = chunk_size_;
    next_offset = next_offset_;
    file_size = current_file_size_;
    snapshot = offset_map_;
  }

  // Serialize offset_map to the binary meta blob (33B header + 28B entries).
  std::string entBlob;
  entBlob.reserve(snapshot.size() * META_ENTRY_SIZE);
  for (const auto& [key, val] : snapshot) {
    size_t colon = key.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= key.size()) {
      continue;  // key not in "high:low" form — cannot represent in binary meta
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

  // Cap to the meta region (chunk 0 = one layout chunk). Exceeding only
  // degrades cold-restart replay; the in-memory map stays fully intact.
  size_t max_entries =
      layout_chunk_size_ > META_HEADER_SIZE
          ? (layout_chunk_size_ - META_HEADER_SIZE) / META_ENTRY_SIZE
          : 0;
  if (num > max_entries) {
    entBlob.resize(max_entries * META_ENTRY_SIZE);
    num = max_entries;
    DBG("save_meta: truncating to %zu entries (meta > %u bytes)", num, chunk_size);
  }

  std::string blob;
  blob.reserve(META_HEADER_SIZE + entBlob.size());
  blob.append(META_MAGIC, 4);
  blob.push_back(static_cast<char>(META_VERSION));
  auto put32 = [&blob](uint32_t v) {
    blob.append(reinterpret_cast<const char*>(&v), 4);
  };
  auto put64 = [&blob](uint64_t v) {
    blob.append(reinterpret_cast<const char*>(&v), 8);
  };
  put32(chunk_size);
  put32(stride);
  put64(next_offset);
  put64(file_size);
  put32(static_cast<uint32_t>(num));
  blob += entBlob;

  // Write the meta blob to chunk 0 of the .store file via StorageClient RDMA.
  size_t stripe = stripe_size_ ? stripe_size_ : 1;
  uint32_t chainIndex = chain_index_list_[0 % stripe];
  uint32_t chainIdRaw = get_chain_id_raw();
  const auto& cmap = get_chain_id_map();
  auto cit = cmap.find(chainIndex);
  if (cit != cmap.end()) chainIdRaw = cit->second;

  hf3fs::meta::ChunkId mcId(cached_inode_id_, 0, 0);
  hf3fs::storage::ChunkId chunkId(std::string_view(mcId.pack().data(), 16));

  uint8_t* buf = meta_buf_alloc(blob.size());
  if (!buf) {
    DBG("save_meta: buffer alloc failed");
    return;
  }
  std::memcpy(buf, blob.data(), blob.size());
  auto reg = storage_client_->registerIOBuffer(buf, blob.size());
  if (reg.hasError()) {
    std::free(buf);
    DBG("save_meta: registerIOBuffer failed: %s", reg.error().describe().c_str());
    return;
  }
  hf3fs::storage::client::IOBuffer ioBuf(std::move(*reg));
  auto wio = storage_client_->createWriteIO(hf3fs::flat::ChainId(chainIdRaw), chunkId,
                                            0, static_cast<uint32_t>(blob.size()),
                                            layout_chunk_size_, buf, &ioBuf);
  hf3fs::flat::UserInfo ui = user_info_;
  hf3fs::storage::client::WriteOptions wo;
  auto res = folly::coro::blockingWait(storage_client_->batchWrite(
      std::span<hf3fs::storage::client::WriteIO>(&wio, 1), ui, wo));
  if (res.hasError()) {
    DBG("save_meta: batchWrite meta failed: %s", res.error().describe().c_str());
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
      meta_save_cv_.wait_for(lock, META_SAVE_INTERVAL, [this] {
        return meta_dirty_ || meta_save_stop_;
      });
      stop = meta_save_stop_;
      should_save = meta_dirty_;
      meta_dirty_ = false;
    }
    // save_meta() does blocking FUSE-backed I/O; never hold meta_save_mu_
    // across it, or concurrent mark_meta_dirty() (called on every store)
    // deadlocks on the same lock like the offset_mu_ variant did.
    if (should_save) save_meta();
    if (stop) return;
  }
}

bool DataStore::load_meta() {
  if (!storage_client_ || chunk_size_ == 0) return false;
  if (chain_index_list_.empty()) return false;

  // Read the whole meta region (chunk 0) via StorageClient RDMA.
  size_t stripe = stripe_size_ ? stripe_size_ : 1;
  uint32_t chainIndex = chain_index_list_[0 % stripe];
  uint32_t chainIdRaw = get_chain_id_raw();
  const auto& cmap = get_chain_id_map();
  auto cit = cmap.find(chainIndex);
  if (cit != cmap.end()) chainIdRaw = cit->second;

  hf3fs::meta::ChunkId mcId(cached_inode_id_, 0, 0);
  hf3fs::storage::ChunkId chunkId(std::string_view(mcId.pack().data(), 16));

  size_t reg_len = layout_chunk_size_;
  uint8_t* buf = meta_buf_alloc(reg_len);
  if (!buf) return false;
  std::memset(buf, 0, reg_len);
  auto reg = storage_client_->registerIOBuffer(buf, reg_len);
  if (reg.hasError()) {
    std::free(buf);
    return false;
  }
  hf3fs::storage::client::IOBuffer ioBuf(std::move(*reg));
  auto rio = storage_client_->createReadIO(hf3fs::flat::ChainId(chainIdRaw), chunkId,
                                           0, static_cast<uint32_t>(reg_len), buf, &ioBuf);
  hf3fs::flat::UserInfo ui = user_info_;
  hf3fs::storage::client::ReadOptions ro;
  ro.set_enableChecksum(false);
  auto res = folly::coro::blockingWait(storage_client_->batchRead(
      std::span<hf3fs::storage::client::ReadIO>(&rio, 1), ui, ro));
  if (res.hasError()) {
    std::free(buf);
    // No meta written yet (chunk empty) — start fresh.
    return false;
  }

  // Parse the binary header (valid only if magic + version + chunk_size match).
  std::string_view blob(reinterpret_cast<const char*>(buf), reg_len);
  bool parsed = false;
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
    uint32_t cs = 0, stride = 0, num = 0;
    uint64_t no = 0, fsz = 0;
    if (read32(cs) && read32(stride) && read64(no) && read64(fsz) && read32(num)) {
      if (cs != chunk_size_) {
        DBG("load_meta: chunk_size mismatch %u != %u, discarding meta", cs, chunk_size_);
      } else {
        std::lock_guard<std::mutex> lock(offset_mu_);
        offset_map_.clear();
        next_offset_ = no;
        current_file_size_ = fsz;
        for (uint32_t i = 0; i < num; ++i) {
          if (off + META_ENTRY_SIZE > blob.size()) break;
          uint64_t hi = 0, lo = 0, ofs = 0;
          uint32_t sz = 0;
          std::memcpy(&hi, blob.data() + off, 8); off += 8;
          std::memcpy(&lo, blob.data() + off, 8); off += 8;
          std::memcpy(&ofs, blob.data() + off, 8); off += 8;
          std::memcpy(&sz, blob.data() + off, 4); off += 4;
          char kb[48];
          int kn = std::snprintf(kb, sizeof(kb), "%llu:%llu",
                                 (unsigned long long)hi, (unsigned long long)lo);
          offset_map_[std::string(kb, kn)] = {ofs, sz};
        }
        parsed = true;
      }
    }
  }
  std::free(buf);

  if (parsed) {
    DBG("load_meta: loaded %zu keys from chunk 0", offset_map_.size());
  }
  return parsed;
}

std::string DataStore::get_meta_path() const {
  return meta_path_;
}

void DataStore::clear_state() {
  std::lock_guard<std::mutex> lock(offset_mu_);
  offset_map_.clear();
  next_offset_ = 0;
  current_file_size_ = 0;
  dirty_count_ = 0;
  // Reset layout to initial state (no inode, no chain mapping yet).
  cached_inode_id_ = hf3fs::meta::InodeId();
  chain_id_ = 0;
  chain_id_map_.clear();
  chain_index_list_.clear();
  layout_ = hf3fs::meta::Layout();
}

}  // namespace connector
}  // namespace lmcache

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.
#pragma once

#include <Python.h>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Self-contained: connector_base.h is a local fork vendored into this dir so
// the raw backend's batch-semantics changes do not touch / overwrite the
// shared csrc/storage_backends/ version. connector_interface.h /
// connector_types.h / event_notifier.h are NOT forked - they are included
// via ../ from the shared include dir (byte-identical to upstream).
#include "connector_base.h"
#include "../connector_types.h"
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

// Minimal 3FS forward declarations — heavy includes go in .cc file.
// We use void* for client pointers to avoid pulling in full headers here.
// IOBuffer must be forward-declared here (unique_ptr requires at least a
// forward declaration), and fully included in the .cc file where the
// destructor needs the complete type.
namespace hf3fs {
namespace net {
class Client;
}
}  // namespace hf3fs
namespace hf3fs {
namespace client {
class MgmtdClientForClient;
class RoutingInfo;
}  // namespace client
}  // namespace hf3fs
namespace hf3fs {
namespace meta {
namespace client {
class MetaClient;
}
}  // namespace meta
}  // namespace hf3fs
namespace hf3fs {
namespace storage {
namespace client {
class IOBuffer;
class StorageClient;
}  // namespace client
}  // namespace storage
}  // namespace hf3fs
namespace hf3fs {
namespace flat {
struct ServiceGroupInfo;
}
}  // namespace hf3fs

// Include MetaClient header to access MetaClient::Config type
#include "client/meta/MetaClient.h"
#include "common/net/Client.h"

// Forward declaration for CPUExecutorGroup
namespace hf3fs {
class CPUExecutorGroup;
}

// ── DBG macro ──────────────────────────────────────────────────────────────
// Runtime-controlled debug logging via atomic flag.
// Controlled by LMCACHE_DBG=1 env var (default off).
// Single-file + offset data store
#include "data_store.h"

// ── DBG macro ──────────────────────────────────────────────────────────────
// Runtime-controlled debug logging via atomic flag.
// Controlled by LMCACHE_DBG=1 env var (default off).
// Defined after data_store.h include so that DataStore class is available.
#include <atomic>
#define DBG(fmt, ...)                                            \
  do {                                                           \
    if (get_dbg_flag())                                          \
      fprintf(stderr, "[LMCache 3FS] " fmt "\n", ##__VA_ARGS__); \
  } while (0)

namespace lmcache {
namespace connector {

inline std::atomic<bool>& get_dbg_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}

// Type-erased owning pointer for an hf3fs::net::Client::Config.
// Defined in the .cc file where the full type is available. The deleter
// calls the Config's destructor through the correct dynamic type.
using NetConfigPtr = std::unique_ptr<void, void (*)(void*)>;

// No-op deleter used when no Config is owned (default-constructed state).
inline void noop_net_config_deleter(void*) {}

// Hot-path timing stats — atomic counters aggregated across all workers.
// Captures the breakdown of do_batch_set / do_batch_get phases.
// All counters are nanoseconds; `count` is the number of batch operations.
struct Hf3fsTimingStats {
  std::atomic<uint64_t> ns_lookup{0};  // offset allocation / lookup_or_alloc
  std::atomic<uint64_t> ns_buf_alloc{
      0};  // total buffer phase (sum of sub-phases below)
  // Sub-phase breakdown of ns_buf_alloc:
  std::atomic<uint64_t> ns_buf_compute{0};     // computing total buffer size
  std::atomic<uint64_t> ns_buf_lock_check{0};  // 1st lock + cache size check
  std::atomic<uint64_t> ns_buf_lock_get{
      0};  // 2nd lock + buffer pointer fetch (cache hit)
  std::atomic<uint64_t> ns_buf_alloc_rdma{0};  // alloc_rdma (cache miss only)
  std::atomic<uint64_t> ns_buf_register{
      0};  // registerIOBuffer (cache miss only)
  std::atomic<uint64_t> ns_buf_cache_store{
      0};  // caching new buffer + 3rd lock (cache miss only)
  std::atomic<uint64_t> ns_buf_cache_hits{
      0};  // number of cache hits (for hit-rate)
  std::atomic<uint64_t> ns_buf_cache_misses{0};  // number of cache misses
  std::atomic<uint64_t> ns_memcpy{
      0};  // memcpy from Python buffers → local buffer
  std::atomic<uint64_t> ns_create_io{0};  // WriteIO / ReadIO construction
  std::atomic<uint64_t> ns_batch_io{0};   // actual batchWrite / batchRead RPC
  std::atomic<uint64_t> ns_sync{0};       // sync_file_size (write path only)
  std::atomic<uint64_t> ns_total{0};      // total wall time per batch
  std::atomic<uint64_t> count{0};         // number of batches observed
};

struct Hf3fsConnection {
  // Owns the hf3fs::net::Client::Config used to construct net_client.
  // The 3FS net::Client stores a const reference to its Config, so the
  // Config object MUST outlive the Client. Storing it here ensures that
  // because the connection's destructor releases the Client before
  // resetting this pointer. Declared before chunk_size to match the
  // initialization order in the constructor (avoid -Wreorder).
  NetConfigPtr net_config;
  void* net_client;      // hf3fs::net::Client*
  void* mgmtd_client;    // hf3fs::client::MgmtdClientForClient*
  void* meta_client;     // hf3fs::meta::client::MetaClient*
  void* storage_client;  // hf3fs::storage::client::StorageClient*
  std::shared_ptr<hf3fs::storage::client::StorageClient> storage_client_;
  std::shared_ptr<void>
      storage_client_keepalive;  // holds the StorageClient shared_ptr
                                 // with noop deleter so the
                                 // StorageClient is NOT destroyed
                                 // when the connection is destroyed
                                 // (it has background threads unsafe
                                 // to shut down from worker threads).
  uint8_t* user_info_buf;        // raw bytes for hf3fs::flat::UserInfo
  uint8_t session_id[16];        // raw bytes for hf3fs::Uuid (SessionId)
  uint32_t chunk_size;
  void* routing_info;  // cached routing info from mgmtd (const
                       // flat::RoutingInfo*)

  // Cache: key -> 3FS path (string)
  // Used for delete operations and to track which files exist.
  std::unordered_map<std::string, std::string> file_cache;

  // Cache: key -> shared_ptr to FileEntry (opaque)
  // Stores chainId and storageChunk computed during SET, so GET can
  // avoid calling MetaClient on existing files (which hangs).
  std::unordered_map<std::string, std::shared_ptr<void>> file_entries;

  // IOBuffer holders — MUST outlive the corresponding ReadIO/WriteIO objects.
  // StorageClient::batchRead/batchWrite do NOT take ownership of IOBuffer
  // objects; the I/O executes asynchronously in worker threads. Keeping the
  // IOBuffers alive in the connection struct (mirroring PioV's design where
  // rios_/wios_ are class members) prevents use-after-free when the async
  // worker accesses them after this function returns.
  //
  // Stored as void* with cleanup performed in the .cc file (where the full
  // IOBuffer type is available for proper destruction via the deleter
  // function defined there).
  std::vector<void*> io_buffers_;

  // IOBuffer pool for reuse — avoids repeated RDMA memory registration.
  // Cached IOBuffers are stored here after batch operations complete.
  // On next batch, we check the pool before allocating a new one.
  // Protected by io_buf_pool_mu_.
  std::mutex io_buf_pool_mu_;
  std::vector<void*> io_buf_pool_;

  // Read buffer cache — reused across batch_get calls when size suffices.
  uint8_t* rd_buf_cache_ = nullptr;
  size_t rd_buf_cache_size_ = 0;
  void* rd_io_buf_cache_ = nullptr;

  // Write buffer cache — reused across batch_set calls when size suffices.
  // Avoids repeated alloc_rdma + registerIOBuffer (the dominant per-batch
  // overhead). Protected by io_buf_pool_mu_.
  uint8_t* wr_buf_cache_ = nullptr;
  size_t wr_buf_cache_size_ = 0;
  void* wr_io_buf_cache_ = nullptr;

  // Per-connection timing stats (aggregated by
  // Hf3fsConnector::print_timing_stats).
  Hf3fsTimingStats set_timing_;     // write-path timing
  Hf3fsTimingStats get_timing_;     // read-path timing
  Hf3fsTimingStats exists_timing_;  // exists-path timing (contains lookup)

  // Deleter for IOBuffer objects stored as void*.
  // Defined in hf3fs_connector.cc where hf3fs::storage::client::IOBuffer
  // is a complete type.
  static void delete_io_buffer(void* p);

  Hf3fsConnection()
      : net_config(nullptr, &noop_net_config_deleter),
        chunk_size(32 * 1024 * 1024),  // 32 MB, default chunk size
        routing_info(nullptr) {
    net_client = nullptr;
    mgmtd_client = nullptr;
    meta_client = nullptr;
    storage_client = nullptr;
    storage_client_keepalive = nullptr;
    user_info_buf = nullptr;
  }

  // Non-copyable: owns raw 3FS resources and a unique_ptr.
  Hf3fsConnection(const Hf3fsConnection&) = delete;
  Hf3fsConnection& operator=(const Hf3fsConnection&) = delete;

  // Movable: the user-defined destructor suppresses the implicit move
  // constructor, so we explicitly define it. This is required because
  // create_connection() returns by value. We must nullify the source's
  // raw pointers after the move — otherwise the source's destructor would
  // free the 3FS clients while the destination still holds dangling
  // pointers, causing "pure virtual method called" / SIGABRT.
  Hf3fsConnection(Hf3fsConnection&& other) noexcept
      : net_config(std::move(other.net_config)),
        storage_client_keepalive(std::move(other.storage_client_keepalive)),
        storage_client_(std::move(other.storage_client_)),
        net_client(other.net_client),
        mgmtd_client(other.mgmtd_client),
        meta_client(other.meta_client),
        storage_client(other.storage_client),
        chunk_size(other.chunk_size),
        routing_info(other.routing_info),
        file_cache(std::move(other.file_cache)),
        file_entries(std::move(other.file_entries)),
        io_buffers_(std::move(other.io_buffers_)) {
    std::memcpy(session_id, other.session_id, sizeof(session_id));
    if (other.user_info_buf) {
      user_info_buf = new uint8_t[256];
      std::memcpy(user_info_buf, other.user_info_buf, 256);
    } else {
      user_info_buf = nullptr;
    }
    other.net_client = nullptr;
    other.mgmtd_client = nullptr;
    other.meta_client = nullptr;
    other.storage_client = nullptr;
    other.storage_client_keepalive.reset();
    other.storage_client_ = nullptr;
    other.file_cache.clear();
    other.file_entries.clear();
    other.user_info_buf = nullptr;
    other.chunk_size = 0;
  }
  Hf3fsConnection& operator=(Hf3fsConnection&& other) noexcept {
    if (this != &other) {
      // Destroy existing resources first.
      net_config.reset();
      delete[] user_info_buf;
      user_info_buf = nullptr;

      // Move-assign new resources.
      net_config = std::move(other.net_config);
      storage_client_keepalive = other.storage_client_keepalive;
      net_client = other.net_client;
      mgmtd_client = other.mgmtd_client;
      meta_client = other.meta_client;
      storage_client = other.storage_client;
      chunk_size = other.chunk_size;
      routing_info = other.routing_info;
      file_cache = std::move(other.file_cache);
      file_entries = std::move(other.file_entries);
      io_buffers_ = std::move(other.io_buffers_);

      std::memcpy(session_id, other.session_id, sizeof(session_id));
      if (other.user_info_buf) {
        user_info_buf = new uint8_t[256];
        std::memcpy(user_info_buf, other.user_info_buf, 256);
      } else {
        user_info_buf = nullptr;
      }
      other.net_client = nullptr;
      other.mgmtd_client = nullptr;
      other.meta_client = nullptr;
      other.storage_client = nullptr;
      other.storage_client_keepalive = nullptr;
      other.routing_info = nullptr;
      other.file_cache.clear();
      other.file_entries.clear();
      other.io_buffers_.clear();
      other.user_info_buf = nullptr;
      other.chunk_size = 0;
    }
    return *this;
  }

  // Destructor: release 3FS resources in the correct order.
  // The destructor is defined in the .cc file where the full client
  // types are available, so we can properly delete them.
  // NOTE: 3FS clients are intentionally NOT destroyed in the destructor
  // to avoid use-after-free from worker threads. See .cc for details.
  ~Hf3fsConnection();
};

/**
 * 3FS-backed connector.
 *
 * Implements the ConnectorBase interface using 3FS as the storage backend.
 * Keys are stored as files under <base_dir>/<key> in 3FS (base_dir
 * configurable, default "lmcache").
 *
 * The connector manages its own 3FS clients (net::Client, MgmtdClientForClient,
 * MetaClient, StorageClient) and handles the lifecycle of these clients,
 * including RDMA connection setup, session management, and routing info
 * refresh.
 */
class Hf3fsConnector : public ConnectorBase<Hf3fsConnection> {
 public:
  // Normalize the configurable cs: force >=4MB and a multiple of 4MB.
  static uint32_t normalize_cs(uint32_t chunk_size) {
    static constexpr unsigned MB = 4u * 1024u * 1024u;
    uint32_t v = (chunk_size < MB) ? MB : chunk_size;
    return ((v + MB - 1) / MB) * MB;
  }

  Hf3fsConnector(
      int num_workers, const std::string& meta_addr,
      const std::string& mgmtd_addr, uint32_t chunk_size,
      uint32_t transport_pool_max_connections = 4,
      size_t max_batch_bytes = 32 * 1024 * 1024,  // 32 MB
      uint32_t max_concurrent_requests = 32,
      uint32_t max_concurrent_requests_per_server = 0,  // 0 = 3FS default (8)
      const std::map<std::string, int>& per_op_workers = {},
      const std::string& cluster_id = "stage", uint32_t chain_table_id = 1,
      const std::string& base_dir = "lmcache", bool force_use_tcp = false,
      uint32_t dir_permission = 0755, uint32_t file_permission = 0644,
      const std::string& token = "");

  ~Hf3fsConnector() override;

  void close();

  // Reset connector state: close and reinitialize.
  // Used when ExtendClientSessionMismatch error is detected.
  void reset_connector();

  // Override to clean keepalive entries for drained future_ids.
  std::vector<Completion> drain_completions() override;

  // Cleanup data files: remove chunks from storage side, delete files from
  // MetaClient, and reset DataStore state. Called before each test to ensure
  // a clean 3FS state and prevent NoSpace(7021) errors.
  void cleanup_data_files();

  // Returns the python_buffer_keepalive_ map for pybind lambdas.
  // Keyed by future_id so entries can be removed when the completion
  // is drained (prevents premature GC of in-flight buffers).
  std::unordered_map<uint64_t, std::vector<py::object>>&
  get_python_buffer_keepalive();

  // Enable/disable debug logging ([LMCache DBG] output).
  static void set_dbg_enabled(bool enabled) { get_dbg_flag() = enabled; }

  void shutdown_connections() override;

  void print_timing_stats(const char* tag = nullptr) const;

  // Aggregated timing stats across all worker connections.
  Hf3fsTimingStats set_timing_;     // write-path timing (all workers)
  Hf3fsTimingStats get_timing_;     // read-path timing (all workers)
  Hf3fsTimingStats exists_timing_;  // exists-path timing (all workers)

 protected:
  void init_clients(const std::string& meta_addr,
                    const std::string& mgmtd_addr);

  // Key parsing helpers - public for pybind11 bindings
  std::string key_to_path(const std::string& key) const;
  std::string key_to_path_fsformat(const std::string& key) const;

  // ── Single-file + offset helpers ────────────────────────────────────
  // Lazily create the single data store on the first SET (size = cs). Returns
  // the created store, or nullptr if 3FS not ready. Thread-safe.
  DataStore* ensure_data_store(size_t first_len);

  // Look up offset for key, or allocate a new one. Thread-safe.
  // Returns (offset, size). Size is 0 if newly allocated.
  std::pair<uint64_t, uint32_t> get_or_allocate_offset(const std::string& key,
                                                       size_t data_len);

  // Remove key from offset map (for delete).
  void remove_offset(const std::string& key);

 protected:
  Hf3fsConnection create_connection() override;

  // Single operations
  void do_single_get(Hf3fsConnection& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;

  void do_single_set(Hf3fsConnection& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;

  bool do_single_exists(Hf3fsConnection& conn, const std::string& key) override;

  bool do_single_delete(Hf3fsConnection& conn, const std::string& key) override;

  // Batch operations (override base class virtual methods)
  void do_batch_get(Hf3fsConnection& conn, const Request& req) override;
  void do_batch_set(Hf3fsConnection& conn, const Request& req) override;
  void do_batch_exists(Hf3fsConnection& conn, const Request& req) override;
  void do_batch_delete(Hf3fsConnection& conn, const Request& req) override;

 private:
  const std::string meta_addr_;
  const std::string mgmtd_addr_;
  // cs = the 3FS layout chunk / per-WriteIO transfer block, normalized at
  // construction to >=4MB and a multiple of 4MB. Not the per-key size.
  const uint32_t chunk_size_;
  const uint32_t transport_pool_max_connections_;
  size_t max_batch_bytes_;
  uint32_t max_concurrent_requests_;
  uint32_t max_concurrent_requests_per_server_;
  std::map<std::string, int> per_op_workers_;

  // 3FS cluster identity / layout config (externalized via config file).
  const std::string cluster_id_;
  const uint32_t chain_table_id_;
  const std::string base_dir_;
  const bool force_use_tcp_;
  const uint32_t dir_permission_;
  const uint32_t file_permission_;
  // Security: token must match process uid/gid (root-token excepted) and is
  // only validated when the meta server has `authenticate` enabled.
  const std::string token_;

  // Python thread state - save before releasing GIL, restore after
  PyThreadState* py_thread_state_{nullptr};

  // Shared 3FS clients — created once in constructor, shared across all
  // workers. Using typed shared_ptrs to properly manage lifetime (noop deleter
  // on void* doesn't increment refcount, causing dangling pointers when locals
  // go out of scope).
  NetConfigPtr shared_net_config_{nullptr, &noop_net_config_deleter};
  std::shared_ptr<hf3fs::net::Client> shared_net_client_;
  std::shared_ptr<hf3fs::client::MgmtdClientForClient> shared_mgmtd_client_;
  std::shared_ptr<hf3fs::meta::client::MetaClient> shared_meta_client_;
  std::shared_ptr<hf3fs::storage::client::StorageClient> shared_storage_client_;
  std::shared_ptr<hf3fs::client::RoutingInfo> routing_info_;

  // Mutex to protect MetaClient operations across worker threads.
  // MetaClient is shared across all workers, and some operations
  // (open/close/sync) may not be thread-safe across multiple simultaneous
  // calls.
  std::mutex meta_mu_;

  // Keep-alive map for Python buffer objects keyed by future_id.
  // Worker threads read Python memory asynchronously, so objects must
  // outlive the operation. Entries removed in drain_completions().
  std::unordered_map<uint64_t, std::vector<py::object>>
      python_buffer_keepalive_;

  // ── Single-file data store (lazy) ─────────────────────────────────────
  // One DataStore backed by a single "lmcache/data_<chunk_size>_store", created
  // lazily on the first SET. data_stores_ holds 0..1 entries (used by cleanup /
  // meta-saver loops); active_store_ is a lock-free snapshot of the created
  // store for the hot I/O paths. unique_ptr because DataStore has a
  // std::mutex and is not movable.
  std::vector<std::unique_ptr<DataStore>> data_stores_;
  std::mutex store_init_mu_;
  std::atomic<DataStore*> active_store_{nullptr};

  // Authenticated user info for 3FS operations
  std::shared_ptr<hf3fs::flat::UserInfo> shared_user_info_;

  // Client configs - stored as members to ensure they outlive the clients.
  // Both MetaClient and StorageClient hold const references to their configs,
  // so the config objects must not be destroyed before the clients.
  std::shared_ptr<hf3fs::meta::client::MetaClient::Config> meta_config_;
  std::shared_ptr<hf3fs::storage::client::StorageClient::Config>
      storage_config_;

  // Dedicated executor for MetaClient's background coroutines.
  // Using a separate executor ensures MetaClient can be safely destroyed
  // without affecting other clients that share the network executor.
  std::unique_ptr<hf3fs::CPUExecutorGroup> meta_executor_;

  // Cached routing info version - only refresh when version changes.
  uint32_t cached_routing_version_{0};

  // Session health check - detect and recover from ExtendClientSessionMismatch.
  // Checks every 30 seconds; if extend fails with session mismatch, rebuilds
  // clients.
  std::chrono::steady_clock::time_point last_session_check_;
  std::mutex session_check_mu_;
  bool session_mismatch_detected_{false};
  bool try_extend_session();
  void check_session_health();

  // net::Client config - stored as member to ensure it outlives net::Client.
  // net::Client holds a const reference to this config.
  std::shared_ptr<hf3fs::net::Client::Config> net_config_;
};

}  // namespace connector
}  // namespace lmcache

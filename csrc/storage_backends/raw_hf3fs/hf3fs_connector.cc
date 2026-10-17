// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include "hf3fs_connector.h"
#include "data_store.h"

#include <Python.h>

#include <folly/Memory.h>
#include <folly/SharedMutex.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/futures/Future.h>
#include <folly/io/async/AsyncSocket.h>
#include <folly/io/async/AsyncTimeout.h>
#include <folly/io/async/EventBaseManager.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>

#include "client/meta/MetaClient.h"
#include "client/meta/ServerSelectionStrategy.h"
#include "client/mgmtd/MgmtdClientForAdmin.h"
#include "client/mgmtd/MgmtdClientForClient.h"
#include "client/mgmtd/RoutingInfo.h"
#include "client/storage/StorageClient.h"
#include "common/app/ClientId.h"
#include "common/logging/LogInit.h"
#include "common/net/ib/IBDevice.h"
#include "common/utils/Duration.h"
#include "common/utils/Result.h"
#include "common/utils/SysResource.h"
#include "fbs/core/user/User.h"
#include "fbs/meta/Common.h"   // InodeId, OpenFlags, PathAt
#include "fbs/meta/Schema.h"   // Inode, Layout, File, ChainId
#include "fbs/mgmtd/RoutingInfo.h"
#include "stubs/common/RealStubFactory.h"
#include "stubs/mgmtd/MgmtdServiceStub.h"

// LMCache 3FS connector: uses MetaClient to manage files and Layouts,
// ensuring correct chunk-to-chain mapping and data integrity.

namespace lmcache {
namespace connector {

// ---------- RDMA helpers (forward-declared for destructor) ----------
static size_t pgsz() {
  long p = sysconf(_SC_PAGESIZE);
  return p > 0 ? static_cast<size_t>(p) : 4096;
}
// Allocate RDMA buffer with NUMA-aware (node 0, where mlx5_0 NIC resides)
// and huge-page hint (MADV_HUGEPAGE) to reduce TLB misses and cross-NUMA
// memory access latency. Falls back gracefully if NUMA/hugepage unavailable.
// `zero_init` controls whether to memset(0) the buffer. GET buffers are
// fully overwritten by RDMA reads, so skipping memset saves ~50ms for a 1GB
// allocation. SET buffers still need zeroing to avoid leaking stale data
// into unwritten padding regions when chunkLen < stride.
static uint8_t* alloc_rdma(size_t sz, bool zero_init = true) {
  size_t ps = pgsz();
  size_t as = (sz + ps - 1) / ps * ps;
  void* p = nullptr;
  if (posix_memalign(&p, ps, as) != 0) return nullptr;
  // Hint kernel to use huge pages (2MB) for this allocation.
  // Reduces TLB misses for large buffers (~256MB per sub-batch).
  madvise(p, as, MADV_HUGEPAGE);
  // Bind to NUMA node 0 (where mlx5_0 RDMA NIC resides) for lower latency.
  if (numa_available() != -1) {
    unsigned long nodemask = 1UL;
    mbind(p, as, MPOL_BIND, &nodemask, 64, MPOL_MF_MOVE);
  }
  if (zero_init) {
    std::memset(p, 0, as);
  }
  return static_cast<uint8_t*>(p);
}
static void free_rdma(uint8_t* b) { free(b); }

void Hf3fsConnection::delete_io_buffer(void* p) {
  delete static_cast<hf3fs::storage::client::IOBuffer*>(p);
}

// Keep-alive list accessed by pybind lambdas to prevent Python GC.
// Accessible from within pybind bindings since the lambda is defined
// inside the py::class_<Hf3fsConnector> scope.
std::unordered_map<uint64_t, std::vector<py::object>>& Hf3fsConnector::get_python_buffer_keepalive() {
  return python_buffer_keepalive_;
}

std::vector<Completion> Hf3fsConnector::drain_completions() {
  auto dst = ConnectorBase::drain_completions();
  // Free keepalive entries for drained future_ids so Python buffers can GC.
  for (auto& c : dst) {
    auto it = python_buffer_keepalive_.find(c.future_id);
    if (it != python_buffer_keepalive_.end()) {
      python_buffer_keepalive_.erase(it);
    }
  }
  return dst;
}

Hf3fsConnection::~Hf3fsConnection() {
  if (user_info_buf) {
    delete[] user_info_buf;
    user_info_buf = nullptr;
  }
  bool buf_shared = (rd_buf_cache_ && wr_buf_cache_ && rd_buf_cache_ == wr_buf_cache_);
  if (rd_buf_cache_) {
    if (!buf_shared) free_rdma(rd_buf_cache_);
    rd_buf_cache_ = nullptr;
    rd_buf_cache_size_ = 0;
  }
  if (rd_io_buf_cache_) {
    if (!(buf_shared && rd_io_buf_cache_ == wr_io_buf_cache_))
      delete static_cast<hf3fs::storage::client::IOBuffer*>(rd_io_buf_cache_);
    rd_io_buf_cache_ = nullptr;
  }
  if (wr_buf_cache_) {
    free_rdma(wr_buf_cache_);
    wr_buf_cache_ = nullptr;
    wr_buf_cache_size_ = 0;
  }
  if (wr_io_buf_cache_) {
    delete static_cast<hf3fs::storage::client::IOBuffer*>(wr_io_buf_cache_);
    wr_io_buf_cache_ = nullptr;
  }
}

// ========== Shared client initialization ==========

void Hf3fsConnector::init_clients(
    const std::string& meta_addr,
    const std::string& mgmtd_addr) {

  // 0. Initialize IBManager (required before any RDMA operations)
  if (!hf3fs::net::IBManager::initialized()) {
    hf3fs::net::IBConfig ibCfg;
    ibCfg.set_allow_no_usable_devices(true);
    auto ibRes = hf3fs::net::IBManager::start(ibCfg);
    if (ibRes.hasError()) {
      DBG("IBManager::start failed: %s",
          std::string(ibRes.error().message()).c_str());
      return;
    }
  }

  // 1. Create net::Client (RDMA transport layer)
  // Store config as member to ensure it outlives net::Client.
  // net::Client holds a const reference to the config.
  net_config_ = std::make_shared<hf3fs::net::Client::Config>();
  // P0: Increase thread pool to avoid IO thread starvation when 4 workers
  // are all waiting for RDMA completions. IO threads handle RDMA completion
  // callbacks (lightweight), proc threads handle serialization (CPU-intensive).
  // IO threads >= proc threads recommended for RDMA workloads.
  net_config_->thread_pool().set_num_proc_threads(4);
  net_config_->thread_pool().set_num_io_threads(16);
  net_config_->thread_pool().set_num_connect_threads(2);
  net_config_->set_force_use_tcp(force_use_tcp_);  // RDMA unless configured otherwise
  // Increase RDMA send buffer count from 32 to 64 per QP to reduce send queue
  // contention with 4 workers x 16 in-flight = 64 concurrent sends.
  net_config_->io_worker().ibsocket().set_send_buf_cnt(64);
  net_config_->io_worker().ibsocket().set_max_rdma_wr(256);

  shared_net_client_ = std::make_shared<hf3fs::net::Client>(*net_config_);
  auto startRes = shared_net_client_->start("lmcache-3fs-connector");
  if (startRes.hasError()) {
    DBG("net::Client start failed: %s",
        std::string(startRes.error().message()).c_str());
    return;
  }

  // 2. Create MgmtdClientForClient
  hf3fs::client::MgmtdClientForClient::Config mgmtdCfg;
  hf3fs::net::Address mgmtAddr = hf3fs::net::Address::fromString(mgmtd_addr);
  std::vector<hf3fs::net::Address> addrs = {mgmtAddr};
  mgmtdCfg.set_mgmtd_server_addresses(addrs);
  mgmtdCfg.set_network_type(hf3fs::net::Address::Type::RDMA);
  mgmtdCfg.set_enable_auto_refresh(false);
  mgmtdCfg.set_enable_auto_heartbeat(false);

  auto ctxCreator = [this](hf3fs::net::Address addr) {
    return shared_net_client_->serdeCtx(addr);
  };

  shared_mgmtd_client_ = std::make_shared<hf3fs::client::MgmtdClientForClient>(
      hf3fs::String(cluster_id_),  // cluster_id from config
      std::make_unique<hf3fs::stubs::RealStubFactory<
          hf3fs::mgmtd::MgmtdServiceStub>>(ctxCreator),
      mgmtdCfg);

  auto physicalHostname = hf3fs::SysResource::hostname(true);
  // Build unique clientId BEFORE start() so the initial authenticateClientSession
  // and subsequent auto-extendClientSession use the correct payload.
  //
  // Use getpid() to make clientId unique across process restarts.
  // Without getpid(), each new process gets clientId=hostname_0, but the old
  // process's session (20-min timeout) is still on the server, causing
  // ExtendClientSessionMismatch when the new process tries to extend with a
  // different clientStart timestamp.
  static uint64_t sessionCounter = 0;
  std::string uniqueClientId = std::string(*physicalHostname) + "_" +
                               std::to_string(static_cast<int>(getpid())) + "_" +
                               std::to_string(__sync_fetch_and_add(&sessionCounter, 1));
  hf3fs::client::IMgmtdClientForClient::ClientSessionPayload payload;
  payload.clientId = hf3fs::String(uniqueClientId);
  payload.nodeType = hf3fs::flat::NodeType::CLIENT;
  payload.userInfo = hf3fs::flat::UserInfo{};
  shared_mgmtd_client_->setClientSessionPayload(std::move(payload));

  auto& tpg = shared_net_client_->tpg();
  auto* bgPool = &tpg.bgThreadPool().randomPick();

  shared_mgmtd_client_->start(bgPool).scheduleOn(bgPool).start();
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  // Detect ExtendClientSessionMismatch early: if the old process is still
  // alive on the server with a different clientStart timestamp, the
  // auto-extend background thread would keep failing every 10 seconds.
  // We detect this and reset the MgmtdClient to create a fresh session.
  {
    DBG("Testing session extend...");
    auto extendRes = folly::coro::blockingWait(shared_mgmtd_client_->extendClientSession());
    if (extendRes.hasError()) {
      std::string errMsg = extendRes.error().describe();
      DBG("extendClientSession failed: %s", errMsg.c_str());
      if (errMsg.find("ExtendClientSessionMismatch") != std::string::npos ||
          errMsg.find("kExtendClientSessionMismatch") != std::string::npos ||
          errMsg.find("clientStart") != std::string::npos) {
        // Close and recreate the MgmtdClient with a fresh session.
        // The old MgmtdClient's startTime_ is stale and will keep mismatching.
        shared_mgmtd_client_.reset();
        
        // Wait a moment for the old session to be noticed as stale
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Recreate MgmtdClient with a new unique clientId (counter already incremented)
        shared_mgmtd_client_ = std::make_shared<hf3fs::client::MgmtdClientForClient>(
            hf3fs::String(cluster_id_),
            std::make_unique<hf3fs::stubs::RealStubFactory<
                hf3fs::mgmtd::MgmtdServiceStub>>(ctxCreator),
            mgmtdCfg);
        
        // Re-set the session payload for the new MgmtdClient
        // Use getpid() to ensure unique clientId even if first extend failed.
        auto physicalHostname2 = hf3fs::SysResource::hostname(true);
        static uint64_t sessionCounter2 = 0;
        std::string uniqueClientId2 = std::string(*physicalHostname2) + "_" +
                                      std::to_string(static_cast<int>(getpid())) + "_" +
                                      std::to_string(__sync_fetch_and_add(&sessionCounter2, 1));
        hf3fs::client::IMgmtdClientForClient::ClientSessionPayload payload2;
        payload2.clientId = hf3fs::String(uniqueClientId2);
        payload2.nodeType = hf3fs::flat::NodeType::CLIENT;
        payload2.userInfo = hf3fs::flat::UserInfo{};
        shared_mgmtd_client_->setClientSessionPayload(std::move(payload2));
        
        shared_mgmtd_client_->start(bgPool).scheduleOn(bgPool).start();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        // Try extend again
        auto extendRes2 = folly::coro::blockingWait(shared_mgmtd_client_->extendClientSession());
        if (extendRes2.hasError()) {
          DBG("WARNING: Session mismatch persists after reset: %s",
              extendRes2.error().describe().c_str());
        } else {
          DBG("Session extend succeeded after reset.");
        }
      }
    } else {
      DBG("extendClientSession succeeded, session is valid.");
    }
  }

  // Refresh routing info (now guaranteed to work)
  // Retry up to 3 times with delays to handle transient Mgmtd issues.
  {
    bool refreshOk = false;
    for (int attempt = 0; attempt < 3 && !refreshOk; ++attempt) {
      auto res = folly::coro::blockingWait(shared_mgmtd_client_->refreshRoutingInfo(true));
      if (res.hasError()) {
        DBG("refreshRoutingInfo attempt %d failed: %s", attempt + 1,
            std::string(res.error().message()).c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      } else {
        refreshOk = true;
        DBG("refreshRoutingInfo succeeded on attempt %d", attempt + 1);
      }
    }
    if (!refreshOk) {
      DBG("refreshRoutingInfo failed after 3 attempts, will retry on next batchWrite");
    }
  }

  // Validate routing info version before caching.
  // A version of 0 means stale/empty RoutingInfo which causes InvalidArg(7002).
  {
    auto ri = shared_mgmtd_client_->getRoutingInfo();
    if (ri && ri->raw()) {
      auto version = ri->raw()->routingInfoVersion.toUnderType();
      if (version == 0) {
        DBG("WARNING: routingInfo version is 0 after refresh, will retry on first operation");
      } else {
        DBG("Cached routingInfo version: %d", version);
      }

      // Debug: dump routing info chain→node mapping (only with LMCACHE_DBG=1)
      DBG("=== RoutingInfo (version=%u chains=%zu nodes=%zu targets=%zu) ===",
          version,
          ri->raw()->chains.size(),
          ri->raw()->nodes.size(),
          ri->raw()->targets.size());
      // Print all nodes
      for (auto& [nodeId, nodeInfo] : ri->raw()->nodes) {
        auto addrs = nodeInfo.extractAddresses(hf3fs::String("StorageService"));
        std::string addr = addrs.empty() ? "-" : addrs[0].toString();
        DBG("  Node: ID=%u addr=%s status=%d",
            nodeId.toUnderType(), addr.c_str(), (int)nodeInfo.status);
      }
      // Print all targets with their chain mapping
      for (auto& [targetId, targetInfo] : ri->raw()->targets) {
        DBG("  Target: id=%lu chainId=%u nodeId=%d",
            targetId.toUnderType(),
            targetInfo.chainId.toUnderType(),
            targetInfo.nodeId.has_value() ? (int)targetInfo.nodeId->toUnderType() : -1);
      }
    } else {
      DBG("WARNING: routingInfo is null after refresh, will retry on first operation");
    }
  }

  hf3fs::storage::client::StorageClient::Config storageCfg;
  storage_config_ = std::make_shared<hf3fs::storage::client::StorageClient::Config>();

  // Set transport_pool max_connections from config (default: 1)
  storage_config_->net_client().io_worker().transport_pool().set_max_connections(transport_pool_max_connections_);
  DBG("StorageClient Config - transport_pool.max_connections=%u (from config)",
      storage_config_->net_client().io_worker().transport_pool().max_connections());
  DBG("StorageClient Config - connect_concurrency.max_concurrency=%u",
      storage_config_->net_client().io_worker().connect_concurrency_limiter().max_concurrency());

  // Apply max_batch_bytes and max_concurrent_requests to traffic_control config
  storage_config_->traffic_control().read().set_max_batch_bytes(max_batch_bytes_);
  storage_config_->traffic_control().read().set_max_concurrent_requests(max_concurrent_requests_);
  storage_config_->traffic_control().write().set_max_batch_bytes(max_batch_bytes_);
  storage_config_->traffic_control().write().set_max_concurrent_requests(max_concurrent_requests_);
  if (max_concurrent_requests_per_server_ > 0) {
    storage_config_->traffic_control().read().set_max_concurrent_requests_per_server(
        max_concurrent_requests_per_server_);
    storage_config_->traffic_control().write().set_max_concurrent_requests_per_server(
        max_concurrent_requests_per_server_);
  }
  DBG("StorageClient traffic_control: max_batch_bytes=%zu, max_concurrent_requests=%u, max_concurrent_requests_per_server=%u",
      max_batch_bytes_, max_concurrent_requests_,
      max_concurrent_requests_per_server_ > 0 ? max_concurrent_requests_per_server_ : 8);

  shared_storage_client_ = hf3fs::storage::client::StorageClient::create(
      hf3fs::ClientId::random(), *storage_config_, *shared_mgmtd_client_);
  if (!shared_storage_client_) {
    DBG("StorageClient creation failed");
    return;
  }

  // 4. Create MetaClient
  // Store config as shared_ptr member to ensure it outlives MetaClient.
  // MetaClient holds a const reference to the config, so the config object
  // must not be destroyed before MetaClient.
  // Also pass shared_storage_client_ (shared_ptr) directly — NOT dereferenced,
  // to ensure the StorageClient lives long enough (MetaClient may store a
  // reference/pointer to it internally).
  meta_config_ = std::make_shared<hf3fs::meta::client::MetaClient::Config>();
  meta_config_->set_selection_mode(hf3fs::meta::client::ServerSelectionMode::RandomFollow);
  meta_config_->set_network_type(hf3fs::net::Address::Type::RDMA);
  // Disable background closer coroutines to avoid segfault during shutdown.
  // MetaClient's background coroutines block in co_dequeue and don't respond
  // to cancellation, causing segfault when destroyed.
  meta_config_->background_closer().coroutine_pool().set_coroutines_num(0);

  shared_meta_client_ = std::make_shared<hf3fs::meta::client::MetaClient>(
      hf3fs::ClientId::random(),
      *meta_config_,
      std::make_unique<hf3fs::meta::client::MetaClient::StubFactory>(ctxCreator),
      shared_mgmtd_client_,
      shared_storage_client_,
      true /* dynStripe */);

  // Create dedicated executor for MetaClient's background coroutines.
  // Using a separate executor prevents conflicts with the shared network executor.
  meta_executor_ = std::make_unique<hf3fs::CPUExecutorGroup>(
      2, "lmcache-meta-client");

  shared_meta_client_->start(*meta_executor_);

  // 5. Authenticate user and initialize LMCache directory
  // Use current process UID/GID so files are owned by the running user
  uid_t current_uid = getuid();
  gid_t current_gid = getgid();
  {
    hf3fs::flat::UserInfo ui(static_cast<hf3fs::flat::Uid>(current_uid),
                              static_cast<hf3fs::flat::Gid>(current_gid));
    auto authRes = folly::coro::blockingWait(shared_meta_client_->authenticate(ui));
    if (authRes.hasError()) {
      DBG("authenticate failed: %s",
          std::string(authRes.error().message()).c_str());
      return;
    }
    // Store authenticated user info in shared storage for later use
    shared_user_info_ = std::make_shared<hf3fs::flat::UserInfo>(*authRes);
  }

  // Create the LMCache dedicated directory (e.g. "lmcache") under 3FS root
  {
    hf3fs::flat::UserInfo ui(static_cast<hf3fs::flat::Uid>(current_uid),
                              static_cast<hf3fs::flat::Gid>(current_gid));
    auto dirResult = folly::coro::blockingWait(
        shared_meta_client_->mkdirs(ui, hf3fs::meta::InodeId::root(),
                                     hf3fs::Path(base_dir_),
                                     hf3fs::flat::Permission(dir_permission_),
                                     true, std::nullopt));
    if (dirResult.hasError()) {
      std::string errMsg = dirResult.error().describe();
      // Ignore "already exists" errors
      if (errMsg.find("Exists") == std::string::npos &&
          errMsg.find("3007") == std::string::npos) {
        DBG("Failed to create lmcache dir: %s", errMsg.c_str());
        // Non-fatal: continue without the directory
      }
    }
  }

  // 5. Get and cache routing info
  routing_info_ = shared_mgmtd_client_->getRoutingInfo();

  // 6. Initialize data stores with size-based chunk allocation.
  // Each size range gets its own 3FS file with matching chunk_size.
  // This ensures each key maps to exactly 1 chunk, eliminating multi-chunk overhead.
  // Size ranges: small (<=4MB), medium (<=8MB), large (<=16MB), extra large (>16MB)
  struct DataStoreRange {
    uint32_t chunk_size;
    uint32_t file_index;
  };
  std::vector<DataStoreRange> store_ranges = {
    {4 * 1024 * 1024, 0},   // 4MB
    {8 * 1024 * 1024, 1},   // 8MB
    {16 * 1024 * 1024, 2},  // 16MB
    {32 * 1024 * 1024, 3},  // 32MB
  };
  
  data_stores_.reserve(store_ranges.size());
  for (auto& range : store_ranges) {
    auto ds = std::make_unique<DataStore>(range.file_index, range.chunk_size,
                                          chain_table_id_);
    ds->init_file(shared_meta_client_.get(), shared_storage_client_.get(),
                  *shared_user_info_, *routing_info_->raw(), range.chunk_size,
                  base_dir_, 1, file_permission_);  // 1 file per range
    data_stores_.push_back(std::move(ds));
    DBG("init_data_stores: created data_store[%u] chunk_size=%u path=%s",
        range.file_index, range.chunk_size, data_stores_.back()->get_meta_path().c_str());
  }
  num_files_ = static_cast<uint32_t>(data_stores_.size());

  // Pre-compute active data_store index from chunk_size_ (never changes).
  active_ds_index_ = compute_data_store_index(chunk_size_);
  DBG("Initialized %u data files, active_ds_index=%u for chunk_size=%u",
      num_files_, active_ds_index_, chunk_size_);
  
  // Start background meta savers for each data store.
  for (auto& ds : data_stores_) {
    ds->start_meta_saver();
  }
}

Hf3fsConnector::Hf3fsConnector(int num_workers,
    const std::string& meta_addr, const std::string& mgmtd_addr,
    uint32_t chunk_size, uint32_t transport_pool_max_connections,
    size_t max_batch_bytes, uint32_t max_concurrent_requests,
    uint32_t max_concurrent_requests_per_server,
    const std::map<std::string, int>& per_op_workers,
    const std::string& cluster_id, uint32_t chain_table_id,
    const std::string& base_dir, bool force_use_tcp,
    uint32_t dir_permission, uint32_t file_permission)
    : ConnectorBase<Hf3fsConnection>(num_workers,
        WorkerPoolConfig{{per_op_workers.begin(), per_op_workers.end()}}),
      meta_addr_(meta_addr), mgmtd_addr_(mgmtd_addr), chunk_size_(chunk_size),
      transport_pool_max_connections_(transport_pool_max_connections),
      max_batch_bytes_(max_batch_bytes),
      max_concurrent_requests_(max_concurrent_requests),
      max_concurrent_requests_per_server_(max_concurrent_requests_per_server),
      per_op_workers_(per_op_workers),
      cluster_id_(cluster_id.empty() ? "stage" : cluster_id),
      chain_table_id_(chain_table_id), base_dir_(base_dir.empty() ? "lmcache" : base_dir),
      force_use_tcp_(force_use_tcp), dir_permission_(dir_permission),
      file_permission_(file_permission) {
  // Check LMCACHE_DBG environment variable for debug output
  const char* dbg_env = std::getenv("LMCACHE_DBG");
  bool dbg = dbg_env && (std::strcmp(dbg_env, "1") == 0 || std::strcmp(dbg_env, "true") == 0);
  if (dbg) {
    get_dbg_flag() = true;
  }
  // Control the 3FS library's own folly-logging output: default to ERR (quiet)
  // so the noisy IBDevice/MgmtdClient/StorageClient INFO (and shutdown WARNING)
  // chatter is suppressed; raise to INFO when LMCACHE_DBG is enabled so the full
  // 3FS trace backs up our DBG() prints. Mirrors the upstream usrbio connector
  // (3FS src/lib/api/UsrbIo.cc initOrDie defaulting to WARN).
  hf3fs::logging::initOrDie(dbg ? "INFO" : "ERR");
  // Save Python thread state before doing 3FS initialization
  py_thread_state_ = PyThreadState_Get();
  init_clients(meta_addr, mgmtd_addr);
  start_workers();
}

Hf3fsConnector::~Hf3fsConnector() {
  // Intentionally do nothing - clients are leaked to prevent destructor segfault.
  // See close() for details.
}

void Hf3fsConnector::close() {
  // Proper cleanup sequence:
  // 1. Stop meta savers (must be before data_stores_ destroyed)
  // 2. Stop MetaClient's background coroutines (uses dedicated executor)
  // 3. Destroy MetaClient (safe because no other threads use its executor)
  // 4. Destroy StorageClient
  // 5. Destroy configs (must be after clients)
  // 6. Destroy other clients

  // Stop background meta savers.
  for (auto& ds : data_stores_) {
    ds->stop_meta_saver();
  }

  if (shared_meta_client_) {
    try {
      shared_meta_client_->stop();
    } catch (...) {
      // Ignore errors during shutdown
    }
    shared_meta_client_.reset();
  }

  // Destroy the dedicated executor AFTER MetaClient is stopped.
  // This ensures all background coroutines have finished.
  meta_executor_.reset();

  if (shared_storage_client_) {
    shared_storage_client_.reset();
  }

  // Reset configs AFTER clients to prevent dangling references.
  meta_config_.reset();
  storage_config_.reset();
  shared_mgmtd_client_.reset();
  if (shared_net_client_) {
    shared_net_client_.reset();
  }
  routing_info_.reset();

  // Emit per-op timing breakdown on close so the L2 bench / adapters that
  // close the connector can see where Load/Store/Lookup cost actually goes
  // (the stats are only populated across the run, so print here).
  if (get_dbg_flag()) {
    print_timing_stats("adapter close");
  }
}

// Check if an error indicates a session mismatch issue.
static bool is_session_mismatch_error(const std::string& error_msg) {
  return error_msg.find("ExtendClientSessionMismatch") != std::string::npos ||
         error_msg.find("kExtendClientSessionMismatch") != std::string::npos ||
         error_msg.find("clientStart") != std::string::npos;
}

void Hf3fsConnector::reset_connector() {
  DBG("Session reset triggered, rebuilding MgmtdClient only...");
  
  // Close existing MgmtdClient (MetaClient is session-independent, leave it alone)
  shared_mgmtd_client_.reset();
  
  // Wait longer for the old session to be noticed as stale by the server.
  // The 3FS server has a session timeout (default 20 minutes), but we need to
  // wait at least a few seconds to ensure the server stops responding to old
  // session extend requests before we create a new client.
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  
  // Recreate MgmtdClient with a new unique clientId
  hf3fs::client::MgmtdClientForClient::Config mgmtdCfg;
  hf3fs::net::Address mgmtAddr = hf3fs::net::Address::fromString(mgmtd_addr_);
  std::vector<hf3fs::net::Address> addrs = {mgmtAddr};
  mgmtdCfg.set_mgmtd_server_addresses(addrs);
  mgmtdCfg.set_network_type(hf3fs::net::Address::Type::RDMA);
  mgmtdCfg.set_enable_auto_refresh(false);
  mgmtdCfg.set_enable_auto_heartbeat(false);
  
  auto ctxCreator = [this](hf3fs::net::Address addr) {
    return shared_net_client_->serdeCtx(addr);
  };
  
  shared_mgmtd_client_ = std::make_shared<hf3fs::client::MgmtdClientForClient>(
      hf3fs::String(cluster_id_),
      std::make_unique<hf3fs::stubs::RealStubFactory<
          hf3fs::mgmtd::MgmtdServiceStub>>(ctxCreator),
      mgmtdCfg);
  
  // Set session payload
  // Use getpid() to ensure unique clientId even if reset_connector is called
  // multiple times within the same process.
  auto physicalHostname = hf3fs::SysResource::hostname(true);
  static uint64_t sessionCounter = 0;
  std::string uniqueClientId = std::string(*physicalHostname) + "_" +
                               std::to_string(static_cast<int>(getpid())) + "_" +
                               std::to_string(__sync_fetch_and_add(&sessionCounter, 1));
  hf3fs::client::IMgmtdClientForClient::ClientSessionPayload payload;
  payload.clientId = hf3fs::String(uniqueClientId);
  payload.nodeType = hf3fs::flat::NodeType::CLIENT;
  payload.userInfo = hf3fs::flat::UserInfo{};
  shared_mgmtd_client_->setClientSessionPayload(std::move(payload));
  
  auto& tpg = shared_net_client_->tpg();
  auto* bgPool = &tpg.bgThreadPool().randomPick();
  shared_mgmtd_client_->start(bgPool).scheduleOn(bgPool).start();
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  
  // Test session extend
  auto extendRes = folly::coro::blockingWait(shared_mgmtd_client_->extendClientSession());
  if (extendRes.hasError()) {
    DBG("WARNING: Session extend failed after reset: %s",
        extendRes.error().describe().c_str());
  } else {
    DBG("Session extend succeeded after reset.");
  }
  
  // Refresh routing info
  auto res = folly::coro::blockingWait(shared_mgmtd_client_->refreshRoutingInfo(true));
  if (res.hasError()) {
    DBG("WARNING: refreshRoutingInfo failed after reset: %s",
        res.error().describe().c_str());
  } else {
    DBG("refreshRoutingInfo succeeded after reset.");
  }
  
  // Update DataStore cached routing info to match new MgmtdClient routing.
  // After MgmtdClient rebuild, the routing info version may have changed,
  // and stale chain IDs in DataStore would cause writes to wrong chains.
  if (shared_mgmtd_client_->getRoutingInfo() && shared_mgmtd_client_->getRoutingInfo()->raw()) {
    const auto& new_ri = *shared_mgmtd_client_->getRoutingInfo()->raw();
    for (auto& ds : data_stores_) {
      ds->refresh_routing_info(new_ri);
    }
    DBG("DataStore routing info refreshed (%u files)", num_files_);
  }
  
  DBG("Connector MgmtdClient rebuilt successfully");
}

// Try to extend the MgmtdClient session. Returns true if successful.
// If the extend fails with ExtendClientSessionMismatch, sets session_mismatch_detected_ flag.
bool Hf3fsConnector::try_extend_session() {
  if (!shared_mgmtd_client_) return false;
  
  auto extendRes = folly::coro::blockingWait(shared_mgmtd_client_->extendClientSession());
  if (!extendRes.hasError()) {
    return true;
  }
  
  std::string errMsg = extendRes.error().describe();
  DBG("try_extend_session failed: %s", errMsg.c_str());
  
  if (is_session_mismatch_error(errMsg)) {
    session_mismatch_detected_ = true;
    DBG("Session mismatch detected - will trigger connector reset on next batch");
    return false;
  }
  
  // Other errors - not session mismatch, ignore for now
  return false;
}

// Periodically check session health. If session mismatch is detected,
// force a connector reset to rebuild all 3FS clients with fresh session.
void Hf3fsConnector::check_session_health() {
  // If we already detected a mismatch, force a reset now (don't wait for 30s check)
  if (session_mismatch_detected_) {
    DBG("Session mismatch detected, resetting connector...");
    session_mismatch_detected_ = false;
    reset_connector();
    return;
  }
  
  // Check at most every 30 seconds
  auto now = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(session_check_mu_);
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_session_check_).count() < 30) {
      return;
    }
    last_session_check_ = now;
  }
  
  // Try to extend session
  if (try_extend_session()) {
    // Session is healthy, no action needed
    return;
  }
  
  // If we get here, extend failed without session mismatch error
  // This could be a transient error, no action needed
  DBG("Session extend failed (non-mismatch), ignoring");
}

void Hf3fsConnector::shutdown_connections() {}

// ---------- Cleanup: release chunks on storage nodes and delete 3FS files ----------
void Hf3fsConnector::cleanup_data_files() {
  if (!shared_storage_client_ || !shared_meta_client_ || !shared_mgmtd_client_ ||
      !shared_user_info_ || !routing_info_ || routing_info_->raw() == nullptr) {
    return;
  }

  const auto& ri = *routing_info_->raw();
  const auto& ui = *shared_user_info_;

  // 1. Release chunks on storage side by calling removeChunks for each chain.
  //    We iterate routing_info_.chains which maps ChainId -> ChainInfo (including chainVersion).
  std::vector<hf3fs::storage::client::RemoveChunksOp> removeOps;
  removeOps.reserve(ri.chains.size());

  for (const auto& [chainIdRaw, chainInfo] : ri.chains) {
    hf3fs::flat::ChainId chainId(chainIdRaw.toUnderType());
    auto op = shared_storage_client_->createRemoveOp(
        chainId, hf3fs::storage::ChunkId(), hf3fs::storage::ChunkId(),
        1000000);
    removeOps.push_back(std::move(op));
  }

  if (!removeOps.empty()) {
    try {
      auto r = folly::coro::blockingWait(
          shared_storage_client_->removeChunks(
              std::span(removeOps.data(), removeOps.size()), ui));
      if (r.hasError()) {
        DBG("cleanup_data_files: removeChunks failed: %s",
            std::string(r.error().message()).c_str());
      } else {
        DBG("cleanup_data_files: removeChunks succeeded, %zu chains cleaned",
            removeOps.size());
      }
    } catch (...) {
      DBG("cleanup_data_files: removeChunks threw exception");
    }
  }

  // 2. Delete the lmcache directory and all its contents from MetaClient.
  //    This removes data_*.store and data_*.meta files from the 3FS metadata tree.
  try {
    auto res = folly::coro::blockingWait(
        shared_meta_client_->remove(ui, hf3fs::meta::InodeId::root(),
                                     std::optional<hf3fs::Path>{hf3fs::Path{base_dir_}},
                                     true /*recursive*/));
    if (res.hasError()) {
      // Ignore "not found" errors — files may have already been deleted.
      std::string errMsg = res.error().describe();
      if (errMsg.find("NotFound") == std::string::npos &&
          errMsg.find("3000") == std::string::npos) {
        DBG("cleanup_data_files: MetaClient::remove failed: %s", errMsg.c_str());
      }
    } else {
      DBG("cleanup_data_files: deleted lmcache directory from MetaClient");
    }
  } catch (...) {
    DBG("cleanup_data_files: MetaClient::remove threw exception");
  }

  // 3. Reset DataStore internal state so next test starts fresh.
  for (auto& ds : data_stores_) {
    ds->clear_state();
  }
  DBG("cleanup_data_files: complete");
}

void Hf3fsConnector::print_timing_stats(const char* tag) const {
  auto print_one = [](const char* name, const Hf3fsTimingStats& t) {
    uint64_t c = t.count.load(std::memory_order_relaxed);
    if (c == 0) {
      fprintf(stderr, "[LMCache 3FS TIMING] %s: no samples\n", name);
      return;
    }
    auto avg = [&](uint64_t v) -> double {
      return static_cast<double>(v) / static_cast<double>(c) / 1000.0;
    };
    double total_us     = avg(t.ns_total.load(std::memory_order_relaxed));
    double lookup_us    = avg(t.ns_lookup.load(std::memory_order_relaxed));
    double buf_us       = avg(t.ns_buf_alloc.load(std::memory_order_relaxed));
    double memcpy_us    = avg(t.ns_memcpy.load(std::memory_order_relaxed));
    double create_us    = avg(t.ns_create_io.load(std::memory_order_relaxed));
    double batch_us     = avg(t.ns_batch_io.load(std::memory_order_relaxed));
    double sync_us      = avg(t.ns_sync.load(std::memory_order_relaxed));
    double buf_compute  = avg(t.ns_buf_compute.load(std::memory_order_relaxed));
    double buf_lock_chk = avg(t.ns_buf_lock_check.load(std::memory_order_relaxed));
    double buf_lock_get = avg(t.ns_buf_lock_get.load(std::memory_order_relaxed));
    double buf_alloc    = avg(t.ns_buf_alloc_rdma.load(std::memory_order_relaxed));
    double buf_register = avg(t.ns_buf_register.load(std::memory_order_relaxed));
    double buf_cache    = avg(t.ns_buf_cache_store.load(std::memory_order_relaxed));
    double buf_subtotal = buf_compute + buf_lock_chk + buf_lock_get
                        + buf_alloc + buf_register + buf_cache;
    uint64_t hits   = t.ns_buf_cache_hits.load(std::memory_order_relaxed);
    uint64_t misses = t.ns_buf_cache_misses.load(std::memory_order_relaxed);
    double hit_rate = (hits + misses > 0)
        ? 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses)
        : 0.0;
    double accounted = lookup_us + buf_us + memcpy_us + create_us + batch_us + sync_us;
    fprintf(stderr,
            "[LMCache 3FS TIMING] %s: count=%lu total=%.2fus "
            "lookup=%.2fus(%.1f%%) buf=%.2fus(%.1f%%) memcpy=%.2fus(%.1f%%) "
            "create_io=%.2fus(%.1f%%) batch_io=%.2fus(%.1f%%) sync=%.2fus(%.1f%%) "
            "accounted=%.2fus(%.1f%%)\n",
            name, c, total_us,
            lookup_us, lookup_us / total_us * 100.0,
            buf_us, buf_us / total_us * 100.0,
            memcpy_us, memcpy_us / total_us * 100.0,
            create_us, create_us / total_us * 100.0,
            batch_us, batch_us / total_us * 100.0,
            sync_us, sync_us / total_us * 100.0,
            accounted, accounted / total_us * 100.0);
    fprintf(stderr,
            "[LMCache 3FS TIMING] %s: buf_sub=[compute=%.2fus lock_chk=%.2fus "
            "lock_get=%.2fus alloc_rdma=%.2fus register=%.2fus cache_store=%.2fus] "
            "subtotal=%.2fus hit_rate=%.1f%%(hits=%lu misses=%lu)\n",
            name,
            buf_compute, buf_lock_chk, buf_lock_get, buf_alloc, buf_register,
            buf_cache, buf_subtotal, hit_rate, hits, misses);
  };
  fprintf(stderr, "[LMCache 3FS TIMING] === %s ===\n", tag ? tag : "summary");
  print_one("SET (write)", set_timing_);
  print_one("GET (read)",  get_timing_);
  print_one("EXISTS (lookup)", exists_timing_);
  fflush(stderr);
}

uint32_t Hf3fsConnector::compute_ds_index_from_size(uint32_t data_size) const {
  // Route by data_size to size-based data store (no key parsing, O(1)).
  if (data_size <= 4 * 1024 * 1024) return 0;
  if (data_size <= 8 * 1024 * 1024) return 1;
  if (data_size <= 16 * 1024 * 1024) return 2;
  return 3;
}

uint32_t Hf3fsConnector::compute_file_index(const std::string& key, uint32_t data_size) const {
  // Legacy: parse key for format detection, then route by data_size.
  // Kept for backward compatibility; hot path should use compute_ds_index_from_size.
  return compute_ds_index_from_size(data_size);
}

uint32_t Hf3fsConnector::compute_data_store_index(uint32_t chunk_size) const {
  for (uint32_t i = 0; i < data_stores_.size(); ++i) {
    if (data_stores_[i]->get_chunk_size() == chunk_size) {
      return i;
    }
  }
  return 0;
}

bool Hf3fsConnector::parse_key(const std::string& key, uint64_t& cid) const {
  auto c = key.find(':');
  if (c == std::string::npos) {
    try { cid = std::stoull(key); } catch (...) { return false; }
  } else {
    try {
      uint64_t h = std::stoull(key.substr(0, c));
      uint64_t l = std::stoull(key.substr(c + 1));
      cid = (h << 32) | l;
    } catch (...) { return false; }
  }
  return true;
}

std::string Hf3fsConnector::key_to_path(const std::string& key) const {
  // Two key formats supported:
  //
  // 1) FSConnector-compatible (new):
  //    <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>[@<cache_salt>]
  //    Output: lmcache/<model_safe>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>[<salt>].data
  //
  // 2) Legacy (backward compat):
  //    <high>:<low>
  //    Output: lmcache/kv_<high>_<low>

  // New format uses '@' as separator — detect first.
  if (key.find('@') != std::string::npos) {
    return key_to_path_fsformat(key);
  }

  std::string prefix = base_dir_ + "/";
  // Legacy "high:low" format
  auto colon = key.find(':');
  if (colon == std::string::npos) {
    return prefix + "kv_0000000000000000_" + key;
  }
  try {
    uint64_t high = std::stoull(key.substr(0, colon));
    uint64_t low  = std::stoull(key.substr(colon + 1));
    char buf[64];
    snprintf(buf, sizeof(buf), "%s/kv_%016lx_%016lx",
             base_dir_.c_str(), (unsigned long)high, (unsigned long)low);
    return std::string(buf);
  } catch (...) {
    return prefix + "kv_" + key;
  }
}

// ---------- FSConnector-compatible key_to_path ----------
static std::string replace_all(const std::string& s,
                                const std::string& from,
                                const std::string& to) {
  std::string result;
  result.reserve(s.size());
  size_t pos = 0;
  while (true) {
    size_t idx = s.find(from, pos);
    if (idx == std::string::npos) {
      result.append(s, pos, std::string::npos);
      break;
    }
    result.append(s, pos, idx - pos);
    result += to;
    pos = idx + from.size();
  }
  return result;
}

std::string Hf3fsConnector::key_to_path_fsformat(const std::string& key) const {
  // Parse "<model>@<rank_hex>@<obj_group_id_hex>@<hash_hex>[@<salt>]"
  // This matches the FSConnector serialization format for ObjectKey.
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == '@') {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  // Must have at least 4 fields: model, rank, obj_group_id, hash
  // Optional 5th field: cache_salt
  if (parts.size() < 4) {
    // Fallback to legacy format
    return base_dir_ + "/kv_" + key;
  }
  const std::string& model_name    = parts[0];
  const std::string& kv_rank_hex   = parts[1];
  const std::string& obj_group_id_hex = parts[2];
  const std::string& chunk_hash    = parts[3];
  const std::string& cache_salt    = parts.size() >= 5 ? parts[4] : std::string();

  // Sanitize model name: replace '/' with '-SEP-' for path safety
  std::string safe_model = replace_all(model_name, "/", "-SEP-");

  // Build path: lmcache/<model>@0x<rank>@<obj_group_id>@<hash>[<salt>].data
  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() +
                 obj_group_id_hex.size() + chunk_hash.size() +
                 cache_salt.size() + base_dir_.size() + 64);
  result += base_dir_;
  result += '/';
  result += safe_model;
  result += '@';
  result += "0x";
  result += kv_rank_hex;
  result += '@';
  result += obj_group_id_hex;
  result += '@';
  result += chunk_hash;
  if (!cache_salt.empty()) {
    result += '@';
    result += cache_salt;
  }
  result += ".data";
  return result;
}

// ---------- IOBuffer pool: avoid repeated RDMA memory registration ----------
// Reuses IOBuffer objects from the pool instead of allocating + RDMA-registering
// new ones each batch. The pool is per-connection (per-worker) and guarded by
// io_buf_pool_mu_.
static hf3fs::storage::client::IOBuffer* pool_get_or_create_io_buffer(
    Hf3fsConnection& conn, hf3fs::storage::client::StorageClient* sc,
    uint8_t* data, size_t size) {
  // 1. Try the pool first — clear stale data before reuse.
  std::lock_guard<std::mutex> pool_lock(conn.io_buf_pool_mu_);
  if (!conn.io_buf_pool_.empty()) {
    void* cached = conn.io_buf_pool_.back();
    conn.io_buf_pool_.pop_back();
    // Clear stale data from previous batch (RDMA read may not overwrite all bytes).
    std::memset(cached, 0, size);
    return static_cast<hf3fs::storage::client::IOBuffer*>(cached);
  }
  // 2. Pool is empty — allocate + RDMA-register.
  auto r = sc->registerIOBuffer(data, size);
  if (r.hasError()) return nullptr;
  return new hf3fs::storage::client::IOBuffer(std::move(*r));
}

static void pool_return_io_buffer(Hf3fsConnection& conn, hf3fs::storage::client::IOBuffer* ioBuf) {
  std::lock_guard<std::mutex> pool_lock(conn.io_buf_pool_mu_);
  conn.io_buf_pool_.push_back(ioBuf);
}

// Single-operation RDMA buffer registration (used by do_single_get/set).
static hf3fs::storage::client::IOBuffer* reg_buf(
    hf3fs::storage::client::StorageClient* sc, uint8_t* d, size_t s) {
  auto r = sc->registerIOBuffer(d, s);
  if (r.hasError()) return nullptr;
  return new hf3fs::storage::client::IOBuffer(std::move(*r));
}

// ---------- deref Expected ----------
template <typename T, typename E>
static const T& de(const folly::Expected<T, E>& x) { return *x; }

// ---------- Helper: convert meta::ChunkId to storage::ChunkId ----------
static hf3fs::storage::ChunkId meta_chunk_id_to_storage(
    const hf3fs::meta::ChunkId& mcId) {
  std::string packed = mcId.pack();
  return hf3fs::storage::ChunkId(std::move(packed));
}

// Creates a ChainRange layout using all available chains in the cluster.
// This ensures chunks are striped across all chains for optimal performance.
static hf3fs::meta::Layout build_lmc_layout(uint32_t chunk_size,
                                             uint32_t chain_table_id = 1) {
  // Use ChainTableId(chain_table_id) with routing info version for chain resolution
  // stripeSize = number of chains, baseChainIndex = 0
  // The Layout will be used by File::getChainId() to resolve chunk -> chain
  return hf3fs::meta::Layout::newChainRange(
    hf3fs::flat::ChainTableId(chain_table_id),
    hf3fs::flat::ChainTableVersion(0),
    chunk_size,
    1,  // stripeSize will be overwritten by MetaClient
    0);
}

// ---------- Helper: resolve file via MetaClient ----------
// Creates or opens a file, returns the Inode with Layout for chain resolution.
// Returns nullptr on failure.
static hf3fs::meta::Inode* resolve_file(
    hf3fs::meta::client::MetaClient* mc,
    const hf3fs::flat::UserInfo& ui,
    const std::string& file_path,
    bool create_if_missing,
    uint32_t file_permission = 0644) {
  if (!mc) return nullptr;

  hf3fs::Path path(file_path);
  hf3fs::meta::client::SessionId session = hf3fs::meta::client::SessionId::random();

  hf3fs::meta::Inode inode;

  // First try to open the file
  auto openRes = folly::coro::blockingWait(
      mc->open(ui, hf3fs::meta::InodeId::root(), path,
               session, static_cast<int>(hf3fs::meta::OpenFlags(O_RDWR))));
  if (openRes.hasError()) {
    // File doesn't exist, try to create it
    if (create_if_missing) {
      auto createRes = folly::coro::blockingWait(
          mc->create(ui, hf3fs::meta::InodeId::root(), path,
                     session, hf3fs::flat::Permission(file_permission),
                      static_cast<int>(hf3fs::meta::OpenFlags(O_RDWR | O_CREAT)),
                      std::nullopt));
      if (createRes.hasError()) {
        DBG("Failed to create file %s: %s",
            file_path.c_str(), createRes.error().describe().c_str());
        return nullptr;
      }
      inode = *createRes;
    } else {
      DBG("File not found: %s", file_path.c_str());
      return nullptr;
    }
  } else {
    inode = *openRes;
  }

  return new hf3fs::meta::Inode(inode);
}

// ---------- Helper: close file after read/write ----------
static void close_file(
    hf3fs::meta::client::MetaClient* mc,
    const hf3fs::flat::UserInfo& ui,
    hf3fs::meta::Inode* inode,
    hf3fs::meta::client::SessionId session,
    bool written) {
  if (!mc || !inode) return;
  auto closeRes = folly::coro::blockingWait(
      mc->close(ui, inode->id, session, /*read=*/ !written, /*written=*/ written));
  if (closeRes.hasError()) {
    DBG("Failed to close file: %s",
        closeRes.error().describe().c_str());
  }
  delete inode;
}

// ---------- create_connection ----------
Hf3fsConnection Hf3fsConnector::create_connection() {
  Hf3fsConnection conn;
  // Use the active store's 3FS layout chunk size for createReadIO/WriteIO's
  // chunkSize param (may differ from chunk_size_ when large keys are stored
  // as multiple striped chunks).
  conn.chunk_size = chunk_size_;
  if (active_ds_index_ < data_stores_.size()) {
    conn.chunk_size = data_stores_[active_ds_index_]->get_layout_chunk_size();
  }

  // Populate connection with shared clients for this worker
  conn.net_client = shared_net_client_.get();
  conn.mgmtd_client = shared_mgmtd_client_.get();
  conn.meta_client = shared_meta_client_.get();
  conn.storage_client = shared_storage_client_.get();

  // Store a noop-deleter shared_ptr to the StorageClient so it is NOT
  // destroyed when the connection goes out of scope (it has background threads
  // unsafe to shut down from worker threads).
  conn.storage_client_keepalive = std::shared_ptr<void>(
      const_cast<void*>(static_cast<const void*>(shared_storage_client_.get())),
      [](void*) {});

  // Cache routing info raw pointer (kept alive by routing_info_ shared_ptr)
  // Store the client::RoutingInfo* (not flat::RoutingInfo*) so that
  // do_batch_get can safely call ri->raw() in resolve_path.
  if (routing_info_) {
    conn.routing_info = const_cast<hf3fs::client::RoutingInfo*>(
        routing_info_.get());
  } else {
    conn.routing_info = nullptr;
  }

  return conn;
}

// ---------- do_batch_set (multi-file + offset, parallel writes) ----------
void Hf3fsConnector::do_batch_set(Hf3fsConnection& conn, const Request& req) {
  auto t_total_start = std::chrono::steady_clock::now();
  auto t_lookup_start = t_total_start;
  auto t_lookup_end = t_total_start;

  auto* sc = static_cast<hf3fs::storage::client::StorageClient*>(conn.storage_client);
  if (!sc) {
    for (size_t i = 0; i < req.keys.size(); ++i)
      req.batch->per_key_results[req.start_idx + i] = 0;
    return;
  }

  const hf3fs::flat::UserInfo& ui = *shared_user_info_;
  hf3fs::storage::client::WriteOptions wo;
  wo.set_enableChecksum(false);

  // Step 1: Resolve offsets using pre-computed active_ds_index_.
  // chunk_size_ is fixed at construction, active_ds_index_ never changes.
  uint32_t ds_idx = active_ds_index_;
  auto& ds = *data_stores_[ds_idx];
  
  // Store offsets directly (no need for KeyEntry struct since only one data_store).
  std::vector<std::pair<uint64_t, uint32_t>> offsets;
  offsets.reserve(req.keys.size());
  for (size_t i = 0; i < req.keys.size(); ++i) {
    offsets.push_back(ds.lookup_or_alloc(req.keys[i], req.buf_lens[i]));
  }
  // Mark the active data_store's offset_map as dirty for lazy persistence.
  if (!offsets.empty()) {
    ds.mark_meta_dirty();
    // Count this write batch toward the sync threshold so the per-batch
    // meta size sync can be throttled (see sync_if_dirty in do_batch_set).
    ds.inc_dirty_count();
  }
  for (size_t i = 0; i < req.keys.size(); ++i)
    req.batch->per_key_results[req.start_idx + i] = 1;
  t_lookup_end = std::chrono::steady_clock::now();

  auto t_buf_start = std::chrono::steady_clock::now();
  // Step 2: Allocate RDMA data buffer sized to chunk_size-aligned regions per key.
  // Using chunk_size (not INTERNAL_STRIDE) ensures the buffer layout matches
  // the 3FS chunk mapping, preventing data corruption for multi-chunk keys.
  // Reuse cached buffers (write cache first, then read cache).
  auto t_buf_compute_start = std::chrono::steady_clock::now();
  size_t ps = pgsz();
  size_t tot = 0;
  uint32_t cs = data_stores_[ds_idx]->get_layout_chunk_size();
  for (const auto& offset : offsets) {
    uint32_t size = offset.second;
    uint64_t pks_i = (static_cast<uint64_t>(size) + cs - 1) / cs * cs;
    tot += static_cast<size_t>(pks_i);
  }
  tot = (tot + ps - 1) / ps * ps;
  auto t_buf_compute_end = std::chrono::steady_clock::now();

  auto t_buf_lock_check_start = std::chrono::steady_clock::now();
  bool using_wr_cache = false;
  {
    std::lock_guard<std::mutex> lock(conn.io_buf_pool_mu_);
    if (conn.wr_buf_cache_ && conn.wr_buf_cache_size_ >= tot) {
      using_wr_cache = true;
    }
  }
  auto t_buf_lock_check_end = std::chrono::steady_clock::now();

  auto t_buf_alloc_rdma_start = t_buf_lock_check_end;
  auto t_buf_alloc_rdma_end = t_buf_alloc_rdma_start;
  auto t_buf_register_start = t_buf_alloc_rdma_end;
  auto t_buf_register_end = t_buf_register_start;
  auto t_buf_cache_store_start = t_buf_register_end;
  auto t_buf_cache_store_end = t_buf_cache_store_start;
  auto t_buf_lock_get_start = t_buf_lock_check_end;
  auto t_buf_lock_get_end = t_buf_lock_get_start;

  uint8_t* buf = nullptr;
  hf3fs::storage::client::IOBuffer* ioBuf = nullptr;
  if (using_wr_cache) {
    t_buf_lock_get_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(conn.io_buf_pool_mu_);
      buf = conn.wr_buf_cache_;
      ioBuf = static_cast<hf3fs::storage::client::IOBuffer*>(conn.wr_io_buf_cache_);
    }
    t_buf_lock_get_end = std::chrono::steady_clock::now();
  } else if (conn.rd_buf_cache_ && conn.rd_buf_cache_size_ >= tot) {
    t_buf_lock_get_start = std::chrono::steady_clock::now();
    buf = conn.rd_buf_cache_;
    ioBuf = static_cast<hf3fs::storage::client::IOBuffer*>(conn.rd_io_buf_cache_);
    t_buf_lock_get_end = std::chrono::steady_clock::now();
  } else {
    t_buf_alloc_rdma_start = std::chrono::steady_clock::now();
    buf = alloc_rdma(tot, /*zero_init=*/false);
    t_buf_alloc_rdma_end = std::chrono::steady_clock::now();
    if (!buf) {
      for (size_t i = 0; i < req.keys.size(); ++i)
        req.batch->per_key_results[req.start_idx + i] = 0;
      return;
    }
    t_buf_register_start = std::chrono::steady_clock::now();
    auto r = sc->registerIOBuffer(buf, tot);
    t_buf_register_end = std::chrono::steady_clock::now();
    if (r.hasError()) {
      free_rdma(buf);
      for (size_t i = 0; i < req.keys.size(); ++i)
        req.batch->per_key_results[req.start_idx + i] = 0;
      return;
    }
    ioBuf = new hf3fs::storage::client::IOBuffer(std::move(*r));
    // Cache for future writes (replace old write cache if exists).
    t_buf_cache_store_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(conn.io_buf_pool_mu_);
      if (conn.wr_buf_cache_ && conn.wr_buf_cache_ != buf) {
        // If rd_buf_cache_ shares the same buffer, clear it first to avoid double-free.
        if (conn.rd_buf_cache_ == conn.wr_buf_cache_) {
          conn.rd_buf_cache_ = nullptr;
          conn.rd_io_buf_cache_ = nullptr;
          conn.rd_buf_cache_size_ = 0;
        }
        free_rdma(conn.wr_buf_cache_);
        delete static_cast<hf3fs::storage::client::IOBuffer*>(conn.wr_io_buf_cache_);
      }
      conn.wr_buf_cache_ = buf;
      conn.wr_buf_cache_size_ = tot;
      conn.wr_io_buf_cache_ = ioBuf;
      // Also cache as read buffer if read cache doesn't exist or is smaller.
      // Skip if rd_buf_cache_ is the same buffer as new buf (avoid double free).
      if (conn.rd_buf_cache_ != buf && (!conn.rd_buf_cache_ || conn.rd_buf_cache_size_ < tot)) {
        if (conn.rd_buf_cache_) {
          free_rdma(conn.rd_buf_cache_);
          delete static_cast<hf3fs::storage::client::IOBuffer*>(conn.rd_io_buf_cache_);
        }
        conn.rd_buf_cache_ = buf;
        conn.rd_buf_cache_size_ = tot;
        conn.rd_io_buf_cache_ = ioBuf;
      }
    }
    t_buf_cache_store_end = std::chrono::steady_clock::now();
  }
  conn.io_buffers_.push_back(ioBuf);
  auto t_buf_end = std::chrono::steady_clock::now();

  auto ns_sub = [](std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) -> uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };
  uint64_t d_buf_compute  = ns_sub(t_buf_compute_start, t_buf_compute_end);
  uint64_t d_buf_lock_chk = ns_sub(t_buf_lock_check_start, t_buf_lock_check_end);
  uint64_t d_buf_lock_get = ns_sub(t_buf_lock_get_start, t_buf_lock_get_end);
  uint64_t d_buf_alloc    = ns_sub(t_buf_alloc_rdma_start, t_buf_alloc_rdma_end);
  uint64_t d_buf_register = ns_sub(t_buf_register_start, t_buf_register_end);
  uint64_t d_buf_cache    = ns_sub(t_buf_cache_store_start, t_buf_cache_store_end);
  bool was_cache_hit = using_wr_cache ||
      (!d_buf_alloc && !d_buf_register && !d_buf_cache);
  conn.set_timing_.ns_buf_compute.fetch_add(d_buf_compute, std::memory_order_relaxed);
  conn.set_timing_.ns_buf_lock_check.fetch_add(d_buf_lock_chk, std::memory_order_relaxed);
  conn.set_timing_.ns_buf_lock_get.fetch_add(d_buf_lock_get, std::memory_order_relaxed);
  conn.set_timing_.ns_buf_alloc_rdma.fetch_add(d_buf_alloc, std::memory_order_relaxed);
  conn.set_timing_.ns_buf_register.fetch_add(d_buf_register, std::memory_order_relaxed);
  conn.set_timing_.ns_buf_cache_store.fetch_add(d_buf_cache, std::memory_order_relaxed);
  if (was_cache_hit) {
    conn.set_timing_.ns_buf_cache_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    conn.set_timing_.ns_buf_cache_misses.fetch_add(1, std::memory_order_relaxed);
  }
  set_timing_.ns_buf_compute.fetch_add(d_buf_compute, std::memory_order_relaxed);
  set_timing_.ns_buf_lock_check.fetch_add(d_buf_lock_chk, std::memory_order_relaxed);
  set_timing_.ns_buf_lock_get.fetch_add(d_buf_lock_get, std::memory_order_relaxed);
  set_timing_.ns_buf_alloc_rdma.fetch_add(d_buf_alloc, std::memory_order_relaxed);
  set_timing_.ns_buf_register.fetch_add(d_buf_register, std::memory_order_relaxed);
  set_timing_.ns_buf_cache_store.fetch_add(d_buf_cache, std::memory_order_relaxed);
  if (was_cache_hit) {
    set_timing_.ns_buf_cache_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    set_timing_.ns_buf_cache_misses.fetch_add(1, std::memory_order_relaxed);
  }

  // Copy data from Python buffers to our local buffer.
  // Buffer is stride-aligned; data is placed at the start of each key's
  // stride-aligned region. Padding bytes between keys are left as whatever
  // the buffer contained (the WriteIOs below only write chunkLen bytes from
  // the valid data portion, not the full stride).
  // Optimization: prefetch next key's source data to hide Python buffer latency.
  // Prefetch is a hint — invalid addresses are safely ignored by the CPU.
  size_t srcOffset = 0;
  const size_t numKeys = req.keys.size();
  for (size_t i = 0; i < numKeys; ++i) {
    // Prefetch next key's source data before copying current key.
    // Skip invalid keys — prefetch on NULL/invalid ptr is a no-op.
    if (i + 1 < numKeys &&
        req.batch->per_key_results[req.start_idx + i + 1] != 0) {
      const uint8_t* nextSrc = static_cast<const uint8_t*>(req.buf_ptrs[i + 1]);
      __builtin_prefetch(nextSrc, 0, 1);  // Read, L2
    }

    if (req.batch->per_key_results[req.start_idx + i] == 0) continue;

    const uint8_t* src = static_cast<const uint8_t*>(req.buf_ptrs[i]);
    std::memcpy(buf + srcOffset, src, req.buf_lens[i]);
    uint32_t cs = data_stores_[ds_idx]->get_layout_chunk_size();
    srcOffset += (static_cast<uint64_t>(req.buf_lens[i]) + cs - 1) / cs * cs;
  }
  auto t_memcpy_end = std::chrono::steady_clock::now();

  // Step 3: Group WriteIOs by (data_store, chain) for parallel chain-
  // aggregated batchWrite. All keys map to the same active data_store
  // (active_ds_index_).
  struct ChainWriteBatch {
    std::vector<hf3fs::storage::client::WriteIO> writeIOs;
  };
  std::map<std::pair<uint32_t, uint32_t>, ChainWriteBatch> chainWriteBatches;
  srcOffset = 0;
  for (size_t i = 0; i < offsets.size(); ++i) {
    uint64_t offset = offsets[i].first;
    uint32_t stored_size = offsets[i].second;
    size_t actual_len = req.buf_lens[i];

    auto& ds = *data_stores_[ds_idx];
    uint64_t numChunks = (static_cast<uint64_t>(stored_size) + cs - 1) / cs;

    auto& chainIndexList = ds.get_chain_index_list();
    uint32_t stripe = ds.get_stripe_size();
    
    DBG("do_batch_set: key=%s offset=%lu stored_size=%u numChunks=%lu stripe=%u chainIndexList.size()=%zu",
        req.keys[i].c_str(), (unsigned long)offset, stored_size, (unsigned long)numChunks, stripe, chainIndexList.size());

    // Cache chainId and basePacked once per file (all chunks share same inode & layout)
    uint32_t cachedChainIdRaw = ds.get_chain_id_raw();
    char basePackedBuf[16];
    std::memcpy(basePackedBuf, hf3fs::meta::ChunkId(ds.get_inode_id(), 0, 0).pack().data(), 16);

    // Use pre-built chain_index->chainId map (bypasses broken getChainId which
    // requires chainTables that are empty in mgmtd routing info).
    const auto& chainIdMap = ds.get_chain_id_map();

    for (uint64_t c = 0; c < numChunks; ++c) {
      // chunk_index uses chunk_size (3FS layout), while stride (1MB) controls buffer layout
      uint64_t chunkIndex = (offset + c * cs) / cs;
      uint64_t stripeIdx = chunkIndex % stripe;
      uint32_t chainIndex = chainIndexList[stripeIdx];
      // Look up chainId from pre-built map; fall back to cachedChainIdRaw.
      auto mapIt = chainIdMap.find(chainIndex);
      uint32_t chainIdRaw = (mapIt != chainIdMap.end()) ? mapIt->second : cachedChainIdRaw;
      
      DBG("do_batch_set: chunk=%lu/%lu chainIndex=%u chainId=%u (from=%s)",
          c, numChunks, chainIndex, chainIdRaw,
          mapIt != chainIdMap.end() ? "map" : "fallback");

      uint64_t chunkOffset = offset + c * cs;
      uint32_t ci = static_cast<uint32_t>(chunkOffset / cs);
      std::memcpy(basePackedBuf + 12, &ci, 4);
      hf3fs::storage::ChunkId chunkId(std::string_view(basePackedBuf, 16));
      // chunkLen must not exceed actual data length — avoid writing
      // uninitialized padding bytes to 3FS.
      uint64_t bytesWrittenSoFar = c * cs;
      uint64_t remaining = (bytesWrittenSoFar < actual_len)
          ? (actual_len - bytesWrittenSoFar) : 0;
      uint32_t chunkLen = static_cast<uint32_t>(std::min(
          static_cast<uint64_t>(cs), remaining));

      uint32_t ds_key = ds_idx;
      chainWriteBatches[{ds_key, chainIdRaw}].writeIOs.push_back(sc->createWriteIO(
          hf3fs::flat::ChainId(chainIdRaw), chunkId,
          static_cast<uint32_t>(chunkOffset % cs),
          chunkLen, conn.chunk_size,
          buf + srcOffset + c * cs, ioBuf));
    }
    srcOffset += (static_cast<uint64_t>(actual_len) + cs - 1) / cs * cs;
  }
  auto t_create_io_end = std::chrono::steady_clock::now();

  // Refresh routing info before batchWrite (only if version changed).
  {
    auto ri = shared_mgmtd_client_->getRoutingInfo();
    if (ri && ri->raw()) {
      auto version = ri->raw()->routingInfoVersion.toUnderType();
      if (version != cached_routing_version_) {
        if (version == 0) {
          DBG("do_batch_set: stale routing version 0, refreshing...");
        } else {
          DBG("do_batch_set: routing version changed %u -> %u, refreshing...",
              cached_routing_version_, version);
        }
        auto rr = folly::coro::blockingWait(shared_mgmtd_client_->refreshRoutingInfo(true));
        if (rr.hasError()) {
          DBG("do_batch_set: routing refresh failed: %s",
              std::string(rr.error().message()).c_str());
        } else {
          cached_routing_version_ = version;
        }
      }
    } else {
      DBG("do_batch_set: routing info null, refreshing...");
      auto rr = folly::coro::blockingWait(shared_mgmtd_client_->refreshRoutingInfo(true));
      if (rr.hasError()) {
        DBG("do_batch_set: routing refresh failed: %s",
            std::string(rr.error().message()).c_str());
      }
    }
  }

  // Step 4: Parallel chain-aggregated batchWrite with intra-chain
  // sub-batching. Each chain's WriteIOs are split into 2 sub-batches so that
  // concurrent RDMA WRs on the same chain can be processed in parallel,
  // doubling intra-chain throughput.
  bool writeOk = true;
  size_t total_wios = 0;
  for (const auto& [_, batch] : chainWriteBatches) {
    total_wios += batch.writeIOs.size();
  }
  DBG("do_batch_set: entering batchWrite, total_wios=%zu", total_wios);
  if (total_wios > 0) {
    try {
      std::vector<folly::coro::Task<hf3fs::Result<hf3fs::Void>>> tasks;
      for (auto& [fileChainKey, batch] : chainWriteBatches) {
        auto& writeIOs = batch.writeIOs;
        if (writeIOs.empty()) continue;
        // Split into 2 sub-batches for intra-chain concurrency.
        size_t half = writeIOs.size() / 2;
        if (half == 0) half = 1;
        std::span writeSpan1(writeIOs.data(), half);
        std::span writeSpan2(writeIOs.data() + half, writeIOs.size() - half);
        DBG("do_batch_set: batchWrite chain=%u ios=%zu sub1=%zu sub2=%zu",
            fileChainKey.second, writeIOs.size(), half, writeIOs.size() - half);
        tasks.push_back(sc->batchWrite(writeSpan1, ui, wo));
        if (writeSpan2.size() > 0) {
          tasks.push_back(sc->batchWrite(writeSpan2, ui, wo));
        }
      }
      auto results = folly::coro::blockingWait(
          folly::coro::collectAllRange(std::move(tasks)));
      for (auto& res : results) {
        if (res.hasError()) {
          writeOk = false;
          break;
        }
      }
      DBG("do_batch_set: batchWrite all done writeOk=%d", (int)writeOk);
    } catch (...) {
      writeOk = false;
    }
  }
  auto t_batch_io_end = std::chrono::steady_clock::now();

  if (writeOk) {
    data_stores_[ds_idx]->sync_if_dirty(shared_meta_client_.get(), ui, true,
                                        32);
  }
  auto t_sync_end = std::chrono::steady_clock::now();

  if (!writeOk) {
    for (size_t i = 0; i < req.keys.size(); ++i)
      req.batch->per_key_results[req.start_idx + i] = 0;
  }

  // No cleanup: buf is wr_buf_cache_ (reused for future writes).
  // Freeing here causes use-after-free when next batch reuses wr_buf_cache_.

  // Accumulate timing stats (atomic, thread-safe across workers).
  auto ns = [](std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) -> uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };
  uint64_t d_lookup = ns(t_total_start, t_lookup_end);
  uint64_t d_buf = ns(t_lookup_end, t_buf_end);
  uint64_t d_memcpy = ns(t_buf_end, t_memcpy_end);
  uint64_t d_create = ns(t_memcpy_end, t_create_io_end);
  uint64_t d_batch = ns(t_create_io_end, t_batch_io_end);
  uint64_t d_sync = ns(t_batch_io_end, t_sync_end);
  uint64_t d_total = ns(t_total_start, t_sync_end);
  conn.set_timing_.ns_lookup.fetch_add(d_lookup, std::memory_order_relaxed);
  conn.set_timing_.ns_buf_alloc.fetch_add(d_buf, std::memory_order_relaxed);
  conn.set_timing_.ns_memcpy.fetch_add(d_memcpy, std::memory_order_relaxed);
  conn.set_timing_.ns_create_io.fetch_add(d_create, std::memory_order_relaxed);
  conn.set_timing_.ns_batch_io.fetch_add(d_batch, std::memory_order_relaxed);
  conn.set_timing_.ns_sync.fetch_add(d_sync, std::memory_order_relaxed);
  conn.set_timing_.ns_total.fetch_add(d_total, std::memory_order_relaxed);
  conn.set_timing_.count.fetch_add(1, std::memory_order_relaxed);
  set_timing_.ns_lookup.fetch_add(d_lookup, std::memory_order_relaxed);
  set_timing_.ns_buf_alloc.fetch_add(d_buf, std::memory_order_relaxed);
  set_timing_.ns_memcpy.fetch_add(d_memcpy, std::memory_order_relaxed);
  set_timing_.ns_create_io.fetch_add(d_create, std::memory_order_relaxed);
  set_timing_.ns_batch_io.fetch_add(d_batch, std::memory_order_relaxed);
  set_timing_.ns_sync.fetch_add(d_sync, std::memory_order_relaxed);
  set_timing_.ns_total.fetch_add(d_total, std::memory_order_relaxed);
  set_timing_.count.fetch_add(1, std::memory_order_relaxed);
}

// ---------- do_batch_get (multi-file + offset, parallel reads) ----------
void Hf3fsConnector::do_batch_get(Hf3fsConnection& conn, const Request& req) {
  auto t_total_start = std::chrono::steady_clock::now();

  auto* sc = static_cast<hf3fs::storage::client::StorageClient*>(conn.storage_client);
  if (!sc) {
    for (size_t i = 0; i < req.keys.size(); ++i)
      req.batch->per_key_results[req.start_idx + i] = 0;
    return;
  }

  const hf3fs::flat::UserInfo& ui = *shared_user_info_;
  hf3fs::storage::client::ReadOptions ro;
  ro.set_enableChecksum(false);

  // Step 1: Resolve offsets using pre-computed active_ds_index_.
  uint32_t ds_idx = active_ds_index_;
  auto& ds = *data_stores_[ds_idx];
  
  // Store (key_idx, offset) pairs for found keys only.
  std::vector<std::pair<size_t, std::pair<uint64_t, uint32_t>>> found;
  found.reserve(req.keys.size());
  for (size_t i = 0; i < req.keys.size(); ++i) {
    auto result = ds.lookup_offset(req.keys[i]);
    if (result.second > 0) {
      found.push_back({i, result});
      req.batch->per_key_results[req.start_idx + i] = 1;
    } else {
      req.batch->per_key_results[req.start_idx + i] = 0;
    }
  }
  DBG("do_batch_get: found.size=%zu, empty=%d", found.size(), found.empty());
  auto t_lookup_end = std::chrono::steady_clock::now();

  if (found.empty()) return;

  // Step 2: Allocate data buffer sized to chunk_size-aligned regions per key.
  // Must use chunk_size (same as do_batch_set) to ensure internal RDMA buffer
  // offsets are consistent between store and load.
  // Reuse cached buffers (write cache first, then read cache).
  auto t_buf_compute_start = std::chrono::steady_clock::now();
  size_t ps = pgsz();
  size_t tot = 0;
  uint32_t cs = data_stores_[ds_idx]->get_layout_chunk_size();
  for (const auto& f : found) {
    uint32_t size = f.second.second;
    uint64_t pks_i = (static_cast<uint64_t>(size) + cs - 1) / cs * cs;
    tot += static_cast<size_t>(pks_i);
  }
  tot = (tot + ps - 1) / ps * ps;
  auto t_buf_compute_end = std::chrono::steady_clock::now();

  // Check write buffer cache first (may have been cached by previous batch_set).
  auto t_buf_lock_check_start = std::chrono::steady_clock::now();
  bool using_wr_cache = false;
  {
    std::lock_guard<std::mutex> lock(conn.io_buf_pool_mu_);
    if (conn.wr_buf_cache_ && conn.wr_buf_cache_size_ >= tot) {
      using_wr_cache = true;
    }
  }
  auto t_buf_lock_check_end = std::chrono::steady_clock::now();

  auto t_buf_alloc_rdma_start = t_buf_lock_check_end;
  auto t_buf_alloc_rdma_end = t_buf_alloc_rdma_start;
  auto t_buf_register_start = t_buf_alloc_rdma_end;
  auto t_buf_register_end = t_buf_register_start;
  auto t_buf_cache_store_start = t_buf_register_end;
  auto t_buf_cache_store_end = t_buf_cache_store_start;
  auto t_buf_lock_get_start = t_buf_lock_check_end;
  auto t_buf_lock_get_end = t_buf_lock_get_start;

  uint8_t* buf = nullptr;
  hf3fs::storage::client::IOBuffer* ioBuf = nullptr;
  if (using_wr_cache) {
    t_buf_lock_get_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(conn.io_buf_pool_mu_);
      buf = conn.wr_buf_cache_;
      ioBuf = static_cast<hf3fs::storage::client::IOBuffer*>(conn.wr_io_buf_cache_);
    }
    t_buf_lock_get_end = std::chrono::steady_clock::now();
  } else if (conn.rd_buf_cache_ && conn.rd_buf_cache_size_ >= tot) {
    t_buf_lock_get_start = std::chrono::steady_clock::now();
    buf = conn.rd_buf_cache_;
    ioBuf = static_cast<hf3fs::storage::client::IOBuffer*>(conn.rd_io_buf_cache_);
    t_buf_lock_get_end = std::chrono::steady_clock::now();
  } else {
    t_buf_alloc_rdma_start = std::chrono::steady_clock::now();
    buf = alloc_rdma(tot, /*zero_init=*/false);
    t_buf_alloc_rdma_end = std::chrono::steady_clock::now();
    if (!buf) {
      for (size_t i = 0; i < req.keys.size(); ++i)
        req.batch->per_key_results[req.start_idx + i] = 0;
      return;
    }
    t_buf_register_start = std::chrono::steady_clock::now();
    auto r = sc->registerIOBuffer(buf, tot);
    t_buf_register_end = std::chrono::steady_clock::now();
    if (r.hasError()) {
      free_rdma(buf);
      for (size_t i = 0; i < req.keys.size(); ++i)
        req.batch->per_key_results[req.start_idx + i] = 0;
      return;
    }
    ioBuf = new hf3fs::storage::client::IOBuffer(std::move(*r));
    // Cache as write buffer (replacing old write cache if smaller).
    t_buf_cache_store_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(conn.io_buf_pool_mu_);
      if (conn.wr_buf_cache_ && conn.wr_buf_cache_ != buf) {
        // If rd_buf_cache_ shares the same buffer, clear it first to avoid double-free.
        if (conn.rd_buf_cache_ == conn.wr_buf_cache_) {
          conn.rd_buf_cache_ = nullptr;
          conn.rd_io_buf_cache_ = nullptr;
          conn.rd_buf_cache_size_ = 0;
        }
        free_rdma(conn.wr_buf_cache_);
        delete static_cast<hf3fs::storage::client::IOBuffer*>(conn.wr_io_buf_cache_);
      }
      conn.wr_buf_cache_ = buf;
      conn.wr_buf_cache_size_ = tot;
      conn.wr_io_buf_cache_ = ioBuf;
      // Cache as read buffer (replacing old read cache if smaller).
      // Skip if rd_buf_cache_ is the same buffer as new buf (avoid double free).
      if (conn.rd_buf_cache_ != buf) {
        if (conn.rd_buf_cache_) {
          free_rdma(conn.rd_buf_cache_);
          delete static_cast<hf3fs::storage::client::IOBuffer*>(conn.rd_io_buf_cache_);
        }
        conn.rd_buf_cache_ = buf;
        conn.rd_buf_cache_size_ = tot;
        conn.rd_io_buf_cache_ = ioBuf;
      }
    }
    t_buf_cache_store_end = std::chrono::steady_clock::now();
  }
  conn.io_buffers_.push_back(ioBuf);
  auto t_buf_end = std::chrono::steady_clock::now();

  auto ns_sub = [](std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) -> uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };
  uint64_t d_buf_compute  = ns_sub(t_buf_compute_start, t_buf_compute_end);
  uint64_t d_buf_lock_chk = ns_sub(t_buf_lock_check_start, t_buf_lock_check_end);
  uint64_t d_buf_lock_get = ns_sub(t_buf_lock_get_start, t_buf_lock_get_end);
  uint64_t d_buf_alloc    = ns_sub(t_buf_alloc_rdma_start, t_buf_alloc_rdma_end);
  uint64_t d_buf_register = ns_sub(t_buf_register_start, t_buf_register_end);
  uint64_t d_buf_cache    = ns_sub(t_buf_cache_store_start, t_buf_cache_store_end);
  bool was_cache_hit = using_wr_cache ||
      (!d_buf_alloc && !d_buf_register && !d_buf_cache);
  conn.get_timing_.ns_buf_compute.fetch_add(d_buf_compute, std::memory_order_relaxed);
  conn.get_timing_.ns_buf_lock_check.fetch_add(d_buf_lock_chk, std::memory_order_relaxed);
  conn.get_timing_.ns_buf_lock_get.fetch_add(d_buf_lock_get, std::memory_order_relaxed);
  conn.get_timing_.ns_buf_alloc_rdma.fetch_add(d_buf_alloc, std::memory_order_relaxed);
  conn.get_timing_.ns_buf_register.fetch_add(d_buf_register, std::memory_order_relaxed);
  conn.get_timing_.ns_buf_cache_store.fetch_add(d_buf_cache, std::memory_order_relaxed);
  if (was_cache_hit) {
    conn.get_timing_.ns_buf_cache_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    conn.get_timing_.ns_buf_cache_misses.fetch_add(1, std::memory_order_relaxed);
  }
  get_timing_.ns_buf_compute.fetch_add(d_buf_compute, std::memory_order_relaxed);
  get_timing_.ns_buf_lock_check.fetch_add(d_buf_lock_chk, std::memory_order_relaxed);
  get_timing_.ns_buf_lock_get.fetch_add(d_buf_lock_get, std::memory_order_relaxed);
  get_timing_.ns_buf_alloc_rdma.fetch_add(d_buf_alloc, std::memory_order_relaxed);
  get_timing_.ns_buf_register.fetch_add(d_buf_register, std::memory_order_relaxed);
  get_timing_.ns_buf_cache_store.fetch_add(d_buf_cache, std::memory_order_relaxed);
  if (was_cache_hit) {
    get_timing_.ns_buf_cache_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    get_timing_.ns_buf_cache_misses.fetch_add(1, std::memory_order_relaxed);
  }

  // Step 3: Collect ReadIOs and group by (file_index, chainIdRaw) for chain-
  // aggregated batchRead. This maximizes RDMA bandwidth by executing
  // batchReads for different chains concurrently.
  struct ChainReadBatch {
    std::vector<hf3fs::storage::client::ReadIO> readIOs;
  };
  std::map<std::pair<uint32_t, uint32_t>, ChainReadBatch> chainBatches;
  size_t offsetAcc = 0;
  auto t_create_io_real_start = std::chrono::steady_clock::now();
  for (const auto& f : found) {
    uint64_t offset_val = f.second.first;
    uint32_t size = f.second.second;

    uint64_t numChunks = (static_cast<uint64_t>(size) + cs - 1) / cs;

    auto& chainIndexList = ds.get_chain_index_list();
    uint32_t stripe = ds.get_stripe_size();

    // Cache chainId→chainIdRaw mapping per unique chainIndex for this file
    uint32_t cachedChainIdRaw = ds.get_chain_id_raw();
    char basePackedBuf[16];
    std::memcpy(basePackedBuf, hf3fs::meta::ChunkId(ds.get_inode_id(), 0, 0).pack().data(), 16);
    // Use cached chainId map (built once per DataStore)
    const auto& chainIndexToId = ds.get_chain_id_map();

    uint32_t chunkAcc = 0;
    for (uint64_t c = 0; c < numChunks; ++c) {
      // chunk_index uses chunk_size (3FS layout), stride (1MB) controls buffer layout
      uint64_t chunkIndex = (offset_val + c * cs) / cs;
      uint64_t stripeIdx = chunkIndex % stripe;
      uint32_t chainIndex = chainIndexList[stripeIdx];
      auto it = chainIndexToId.find(chainIndex);
      uint32_t chainIdRaw = (it != chainIndexToId.end()) ? it->second : cachedChainIdRaw;

      uint32_t ci = static_cast<uint32_t>(chunkIndex);
      std::memcpy(basePackedBuf + 12, &ci, 4);
      hf3fs::storage::ChunkId chunkId(std::string_view(basePackedBuf, 16));
      uint32_t chunkLen = static_cast<uint32_t>(std::min(
          static_cast<uint64_t>(cs),
          static_cast<uint64_t>(size) - c * cs));

      // Split large chunkLen into sub-IOs no larger than the client-side
      // max_batch_bytes_ (default 32MB, NOT a hardcoded server limit) so
      // each readIO fits in one batchRead and a 32MB key is 1 readIO rather
      // than 8 x 4MB RDMA reads (8x per-key request overhead that capped
      // 32MB Load). Must match traffic_control().read().max_batch_bytes().
      const uint32_t SUB_IO_SIZE = static_cast<uint32_t>(max_batch_bytes_);
      uint32_t subOffset = 0;
      uint32_t subAcc = 0;
      while (subOffset < chunkLen) {
        uint32_t subLen = std::min(SUB_IO_SIZE, chunkLen - subOffset);

        auto subKey = std::make_pair(ds_idx, chainIdRaw);
        chainBatches[subKey].readIOs.push_back(sc->createReadIO(
            hf3fs::flat::ChainId(chainIdRaw), chunkId,
            static_cast<uint32_t>((offset_val + c * cs) % cs + subOffset),
            subLen,
            buf + offsetAcc + chunkAcc + subAcc, ioBuf));
        subAcc += subLen;
        subOffset += subLen;
      }
      chunkAcc += (static_cast<uint64_t>(chunkLen) + cs - 1) / cs * cs;
    }
    offsetAcc += (static_cast<uint64_t>(size) + cs - 1) / cs * cs;
  }
  auto t_create_io_end = std::chrono::steady_clock::now();

  // Refresh routing info only if version changed (avoid per-batch RPC).
  {
    auto ri = shared_mgmtd_client_->getRoutingInfo();
    if (ri && ri->raw()) {
      auto version = ri->raw()->routingInfoVersion.toUnderType();
      if (version != cached_routing_version_) {
        if (version == 0) {
          DBG("do_batch_get: stale routing version 0, refreshing...");
        } else {
          DBG("do_batch_get: routing version changed %u -> %u, refreshing...",
              cached_routing_version_, version);
        }
        auto rr = folly::coro::blockingWait(shared_mgmtd_client_->refreshRoutingInfo(true));
        if (rr.hasError()) {
          DBG("do_batch_get: routing refresh failed: %s",
              std::string(rr.error().message()).c_str());
        } else {
          cached_routing_version_ = version;
        }
      }
    } else {
      DBG("do_batch_get: routing info null, refreshing...");
      auto rr = folly::coro::blockingWait(shared_mgmtd_client_->refreshRoutingInfo(true));
      if (rr.hasError()) {
        DBG("do_batch_get: routing refresh failed: %s",
            std::string(rr.error().message()).c_str());
      }
    }
  }

// Step 4: Parallel chain-aggregated batchRead with intra-chain sub-batching.
  // Each chain's ReadIOs are split into N sub-batches so that concurrent
  // RDMA WRs on the same chain can be processed in parallel, improving
  // intra-chain throughput. collectAllRange issues all batchRead calls
  // concurrently across chains and waits for all.
  bool readOk = true;
  try {
    std::vector<folly::coro::Task<hf3fs::Result<hf3fs::Void>>> tasks;
    static constexpr int SUB_BATCH_COUNT = 4; // Try 4 sub-batches for better parallelism
    for (auto& [fileChainKey, batch] : chainBatches) {
      auto& readIOs = batch.readIOs;
      if (readIOs.empty()) continue;
      // Split into SUB_BATCH_COUNT sub-batches for intra-chain concurrency.
      size_t sub_size = readIOs.size() / SUB_BATCH_COUNT;
      if (sub_size == 0) sub_size = 1;
      size_t offset = 0;
      for (int i = 0; i < SUB_BATCH_COUNT && offset < readIOs.size(); ++i) {
        size_t size = std::min(sub_size, readIOs.size() - offset);
        std::span subSpan(readIOs.data() + offset, size);
        DBG("do_batch_get: batchRead chain=%u ios=%zu sub_batch=%d size=%zu",
            fileChainKey.second, readIOs.size(), i, size);
        tasks.push_back(sc->batchRead(subSpan, ui, ro));
        offset += size;
      }
    }
    auto results = folly::coro::blockingWait(
        folly::coro::collectAllRange(std::move(tasks)));
    for (auto& res : results) {
      if (res.hasError()) {
        readOk = false;
        break;
      }
    }
  } catch (...) {
    readOk = false;
  }
  auto t_batch_io_end = std::chrono::steady_clock::now();

  if (readOk) {
    // Parallel copy buffer -> Python. Key regions are independent (distinct
    // buf offsets + distinct destinations); copy was ~32% of Load time for
    // large keys. srcOff precomputed so threads read the same regions as the
    // original sequential copy.
    const uint32_t cs = data_stores_[ds_idx]->get_layout_chunk_size();
    const size_t numFound = found.size();

    std::vector<size_t> srcOff(numFound);
    size_t off = 0;
    for (size_t e = 0; e < numFound; ++e) {
      srcOff[e] = off;
      const uint32_t size = found[e].second.second;
      off += (static_cast<uint64_t>(size) + cs - 1) / cs * cs;
    }

    if (numFound > 1) {
      unsigned hw = std::thread::hardware_concurrency();
      if (hw == 0) hw = 2;
      const unsigned nthreads = std::min<unsigned>(8u, std::min(hw, (unsigned)numFound));
      std::vector<std::thread> threads;
      threads.reserve(nthreads - 1);
      for (unsigned t = 1; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
          for (size_t e = t; e < numFound; e += nthreads) {
            const size_t i = found[e].first;
            std::memcpy(req.buf_ptrs[i], buf + srcOff[e], req.buf_lens[i]);
          }
        });
      }
      for (size_t e = 0; e < numFound; e += nthreads) {
        const size_t i = found[e].first;
        std::memcpy(req.buf_ptrs[i], buf + srcOff[e], req.buf_lens[i]);
      }
      for (auto& th : threads) th.join();
    } else if (numFound == 1) {
      const size_t i = found[0].first;
      std::memcpy(req.buf_ptrs[i], buf, req.buf_lens[i]);
    }
  } else {
    for (size_t i = 0; i < req.keys.size(); ++i)
      req.batch->per_key_results[req.start_idx + i] = 0;
  }
  auto t_memcpy_end = std::chrono::steady_clock::now();

  // No cleanup: buf is wr_buf_cache_ (reused for future writes).
  // Freeing here causes use-after-free when next batch reuses wr_buf_cache_.

  // Accumulate timing stats (atomic, thread-safe across workers).
  auto ns = [](std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) -> uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };
  uint64_t d_lookup = ns(t_total_start, t_lookup_end);
  uint64_t d_buf = ns(t_lookup_end, t_buf_end);
  uint64_t d_create = ns(t_buf_end, t_create_io_end);
  uint64_t d_batch = ns(t_create_io_end, t_batch_io_end);
  uint64_t d_memcpy = ns(t_batch_io_end, t_memcpy_end);
  uint64_t d_total = ns(t_total_start, t_memcpy_end);
  conn.get_timing_.ns_lookup.fetch_add(d_lookup, std::memory_order_relaxed);
  conn.get_timing_.ns_buf_alloc.fetch_add(d_buf, std::memory_order_relaxed);
  conn.get_timing_.ns_create_io.fetch_add(d_create, std::memory_order_relaxed);
  conn.get_timing_.ns_batch_io.fetch_add(d_batch, std::memory_order_relaxed);
  conn.get_timing_.ns_memcpy.fetch_add(d_memcpy, std::memory_order_relaxed);
  conn.get_timing_.ns_total.fetch_add(d_total, std::memory_order_relaxed);
  conn.get_timing_.count.fetch_add(1, std::memory_order_relaxed);
  get_timing_.ns_lookup.fetch_add(d_lookup, std::memory_order_relaxed);
  get_timing_.ns_buf_alloc.fetch_add(d_buf, std::memory_order_relaxed);
  get_timing_.ns_create_io.fetch_add(d_create, std::memory_order_relaxed);
  get_timing_.ns_batch_io.fetch_add(d_batch, std::memory_order_relaxed);
  get_timing_.ns_memcpy.fetch_add(d_memcpy, std::memory_order_relaxed);
  get_timing_.ns_total.fetch_add(d_total, std::memory_order_relaxed);
  get_timing_.count.fetch_add(1, std::memory_order_relaxed);
}

// ---------- do_batch_exists (offset-map based, consistent with do_batch_set) ----------
void Hf3fsConnector::do_batch_exists(Hf3fsConnection& conn, const Request& req) {
  const size_t num_keys = req.keys.size();

  if (num_keys == 0) {
    return;
  }

  // Use pre-computed active_ds_index_ (chunk_size_ fixed at construction).
  DataStore& ds = *data_stores_[active_ds_index_];

  auto t_total_start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < num_keys; ++i) {
    auto t_key_start = std::chrono::steady_clock::now();
    req.batch->per_key_results[req.start_idx + i] = ds.contains(req.keys[i]) ? 1 : 0;
    auto t_key_end = std::chrono::steady_clock::now();
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_key_end - t_key_start).count();
    conn.exists_timing_.ns_lookup.fetch_add(ns, std::memory_order_relaxed);
    exists_timing_.ns_lookup.fetch_add(ns, std::memory_order_relaxed);
  }

  auto t_total_end = std::chrono::steady_clock::now();
  uint64_t d_total = std::chrono::duration_cast<std::chrono::nanoseconds>(t_total_end - t_total_start).count();
  conn.exists_timing_.ns_total.fetch_add(d_total, std::memory_order_relaxed);
  exists_timing_.ns_total.fetch_add(d_total, std::memory_order_relaxed);
  conn.exists_timing_.count.fetch_add(1, std::memory_order_relaxed);
  exists_timing_.count.fetch_add(1, std::memory_order_relaxed);
}

// ---------- do_batch_delete (multi-file) ----------
void Hf3fsConnector::do_batch_delete(Hf3fsConnection& conn, const Request& req) {
  for (size_t i = 0; i < req.keys.size(); ++i) {
    // Try to remove from all size-based data stores
    bool removed = false;
    for (auto& ds : data_stores_) {
      if (ds->remove_offset(req.keys[i])) {
        removed = true;
        break;
      }
    }
    req.batch->per_key_results[req.start_idx + i] = removed ? 1 : 0;
  }
}

// ---------- do_single_get ----------
void Hf3fsConnector::do_single_get(Hf3fsConnection& conn, const std::string& key,
    void* buf, size_t len, size_t cs) {
  DBG("GET-START key=%s len=%zu cs=%zu", key.c_str(), len, cs);
  auto* sc = static_cast<hf3fs::storage::client::StorageClient*>(conn.storage_client);
  auto* mc = static_cast<hf3fs::meta::client::MetaClient*>(conn.meta_client);
  if (!sc || !mc) {
    DBG("GET sc=%p mc=%p", sc, mc);
    return;
  }

  const hf3fs::flat::UserInfo& ui = *shared_user_info_;
  auto* ri = static_cast<const hf3fs::client::RoutingInfo*>(conn.routing_info);
  if (!ri) return;

  // Resolve file via MetaClient
  std::string file_path = key_to_path(key);
  hf3fs::meta::Inode* inode = resolve_file(mc, ui, file_path, false, file_permission_);
  if (!inode) return;

  // Look up offset using DataStore (same as batch operations)
  DataStore& ds = *data_stores_[active_ds_index_];
  auto [offset, stored_size] = ds.lookup_offset(key);
  if (stored_size == 0) {
    DBG("GET offset not found or zero size for key=%s", key.c_str());
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  size_t read_len = std::min(static_cast<size_t>(stored_size), len);

  // Compute chainId and chunkId
  const auto& file = inode->asFile();
  auto chunkIdRes = file.getChunkId(inode->id, offset);
  if (!chunkIdRes) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  auto chainIdRes = file.getChainId(*inode, offset, *ri->raw(), 0);
  if (!chainIdRes) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }

  // Read data
  uint8_t* rb = alloc_rdma(read_len);
  if (!rb) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  auto* ioBuf = reg_buf(sc, rb, read_len);
  if (!ioBuf) {
    free_rdma(rb);
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  conn.io_buffers_.push_back(ioBuf);

  hf3fs::storage::client::ReadOptions ro;
  ro.set_enableChecksum(false);
  // NOTE: do NOT set bypass_disk_io(true) for reads — see do_batch_get comment.

  hf3fs::storage::client::ReadIO rio = sc->createReadIO(*chainIdRes, meta_chunk_id_to_storage(*chunkIdRes), offset,
      static_cast<uint32_t>(read_len), rb, ioBuf);
  auto res = folly::coro::blockingWait(sc->read(rio, ui, ro));
  delete ioBuf; free_rdma(rb);

  if (res.hasError() || rio.result.lengthInfo.hasError()) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }

  size_t cl = rio.result.lengthInfo ? *rio.result.lengthInfo : 0;
  if (cl > read_len) cl = read_len;
  std::memcpy(buf, rb, cl);

  // Close file (read operation)
  close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
}

// ---------- do_single_set ----------
void Hf3fsConnector::do_single_set(Hf3fsConnection& conn, const std::string& key,
    const void* buf, size_t len, size_t cs) {
  auto* sc = static_cast<hf3fs::storage::client::StorageClient*>(conn.storage_client);
  auto* mc = static_cast<hf3fs::meta::client::MetaClient*>(conn.meta_client);
  if (!sc || !mc) return;

  const hf3fs::flat::UserInfo& ui = *shared_user_info_;
  auto* ri = static_cast<const hf3fs::client::RoutingInfo*>(conn.routing_info);
  if (!ri) return;

  // Resolve file via MetaClient (create if missing)
  std::string file_path = key_to_path(key);
  hf3fs::meta::Inode* inode = resolve_file(mc, ui, file_path, true, file_permission_);
  if (!inode) return;

  // Look up or allocate offset using DataStore (same as batch operations)
  DataStore& ds = *data_stores_[active_ds_index_];
  auto [offset, stored_size] = ds.lookup_or_alloc(key, len);
  if (stored_size == 0) {
    DBG("SET failed to allocate offset for key=%s", key.c_str());
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  ds.mark_meta_dirty();

  // Compute chainId and chunkId
  const auto& file = inode->asFile();
  auto chunkIdRes = file.getChunkId(inode->id, offset);
  if (!chunkIdRes) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  auto chainIdRes = file.getChainId(*inode, offset, *ri->raw(), 0);
  if (!chainIdRes) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }

  // Write data
  uint8_t* rb = alloc_rdma(len);
  if (!rb) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  std::memcpy(rb, buf, len);
  auto* ioBuf = reg_buf(sc, rb, len);
  if (!ioBuf) { free_rdma(rb);
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }
  conn.io_buffers_.push_back(ioBuf);

  hf3fs::storage::client::WriteOptions wo;
  wo.set_enableChecksum(false);
  // NOTE: do NOT set bypass_disk_io(true) — we want data persisted to disk.

  hf3fs::storage::client::WriteIO wio = sc->createWriteIO(
      *chainIdRes, meta_chunk_id_to_storage(*chunkIdRes), 0, static_cast<uint32_t>(len), conn.chunk_size, rb, ioBuf);
  auto res = folly::coro::blockingWait(sc->write(wio, ui, wo));
  delete ioBuf; free_rdma(rb);

  if (res.hasError() || wio.result.lengthInfo.hasError()) {
    close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), false);
    return;
  }

  // Sync metadata and close file (write operation)
  // Build VersionedLength hint with the written data length
  hf3fs::meta::VersionedLength vlen{len, 0};
  std::optional<hf3fs::meta::VersionedLength> hint = vlen;
  auto syncRes = folly::coro::blockingWait(
      mc->sync(ui, inode->id, /*read=*/ false, /*written=*/ true, hint));
  if (syncRes.hasError()) {
    DBG("sync failed for %s: %s",
        file_path.c_str(), syncRes.error().describe().c_str());
  }
  close_file(mc, ui, inode, hf3fs::meta::client::SessionId::random(), true);
}

// ---------- do_single_exists ----------
bool Hf3fsConnector::do_single_exists(Hf3fsConnection& conn, const std::string& key) {
  auto* mc = static_cast<hf3fs::meta::client::MetaClient*>(conn.meta_client);
  if (!mc) return false;

  const hf3fs::flat::UserInfo& ui = *shared_user_info_;
  hf3fs::meta::InodeId inv;
  try {
    auto e = folly::coro::blockingWait(
        mc->stat(ui, inv,
                 std::optional<hf3fs::Path>{hf3fs::Path{key_to_path(key)}}, true));
    return e.hasError() ? false : de(e).isFile();
  } catch (...) { return false; }
}

bool Hf3fsConnector::do_single_delete(Hf3fsConnection& conn, const std::string& key) {
  auto* mc = static_cast<hf3fs::meta::client::MetaClient*>(conn.meta_client);
  if (!mc) return true;

  const hf3fs::flat::UserInfo& ui = *shared_user_info_;
  hf3fs::meta::InodeId parentIno(1);
  try {
    auto e = folly::coro::blockingWait(
        mc->remove(ui, parentIno,
                   std::optional<hf3fs::Path>{hf3fs::Path{key_to_path(key)}}, true));
    return e.hasError() ? false : true;
  } catch (...) { return false; }
}

}  // namespace connector
}  // namespace lmcache

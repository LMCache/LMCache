// SPDX-License-Identifier: Apache-2.0

#include "spdk.hpp"
#include <sched.h>
#include <pthread.h>
#include <chrono>
#include <sstream>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <unistd.h>
#include <thread>
#include <condition_variable>
#include <new>  // std::bad_alloc

// I/O ring capacity shared between the io worker and producer threads.
#define IO_RING_SIZE 4096

// Max re-enqueues of a failed submit before the I/O is cancelled.
#define MAX_RE_ENQUEUE_COUNT 100

// Completions reaped per pass in the io worker.
#define IO_WORKER_COMPLETION_REAP_SIZE 32

// Max time a synchronous (spdk_io) or batch (spdk_wait_batch) caller blocks
constexpr std::chrono::seconds g_spdk_wait_timeout{30};

// Max time launch_spdk_workers waits for the io_worker to finish device setup
// (connection + qpair allocation) before giving up.
constexpr std::chrono::seconds g_spdk_connect_timeout{10};

// Max IO command timeout (ms)
constexpr uint64_t g_spdk_cmd_timeout_us_default = 10ULL * 1000ULL * 1000ULL;

// Max admin command timeout (ms)
constexpr uint64_t g_spdk_admin_cmd_timeout_us = 30ULL * 1000ULL * 1000ULL;

// IoContext objects preallocated for the per-adapter pool.
#define IO_CTX_POOL_SIZE IO_RING_SIZE

// Time the admin_worker sleeps between admin-completion passes to avoid busy
// spinning on an idle controller.
constexpr std::chrono::milliseconds g_admin_worker_idle_ms{1};

// Process-global EAL reference counting; the one-shot EAL is shared across
// adapters and torn down only when the last one deinitializes.
std::atomic<int> SpdkIoEngineCore::g_eal_refcount_{0};
std::mutex SpdkIoEngineCore::g_eal_mutex_;

// ---------------------------------------------------------------------------
// File-local helper functions
// ---------------------------------------------------------------------------

// Resolve the per-command NVMe timeout (microseconds) from the environment.
static uint64_t resolve_spdk_cmd_timeout_us() {
  const uint64_t wait_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(g_spdk_wait_timeout)
          .count());
  const char* raw = std::getenv("LMCACHE_SPDK_COMMAND_TIMEOUT_US");
  if (raw == nullptr || *raw == '\0') {
    return g_spdk_cmd_timeout_us_default;
  }
  char* end = nullptr;
  errno = 0;
  unsigned long long value = std::strtoull(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || value == 0 ||
      value >= wait_us) {
    std::cerr << "[spdk_timeout] Ignoring invalid "
                 "LMCACHE_SPDK_COMMAND_TIMEOUT_US="
              << raw << " (must be in 1.." << (wait_us - 1)
              << "); using default " << g_spdk_cmd_timeout_us_default
              << std::endl;
    return g_spdk_cmd_timeout_us_default;
  }
  return static_cast<uint64_t>(value);
}

// Abort completion handling
static void spdk_abort_complete_cb(void* cb_arg,
                                   const struct spdk_nvme_cpl* completion) {
  SpdkIoEngineCore* engine = static_cast<SpdkIoEngineCore*>(cb_arg);
  bool aborted = spdk_nvme_cpl_is_abort_success(completion);
  if (aborted) {
    return;
  }
  std::cerr << "[spdk_timeout] Abort request did not abort the timed-out "
               "command; failing the controller"
            << std::endl;
  if (engine != nullptr) {
    engine->request_ctrlr_fail();
  }
}

// Per-command timeout callback: aborts the offending command or fails the
// controller.
static void timeout_cb(void* cb_arg, struct spdk_nvme_ctrlr* ctrlr,
                       struct spdk_nvme_qpair* qpair, uint16_t cid) {
  ConnectionContext* ctx = static_cast<ConnectionContext*>(cb_arg);
  SpdkIoEngineCore* engine = (ctx != nullptr) ? ctx->engine : nullptr;

  if (spdk_nvme_qpair_get_id(qpair) == 0) {
    std::cerr << "[spdk_timeout] Admin command timed out (cid=" << cid
              << "); failing the controller" << std::endl;
    if (engine != nullptr) {
      engine->request_ctrlr_fail();
    }
    return;
  }

  std::cerr << "[spdk_timeout] Command timed out (cid=" << cid << "); aborting"
            << std::endl;
  if (spdk_nvme_ctrlr_cmd_abort(ctrlr, qpair, cid, spdk_abort_complete_cb,
                                engine) != 0) {
    std::cerr << "[spdk_timeout] Failed to submit abort for cid=" << cid
              << "; failing the controller" << std::endl;
    if (engine != nullptr) {
      engine->request_ctrlr_fail();
    }
  }
}

static bool probe_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid,
                     struct spdk_nvme_ctrlr_opts* opts) {
  // 0xFFFFFFFFu is SPDK's "use the controller default/maximum" sentinel, so the
  // driver picks the largest queue depth it supports.
  opts->io_queue_size = 0xFFFFFFFFu;
  return true;
}

static void attach_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid,
                      struct spdk_nvme_ctrlr* ctrlr,
                      const struct spdk_nvme_ctrlr_opts* opts) {
  ConnectionContext* ctx = static_cast<ConnectionContext*>(cb_ctx);
  ctx->ctrlr = ctrlr;

  // Register the per-command timeout once.
  spdk_nvme_ctrlr_register_timeout_callback(
      ctrlr, resolve_spdk_cmd_timeout_us(), g_spdk_admin_cmd_timeout_us,
      timeout_cb, ctx);

  ctx->io_queue_depth.store(spdk_nvme_ctrlr_get_opts(ctrlr)->io_queue_size);
  ctx->io_queue_requests.store(
      spdk_nvme_ctrlr_get_opts(ctrlr)->io_queue_requests);

  uint32_t num_ns = spdk_nvme_ctrlr_get_num_ns(ctrlr);
  for (uint32_t i = 1; i <= num_ns; i++) {
    struct spdk_nvme_ns* ns = spdk_nvme_ctrlr_get_ns(ctrlr, i);
    if (ns == nullptr || !spdk_nvme_ns_is_active(ns)) {
      continue;
    }
    ctx->ns = ns;
    break;
  }

  ctx->is_connected.store(ctx->ns != nullptr);
}

// Map a caller transport string ("tcp"/"rdma") to the SPDK NVMf transport type.
static enum spdk_nvme_transport_type transport_type_to_spdk(
    const char* transport_type) {
  if (transport_type != nullptr && strcmp(transport_type, "rdma") == 0) {
    return SPDK_NVME_TRANSPORT_RDMA;
  }
  return SPDK_NVME_TRANSPORT_TCP;
}

/* Pick the address family from the textual address. A literal IPv6 address
contains ':' whereas an IPv4 address does not */
static enum spdk_nvmf_adrfam adrfam_for_address(const char* ip_addr) {
  if (ip_addr != nullptr && strchr(ip_addr, ':') != nullptr) {
    return SPDK_NVMF_ADRFAM_IPV6;
  }
  return SPDK_NVMF_ADRFAM_IPV4;
}

static int connect_remote_device(const char* transport_type,
                                 const char* ip_addr, const char* port,
                                 const char* nqn, ConnectionContext* ctx) {
  struct spdk_nvme_transport_id trid = {};

  trid.trtype = transport_type_to_spdk(transport_type);
  trid.adrfam = adrfam_for_address(ip_addr);

  snprintf(trid.traddr, sizeof(trid.traddr), "%s", ip_addr);
  snprintf(trid.trsvcid, sizeof(trid.trsvcid), "%s", port);
  snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", nqn);

  int rc = spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, nullptr);
  if (rc != 0 || !ctx->is_connected.load()) {
    std::cerr << "Failed to connect to NVMe-oF "
              << (transport_type ? transport_type : "") << " target"
              << std::endl;
    // attach_cb sets ctrlr even when no active namespace is found, so the
    // controller would leak here. Detach it now that probe has returned and
    // the attach phase is complete (detaching inside attach_cb would race the
    // in-progress probe).
    if (ctx->ctrlr != nullptr) {
      spdk_nvme_detach(ctx->ctrlr);
      ctx->ctrlr = nullptr;
    }
    ctx->ns = nullptr;
    return -1;
  }

  return 0;
}

static int connect_local_device(const char* pcie_addr, ConnectionContext* ctx) {
  struct spdk_nvme_transport_id trid = {};

  trid.trtype = SPDK_NVME_TRANSPORT_PCIE;
  snprintf(trid.traddr, sizeof(trid.traddr), "%s", pcie_addr);

  int rc = spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, nullptr);
  if (rc != 0 || !ctx->is_connected.load()) {
    std::cerr << "Failed to probe PCIe NVMe device at " << pcie_addr
              << std::endl;
    // attach_cb sets ctrlr even when no active namespace is found, so the
    // controller would leak here. Detach it now that probe has returned and
    // the attach phase is complete (detaching inside attach_cb would race the
    // in-progress probe).
    if (ctx->ctrlr != nullptr) {
      spdk_nvme_detach(ctx->ctrlr);
      ctx->ctrlr = nullptr;
    }
    ctx->ns = nullptr;
    return -1;
  }

  return 0;
}

static void io_complete_cb(void* cb_arg,
                           const struct spdk_nvme_cpl* completion) {
  struct IoContext* context = static_cast<struct IoContext*>(cb_arg);

  if (spdk_nvme_cpl_is_error(completion)) {
    context->status = -1;
  } else {
    context->status = 0;
    if (context->op == IO_READ && context->bounce_buffer != nullptr &&
        context->caller_buffer != nullptr && context->requested_bytes > 0) {
      std::memcpy(context->caller_buffer, context->bounce_buffer,
                  context->requested_bytes);
    }
  }

  // Capture this context's pointers before publishing is_completed: a waiter
  // frees (and recycles) the context the moment it sees the flag.
  SpdkIoEngineCore* engine = context->engine;
  BatchEntry* batch = context->batch;
  if (engine != nullptr) {
    // Return this command's full child-command count.
    engine->record_io_complete(context->command_count);
  }

  context->is_completed.store(true, std::memory_order_release);
  if (batch != nullptr) {
    batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
  }
}

static int spdk_command_count_for(struct spdk_nvme_ns* ns, uint64_t lba_count) {
  uint32_t max_xfer = spdk_nvme_ns_get_max_io_xfer_size(ns);
  if (max_xfer == 0) {
    // max_xfer==0 means a namespace with no usable transfer limit (malformed
    // MDTS)
    return 1;
  }
  uint64_t payload = (uint64_t)lba_count * spdk_nvme_ns_get_sector_size(ns);
  return (int)((payload + max_xfer - 1) / max_xfer);
}

// ---------------------------------------------------------------------------
// SpdkIoEngineCore member functions
// ---------------------------------------------------------------------------

SpdkIoEngineCore::SpdkIoEngineCore() { m_ctx_.engine = this; }

SpdkIoEngineCore::~SpdkIoEngineCore() { deinit_spdk(); }

// Allocate an IoContext from the per-adapter pool (lock-free), falling back to
// a heap allocation. Marks from_mempool so free_io_ctx uses the matching path.
IoContext* SpdkIoEngineCore::alloc_io_ctx() {
  if (m_io_ctx_pool_ != nullptr) {
    void* obj = nullptr;
    if (rte_mempool_get(m_io_ctx_pool_, &obj) == 0 && obj != nullptr) {
      // The mempool hands back memory reused by prior I/Os, so re-initialize in
      // place to mirror the heap path's value-init.
      IoContext* ctx = new (static_cast<IoContext*>(obj)) IoContext();
      ctx->from_mempool = true;
      return ctx;
    }
  }
  // Pool exhausted; heap-allocate. from_mempool stays false (default init), and
  // converting bad_alloc to nullptr lets callers fail the I/O gracefully.
  try {
    return new IoContext;
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

// Return an IoContext to the pool (lock-free)
void SpdkIoEngineCore::free_io_ctx(IoContext* io_ctx) {
  if (io_ctx == nullptr) {
    return;
  }
  // Release an owned partial-sector bounce buffer before the IoContext is
  // reclaimed.
  if (io_ctx->bounce_buffer != nullptr) {
    spdk_dma_free(io_ctx->bounce_buffer);
    io_ctx->bounce_buffer = nullptr;
  }
  if (io_ctx->from_mempool) {
    rte_mempool_put(m_io_ctx_pool_, io_ctx);
    return;
  }
  delete io_ctx;
}

// Release one EAL reference, finalizing the EAL when the last one goes.
void SpdkIoEngineCore::release_eal_ref() {
  std::lock_guard<std::mutex> lock(g_eal_mutex_);
  if (m_eal_refs_ != 1) {
    return;
  }
  int remaining = g_eal_refcount_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  m_eal_refs_ = 0;
  if (remaining <= 0) {
    spdk_env_fini();
  }
}

// Release the in-flight reservation for a completed I/O.
void SpdkIoEngineCore::record_io_complete(uint32_t command_count) {
  m_in_flight_.fetch_sub(command_count, std::memory_order_acq_rel);
}

// Record the io_worker setup outcome and wake launch_spdk_workers, which waits
// on the setup handshake instead of a fixed sleep.
void SpdkIoEngineCore::signal_setup_complete(int result) {
  {
    std::lock_guard<std::mutex> lock(m_setup_mutex_);
    m_setup_complete_ = true;
    m_setup_result_ = result;
  }
  m_setup_cv_.notify_all();
}

int SpdkIoEngineCore::resolve_worker_cores(const char* core_mask) {
  if (core_mask == nullptr || core_mask[0] == '\0') {
    std::cerr << "[resolve_worker_cores] core_mask must not be empty"
              << std::endl;
    return -1;
  }

  unsigned long mask = 0;
  std::vector<int> cores;

  // Derive the worker cores from the DPDK core mask when one is configured.
  try {
    mask = std::stoul(core_mask, nullptr, 16);
  } catch (...) {
    std::cerr << "[resolve_worker_cores] Invalid core mask" << std::endl;
    mask = 0;
    return -1;
  }

  // A mask that parses to zero (e.g. "0x0") selects no cores. Reject it before
  // the loop, since an empty core list would otherwise be dereferenced below.
  if (mask == 0) {
    std::cerr << "[resolve_worker_cores] core_mask has no cores set"
              << std::endl;
    return -1;
  }

  for (int i = 0; i < 64; i++) {
    if (mask & (1UL << i)) {
      cores.push_back(i);
    }
  }

  if (cores.size() == 1) {
    m_io_worker_core_ = cores[0];
    m_admin_worker_core_ = cores[0];
  } else {
    // io_worker takes the highest core, admin_worker the second-highest.
    m_io_worker_core_ = cores.back();
    m_admin_worker_core_ = cores[cores.size() - 2];
  }

  return 0;
}

int SpdkIoEngineCore::init_spdk(const char* core_mask, int mem_size_mb) {
  struct spdk_env_opts m_opts;

  spdk_env_opts_init(&m_opts);
  m_opts.name = "lmcache_spdk";

  if ((core_mask == nullptr) || (core_mask[0] == '\0') || (0 == mem_size_mb)) {
    std::cerr << "[init_spdk] Core Mask, mem size cannot be uninitialized"
              << std::endl;
    return -1;
  }

  // Resolve worker cores from the provided core mask; fail if it is empty.
  if (resolve_worker_cores(core_mask) != 0) {
    return -1;
  }

  // Use the resolved (possibly synthesized) core mask.
  m_opts.core_mask = core_mask;

  // Set SPDK memory size for hugepage allocation (if configured)
  if (mem_size_mb > 0) {
    m_opts.mem_size = mem_size_mb;
  }

  // The SPDK EAL is one-shot per process, so reference-count it: initialize it
  // for the first adapter and only tear it down when the last one
  // deinitializes.
  {
    std::lock_guard<std::mutex> lock(g_eal_mutex_);
    if (g_eal_refcount_.load(std::memory_order_acquire) == 0) {
      if (spdk_env_init(&m_opts) < 0) {
        std::cerr << "[init_spdk] SPDK env init failed" << std::endl;
        return -1;
      }
    }
    g_eal_refcount_.fetch_add(1, std::memory_order_acq_rel);
    m_eal_refs_ = 1;
  }

  m_io_ring_ = rte_ring_create("io_ring", IO_RING_SIZE, 0, RING_F_SC_DEQ);
  if (!m_io_ring_) {
    std::cerr << "[init_spdk] Failed to create rte_ring" << std::endl;
    release_eal_ref();
    return -1;
  }

  m_io_ctx_pool_ = rte_mempool_create("io_ctx_pool", IO_CTX_POOL_SIZE,
                                      sizeof(IoContext), 0, 0, nullptr, nullptr,
                                      nullptr, nullptr, SOCKET_ID_ANY, 0);
  if (!m_io_ctx_pool_) {
    std::cerr << "[init_spdk] Failed to create io_ctx mempool" << std::endl;
    rte_ring_free(m_io_ring_);
    m_io_ring_ = nullptr;
    release_eal_ref();
    return -1;
  }

  return 0;
}

void SpdkIoEngineCore::deinit_spdk() {
  // Guard against a second deinit.
  if (m_deinitialized_.exchange(true)) {
    return;
  }

  shutdown_spdk_workers();

  if (m_io_ring_ != nullptr) {
    rte_ring_free(m_io_ring_);
    m_io_ring_ = nullptr;
  }

  if (m_io_ctx_pool_ != nullptr) {
    rte_mempool_free(m_io_ctx_pool_);
    m_io_ctx_pool_ = nullptr;
  }

  // Release this adapter's EAL reference.
  release_eal_ref();
}

void SpdkIoEngineCore::shutdown_spdk_workers() {
  // Join once under this lock: a deadline-expiry, close, and destructor can
  // all arrive here concurrently. The io_worker never takes this mutex, so no
  // deadlock with its own teardown.
  std::lock_guard<std::mutex> lock(m_teardown_mutex_);

  // Do not fail the controller here (a clean close confirms everything; failing
  // a healthy controller only logs SPDK errors). Other callers fail it first if
  // needed: teardown_io_device() already does so when a command is outstanding.
  m_shutdown_.store(true, std::memory_order_release);

  // Join the io_worker first: its teardown cancels the ring, aborts outstanding
  // commands, frees the qpair, and joins the admin thread, so no SPDK callback
  // runs afterwards.
  if (m_io_thread_.joinable()) {
    m_io_thread_.join();
  }

  if (m_admin_thread_.joinable()) {
    m_admin_thread_.join();
  }
}

uintptr_t SpdkIoEngineCore::allocate_spdk_memory(size_t size, size_t align,
                                                 int numa_id) {
  void* buffer = spdk_dma_zmalloc_socket(size, align, nullptr, numa_id);
  if (buffer == nullptr) {
    std::cerr << "[allocate_spdk_memory] Failed to allocate " << size
              << " bytes" << std::endl;
    return 0;
  }
  return reinterpret_cast<uintptr_t>(buffer);
}

void SpdkIoEngineCore::free_spdk_memory(uintptr_t buff) {
  if (buff != 0) {
    void* buffer = reinterpret_cast<void*>(buff);
    spdk_dma_free(buffer);
  }
}

void SpdkIoEngineCore::cancel_io_ctx(IoContext* io_ctx) {
  io_ctx->status = -ECANCELED;
  // Same publication order as io_complete_cb: capture the batch before the flag
  // is published, then bump the count, so a waiter that frees on the flag never
  // races a read of the IoContext.
  BatchEntry* batch = io_ctx->batch;
  io_ctx->is_completed.store(true, std::memory_order_release);
  if (batch != nullptr) {
    batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
  }
}

void SpdkIoEngineCore::handle_submission_failure(
    int rc, IoContext* io_ctx, uint32_t current_inflight,
    struct spdk_nvme_qpair* qpair) {
  if (-ENOMEM != rc && -EAGAIN != rc) {
    std::cerr << "[io_worker] Cancelling I/O after a permanent submit failure "
                 "(LBA="
              << io_ctx->lba
              << ", op=" << (io_ctx->op == IO_WRITE ? "write" : "read") << ")"
              << " Error code= " << rc << std::endl;
    cancel_io_ctx(io_ctx);
    return;
  }
  // Respect the re-enqueue limit before retrying a transient error.
  if (io_ctx->re_enqueue_count >= MAX_RE_ENQUEUE_COUNT) {
    std::cerr << "[io_worker] Cancelling I/O after " << MAX_RE_ENQUEUE_COUNT
              << " failed re-enqueues (LBA=" << io_ctx->lba
              << ", op=" << (io_ctx->op == IO_WRITE ? "write" : "read") << ")"
              << " Error code= " << rc << std::endl;
    cancel_io_ctx(io_ctx);
  } else if (rte_ring_enqueue(m_io_ring_, reinterpret_cast<void*>(io_ctx)) !=
             0) {
    std::cerr << "[io_worker] Ring full" << std::endl;
    cancel_io_ctx(io_ctx);
  } else {
    io_ctx->re_enqueue_count++;
    /* Handle memory pressure by aggressively reaping completions */
    if (-ENOMEM == rc) {
      uint32_t required_headroom = (2 * io_ctx->command_count);
      uint32_t required_inflight = (current_inflight <= required_headroom)
                                       ? 0
                                       : (current_inflight - required_headroom);
      // Reap only while the device is recoverable, so a wedged command cannot
      // wedge the worker; the re-enqueued I/O is cancelled by teardown's drain.
      while (m_in_flight_.load(std::memory_order_acquire) > required_inflight &&
             !m_shutdown_.load() && !ctrlr_failing()) {
        spdk_nvme_qpair_process_completions(qpair,
                                            IO_WORKER_COMPLETION_REAP_SIZE);
      }
    }
  }
}

void SpdkIoEngineCore::request_ctrlr_fail() {
  CtrlrState expected = CtrlrState::Healthy;
  m_ctrlr_state_.compare_exchange_strong(expected, CtrlrState::FailRequested);
}

bool SpdkIoEngineCore::ctrlr_failing() const {
  return m_ctrlr_state_.load(std::memory_order_acquire) != CtrlrState::Healthy;
}

void SpdkIoEngineCore::fail_controller_now() {
  std::lock_guard<std::mutex> lock(m_ctrlr_mutex_);
  if (m_ctrlr_state_.exchange(CtrlrState::Failed) == CtrlrState::Failed) {
    return;
  }
  if (m_ctx_.ctrlr != nullptr) {
    spdk_nvme_ctrlr_fail(m_ctx_.ctrlr);
  }
}

void SpdkIoEngineCore::teardown_io_device(struct spdk_nvme_qpair* qpair) {
  if (ctrlr_failing() || m_in_flight_.load(std::memory_order_acquire) > 0) {
    fail_controller_now();
  }

  m_shutdown_.store(true, std::memory_order_release);
  m_ctx_.is_connected.store(false);

  IoContext* pending = nullptr;
  constexpr int k_max_ring_drain =
      IO_RING_SIZE + IO_WORKER_COMPLETION_REAP_SIZE;
  int ring_drained = 0;
  while (ring_drained < k_max_ring_drain &&
         rte_ring_dequeue(m_io_ring_, reinterpret_cast<void**>(&pending)) ==
             0) {
    std::cerr << "[io_worker] Cancelling I/O from ring (LBA=" << pending->lba
              << ", op=" << (pending->op == IO_WRITE ? "write" : "read") << ")"
              << std::endl;
    cancel_io_ctx(pending);
    ring_drained++;
  }

  if (m_admin_thread_.joinable()) {
    m_admin_thread_.join();
  }

  if (qpair != nullptr) {
    spdk_nvme_ctrlr_free_io_qpair(qpair);
  }

  {
    std::lock_guard<std::mutex> lock(m_ctrlr_mutex_);
    if (m_ctx_.ctrlr != nullptr) {
      spdk_nvme_detach(m_ctx_.ctrlr);
      m_ctx_.ctrlr = nullptr;
    }
  }

  if (m_in_flight_.load(std::memory_order_acquire) != 0) {
    std::cerr << "[io_worker] Teardown left " << m_in_flight_.load()
              << " I/O still in flight" << std::endl;
  }
}

void SpdkIoEngineCore::io_worker() {
  IoWorkerAliveGuard alive_guard(&m_worker_alive_);
  cpu_set_t cpuset;
  int rc = 0;
  IoContext* io_ctx = nullptr;
  struct spdk_nvme_qpair* qpair = nullptr;
  uint32_t queue_depth = 0;

  if (-1 == m_io_worker_core_) {
    std::cerr << "[io_worker] core is not initialized" << std::endl;
    signal_setup_complete(-1);
    teardown_io_device(nullptr);
    return;
  }
  CPU_ZERO(&cpuset);
  CPU_SET(m_io_worker_core_, &cpuset);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    std::cerr << "[io_worker] Failed to set thread affinity" << std::endl;
    signal_setup_complete(-1);
    teardown_io_device(nullptr);
    return;
  }

  if (m_transport_type_ == "pcie") {
    if (m_pcie_addr_.empty()) {
      std::cerr << "[io_worker] PCIe address not configured" << std::endl;
      signal_setup_complete(-1);
      teardown_io_device(nullptr);
      return;
    }
    if (connect_local_device(m_pcie_addr_.c_str(), &m_ctx_) != 0) {
      std::cerr << "[io_worker] Failed to connect to PCIe device" << std::endl;
      signal_setup_complete(-1);
      teardown_io_device(nullptr);
      return;
    }
  } else {
    if (connect_remote_device(m_transport_type_.c_str(), m_target_ip_.c_str(),
                              m_target_port_.c_str(), m_target_nqn_.c_str(),
                              &m_ctx_) != 0) {
      std::cerr << "[io_worker] Failed to connect to NVMe-oF target"
                << std::endl;
      signal_setup_complete(-1);
      teardown_io_device(nullptr);
      return;
    }
  }

  queue_depth = m_ctx_.io_queue_depth.load();
  if (0 == queue_depth) {
    std::cerr << "[io_worker] Invalid IO queue depth" << std::endl;
    // Controller is attached here (connect succeeded above); the single
    // teardown detaches it so the resource is not leaked.
    signal_setup_complete(-1);
    teardown_io_device(nullptr);
    return;
  }

  struct spdk_nvme_io_qpair_opts qpair_opts = {};
  spdk_nvme_ctrlr_get_default_io_qpair_opts(m_ctx_.ctrlr, &qpair_opts,
                                            sizeof(qpair_opts));
  /* Utilize controller reported config for queue allocation */
  qpair_opts.io_queue_size = m_ctx_.io_queue_depth.load();
  qpair_opts.io_queue_requests = m_ctx_.io_queue_requests.load();

  qpair_opts.opts_size = sizeof(qpair_opts);
  qpair = spdk_nvme_ctrlr_alloc_io_qpair(m_ctx_.ctrlr, &qpair_opts,
                                         sizeof(qpair_opts));
  if (!qpair) {
    std::cerr << "[io_worker] Failed to allocate qpair" << std::endl;
    signal_setup_complete(-1);
    teardown_io_device(nullptr);
    return;
  }

  // Device setup (connection + qpair allocation) succeeded: notify
  // launch_spdk_workers so it can start the admin worker and return.
  signal_setup_complete(0);

  while (true) {
    if (m_shutdown_.load()) {
      break;
    }

    // A failed controller has no reset path, so the loop must not spin
    // process_completions on it, break to the single teardown.
    if (ctrlr_failing()) {
      break;
    }

    if (rte_ring_dequeue(m_io_ring_, reinterpret_cast<void**>(&io_ctx)) != 0) {
      spdk_nvme_qpair_process_completions(qpair,
                                          IO_WORKER_COMPLETION_REAP_SIZE);
      continue;
    }

    const uint32_t reserved = io_ctx->command_count;
    if (reserved > queue_depth) {
      std::cerr << "[io_worker] I/O (LBA=" << io_ctx->lba
                << ", op=" << (io_ctx->op == IO_WRITE ? "write" : "read")
                << ") needs " << reserved
                << " child-command slots > queue depth " << queue_depth
                << "; rejecting." << std::endl;
      cancel_io_ctx(io_ctx);
      continue;
    }

    while (m_in_flight_.load(std::memory_order_acquire) + reserved >
               queue_depth &&
           !m_shutdown_.load()) {
      spdk_nvme_qpair_process_completions(qpair,
                                          IO_WORKER_COMPLETION_REAP_SIZE);
    }

    m_in_flight_.fetch_add(reserved, std::memory_order_acq_rel);
    uint32_t submitted_command_count = reserved;

    if (io_ctx->op == IO_WRITE) {
      rc = spdk_nvme_ns_cmd_write(m_ctx_.ns, qpair, io_ctx->buffer, io_ctx->lba,
                                  io_ctx->lba_count, io_complete_cb, io_ctx, 0);
    } else {
      rc = spdk_nvme_ns_cmd_read(m_ctx_.ns, qpair, io_ctx->buffer, io_ctx->lba,
                                 io_ctx->lba_count, io_complete_cb, io_ctx, 0);
    }

    if (rc != 0) {
      uint32_t current_inflight = m_in_flight_.fetch_sub(
          io_ctx->command_count, std::memory_order_acq_rel);
      handle_submission_failure(rc, io_ctx, current_inflight, qpair);
    }

    if (m_in_flight_.load(std::memory_order_acquire) >= 1) {
      spdk_nvme_qpair_process_completions(qpair, submitted_command_count);
    }
  }

  // Single device teardown for every exit path: normal shutdown, or the
  // controller-fail self-shutdown that broke out of the loop above.
  teardown_io_device(qpair);
}

void SpdkIoEngineCore::admin_worker() {
  cpu_set_t cpuset;

  if (-1 == m_admin_worker_core_) {
    std::cerr << "[admin_worker] core is not initialized" << std::endl;
    return;
  }

  CPU_ZERO(&cpuset);
  CPU_SET(m_admin_worker_core_, &cpuset);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    std::cerr << "[admin_worker] Failed to set thread affinity" << std::endl;
    return;
  }

  while (true) {
    if (m_shutdown_.load()) {
      if (m_ctx_.ctrlr) {
        spdk_nvme_ctrlr_process_admin_completions(m_ctx_.ctrlr);
      }
      break;
    }

    if (!m_ctx_.is_connected.load()) {
      std::this_thread::sleep_for(g_admin_worker_idle_ms);
      continue;
    }

    spdk_nvme_ctrlr_process_admin_completions(m_ctx_.ctrlr);
    std::this_thread::sleep_for(g_admin_worker_idle_ms);
  }
}

int SpdkIoEngineCore::set_connection_params(const char* transport_type,
                                            const char* addr, const char* port,
                                            const char* nqn) {
  if (transport_type) {
    m_transport_type_ = transport_type;
  }

  if (m_transport_type_ == "pcie") {
    if (addr) {
      m_pcie_addr_ = addr;
    }
  } else {
    if (addr) m_target_ip_ = addr;
    if (port) m_target_port_ = port;
    if (nqn) m_target_nqn_ = nqn;
  }
  return 0;
}

int SpdkIoEngineCore::launch_spdk_workers(const char* transport_type,
                                          const char* addr, const char* port,
                                          const char* nqn) {
  set_connection_params(transport_type, addr, port, nqn);

  // Reset the one-shot setup handshake before the io_worker can signal it.
  {
    std::lock_guard<std::mutex> lock(m_setup_mutex_);
    m_setup_complete_ = false;
    m_setup_result_ = 0;
  }
  m_worker_alive_.store(true, std::memory_order_release);

  m_io_thread_ = std::thread(&SpdkIoEngineCore::io_worker, this);

  std::unique_lock<std::mutex> lock(m_setup_mutex_);
  bool ready = m_setup_cv_.wait_for(lock, g_spdk_connect_timeout, [this] {
    return m_setup_complete_ ||
           !m_worker_alive_.load(std::memory_order_acquire);
  });
  int setup_result = ready ? m_setup_result_ : -1;
  if (!ready || setup_result != 0) {
    std::cerr << "[launch_spdk_workers] Device setup did not complete "
                 "(result="
              << setup_result << ")" << std::endl;
    return -1;
  }
  lock.unlock();

  if (m_admin_worker_core_ != -1) {
    m_admin_thread_ = std::thread(&SpdkIoEngineCore::admin_worker, this);
  }

  return 0;
}

// Populate one IoContext for an NVMe transfer segment (head or tail), sharing
// this file-local helper so both segments initialize identically.
static void set_io_context(IoContext* ctx, SpdkIoEngineCore* engine,
                           uint64_t lba, uint32_t lba_count, IoOp op,
                           void* buffer, void* bounce_buffer,
                           void* caller_buffer, uint64_t requested_bytes,
                           uint32_t command_count) {
  ctx->is_completed.store(false);
  ctx->status = 0;
  ctx->op = op;
  ctx->lba = lba;
  ctx->lba_count = lba_count;
  ctx->command_count = command_count;
  ctx->buffer = buffer;
  ctx->bounce_buffer = bounce_buffer;
  ctx->caller_buffer = caller_buffer;
  ctx->requested_bytes = requested_bytes;
  ctx->batch = nullptr;
  ctx->engine = engine;
  ctx->re_enqueue_count = 0;
}

int SpdkIoEngineCore::append_io_contexts(uint64_t byte_offset,
                                         uint64_t byte_count, void* buffer,
                                         IoOp op,
                                         std::vector<IoContext*>& out) {
  uint32_t sector_size = spdk_nvme_ns_get_sector_size(m_ctx_.ns);
  uint64_t aligned_head = byte_count - (byte_count % sector_size);
  uint64_t tail_bytes = byte_count - aligned_head;

  // A partial-sector request splits into an aligned head (direct DMA into the
  // caller buffer) and an unaligned tail DMAed through an owned bounce buffer.
  void* bounce_buffer = nullptr;
  if (tail_bytes > 0) {
    bounce_buffer = spdk_dma_zmalloc_socket(sector_size, sector_size, nullptr,
                                            SPDK_ENV_SOCKET_ID_ANY);
    if (bounce_buffer == nullptr) {
      std::cerr << "[spdk] Failed to allocate bounce buffer for "
                   "partial-sector tail"
                << std::endl;
      return -1;
    }
    if (op == IO_WRITE) {
      const void* tail_src = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(buffer) + aligned_head);
      std::memcpy(bounce_buffer, tail_src, tail_bytes);
      std::memset(static_cast<char*>(bounce_buffer) + tail_bytes, 0,
                  sector_size - tail_bytes);
    }
    IoContext* tail = alloc_io_ctx();
    if (tail == nullptr) {
      spdk_dma_free(bounce_buffer);
      std::cerr << "[spdk] Failed to allocate IoContext for partial-sector tail"
                << std::endl;
      return -1;
    }
    uint64_t tail_offset = byte_offset + aligned_head;
    set_io_context(tail, this, tail_offset / sector_size, 1, op, bounce_buffer,
                   bounce_buffer,
                   reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) +
                                           aligned_head),
                   (op == IO_READ) ? tail_bytes : 0,
                   spdk_command_count_for(m_ctx_.ns, 1));
    out.push_back(tail);
  }

  if (aligned_head > 0) {
    IoContext* head = alloc_io_ctx();
    if (head == nullptr) {
      // free_io_ctx also releases the tail's bounce buffer.
      for (IoContext* ctx : out) {
        free_io_ctx(ctx);
      }
      out.clear();
      std::cerr << "[spdk] Failed to allocate IoContext for aligned head"
                << std::endl;
      return -1;
    }
    uint32_t head_lba_count = (uint32_t)(aligned_head / sector_size);
    set_io_context(head, this, byte_offset / sector_size, head_lba_count, op,
                   buffer, nullptr, buffer, 0,
                   spdk_command_count_for(m_ctx_.ns, head_lba_count));
    out.push_back(head);
  }
  return 0;
}

int SpdkIoEngineCore::enqueue_io_context(IoContext* io_ctx, BatchEntry* batch) {
  io_ctx->batch = batch;
  io_ctx->engine = this;
  io_ctx->re_enqueue_count = 0;

  int rc = rte_ring_enqueue(m_io_ring_, reinterpret_cast<void*>(io_ctx));
  if (rc != 0) {
    std::cerr << (batch != nullptr ? "[spdk_batch_io_submit] Ring full"
                                   : "[spdk_io] Ring full")
              << std::endl;
    if (batch != nullptr) {
      batch->io_ctxs.push_back(io_ctx);
    }
    cancel_io_ctx(io_ctx);
    return -1;
  }

  if (batch != nullptr) {
    batch->io_ctxs.push_back(io_ctx);
  }
  return 0;
}

int SpdkIoEngineCore::spdk_io(uint64_t byte_offset, uint64_t byte_count,
                              void* buffer, IoOp op) {
  if (!buffer) {
    std::cerr << "[spdk_io] Null buffer" << std::endl;
    return -1;
  }

  if (!m_ctx_.is_connected.load() || m_ctx_.ns == nullptr) {
    std::cerr << "[spdk_io] No namespace connected. Call launch_spdk_workers "
                 "first."
              << std::endl;
    return -1;
  }

  if (ctrlr_failing()) {
    std::cerr << "[spdk_io] Request rejected: controller failed" << std::endl;
    return -1;
  }

  uint32_t sector_size = spdk_nvme_ns_get_sector_size(m_ctx_.ns);
  if (byte_offset % sector_size != 0) {
    std::cerr << "[spdk_io] byte_offset=" << byte_offset
              << " is not a multiple of sector size=" << sector_size
              << std::endl;
    return -1;
  }

  // A partial-sector request is built (aligned head + bounced tail) by
  // append_io_contexts, so this path shares its allocation-failure cleanup.
  std::vector<IoContext*> io_ctxs;
  io_ctxs.reserve(2);
  if (append_io_contexts(byte_offset, byte_count, buffer, op, io_ctxs) != 0) {
    std::cerr << "[spdk_io] Failed to prepare I/O contexts" << std::endl;
    return -1;
  }

  bool submit_failed = false;
  for (IoContext* io_ctx : io_ctxs) {
    if (enqueue_io_context(io_ctx, nullptr) != 0) {
      submit_failed = true;
    }
  }

  // Never block forever: stop spinning once every I/O completes.
  auto io_deadline = std::chrono::steady_clock::now() + g_spdk_wait_timeout;
  bool timed_out = false;
  for (IoContext* io_ctx : io_ctxs) {
    while (!io_ctx->is_completed.load() && m_worker_alive_.load() &&
           std::chrono::steady_clock::now() < io_deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!io_ctx->is_completed.load()) {
      timed_out = true;
    }
  }

  if (timed_out) {
    std::cerr << "[spdk_io] Wait deadline reached with I/O still outstanding; "
                 "tearing the device down to confirm it"
              << std::endl;
    fail_controller_now();
    shutdown_spdk_workers();
  }

  const char* op_name = (op == IO_WRITE) ? "write" : "read";
  bool failed = submit_failed || timed_out;
  for (IoContext* io_ctx : io_ctxs) {
    if (io_ctx->status != 0) {
      failed = true;
    }
    if (!io_ctx->is_completed.load(std::memory_order_acquire)) {
      std::cerr << "[spdk_io] Internal error: I/O still incomplete after "
                   "teardown (LBA="
                << io_ctx->lba << ", op=" << op_name << ")" << std::endl;
      failed = true;
    }
    free_io_ctx(io_ctx);
  }

  return failed ? -1 : 0;
}

int SpdkIoEngineCore::register_external_memory(uintptr_t ptr, size_t size) {
  if (ptr == 0 || size == 0) {
    std::cerr << "[register_external_memory] Invalid args" << std::endl;
    return -1;
  }

  int rc = spdk_mem_register(reinterpret_cast<void*>(ptr), size);
  if (rc != 0) {
    std::cerr << "[register_external_memory] Failed" << std::endl;
    return -1;
  }

  return 0;
}

int SpdkIoEngineCore::unregister_external_memory(uintptr_t ptr, size_t size) {
  if (ptr == 0 || size == 0) {
    std::cerr << "[unregister_external_memory] Invalid args" << std::endl;
    return -1;
  }

  int rc = spdk_mem_unregister(reinterpret_cast<void*>(ptr), size);
  if (rc != 0) {
    std::cerr << "[unregister_external_memory] Failed" << std::endl;
    return -1;
  }

  return 0;
}

int SpdkIoEngineCore::get_device_size(uint64_t& result_size) {
  if (!m_ctx_.is_connected.load() || m_ctx_.ns == nullptr) {
    std::cerr << "[get_device_size] No namespace connected" << std::endl;
    return -1;
  }

  uint64_t num_sectors = spdk_nvme_ns_get_num_sectors(m_ctx_.ns);
  uint32_t sector_size = spdk_nvme_ns_get_sector_size(m_ctx_.ns);
  result_size = num_sectors * sector_size;

  return 0;
}

int SpdkIoEngineCore::get_max_io_depth(uint32_t& depth_out) {
  if (!m_ctx_.is_connected.load() || m_ctx_.ns == nullptr) {
    std::cerr << "[get_max_io_depth] No namespace connected" << std::endl;
    return -1;
  }

  depth_out = m_ctx_.io_queue_depth.load();
  return 0;
}

int SpdkIoEngineCore::get_max_transfer_size(uint32_t& transfer_out) {
  if (!m_ctx_.is_connected.load() || m_ctx_.ns == nullptr) {
    std::cerr << "[get_max_transfer_size] No namespace connected" << std::endl;
    return -1;
  }

  transfer_out = spdk_nvme_ns_get_max_io_xfer_size(m_ctx_.ns);
  return 0;
}

int SpdkIoEngineCore::spdk_batch_io_submit(uint64_t* offsets,
                                           uint64_t* total_lens,
                                           uint64_t* buf_ptrs, int count,
                                           IoOp op, uint32_t* batch_id_out) {
  if (!m_ctx_.is_connected.load() || m_ctx_.ns == nullptr) {
    std::cerr << "[spdk_batch_io_submit] No namespace connected" << std::endl;
    return -1;
  }

  if (ctrlr_failing()) {
    std::cerr << "[spdk_batch_io_submit] Request rejected: controller failed"
              << std::endl;
    return -1;
  }

  if ((batch_id_out == nullptr) || (count <= 0)) {
    std::cerr << "[spdk_batch_io_submit] Invalid batch id buffer or count"
              << std::endl;
    return -1;
  }

  uint32_t sector_size = spdk_nvme_ns_get_sector_size(m_ctx_.ns);

  // Validate the whole batch before allocating so no IoContext is enqueued for
  // a batch that might later be abandoned.
  for (int i = 0; i < count; ++i) {
    if (offsets[i] % sector_size != 0) {
      std::cerr << "[spdk_batch_io_submit] offsets[" << i << "]=" << offsets[i]
                << " is not a multiple of sector size=" << sector_size
                << std::endl;
      return -1;
    }
    if (buf_ptrs[i] == 0) {
      std::cerr << "[spdk_batch_io_submit] Null buffer pointer at index " << i
                << std::endl;
      return -1;
    }
  }

  uint32_t batch_id = m_next_batch_id_.fetch_add(1, std::memory_order_relaxed);
  auto batch = std::make_shared<BatchEntry>();
  batch->batch_id = batch_id;
  batch->io_ctxs.reserve(2 * (size_t)count);
  batch->completed_count.store(0, std::memory_order_relaxed);
  batch->alloc_fail_count = 0;

  for (int i = 0; i < count; ++i) {
    // Build this slot's head/tail via the shared helper (bounce buffer only for
    // the tail); a slot-level alloc failure is recorded as a batch failure so
    // waiters still settle.
    std::vector<IoContext*> slot_ctxs;
    slot_ctxs.reserve(2);
    if (append_io_contexts(offsets[i], total_lens[i],
                           reinterpret_cast<void*>(buf_ptrs[i]), op,
                           slot_ctxs) != 0) {
      std::cerr << "[spdk_batch_io_submit] Failed to prepare I/O at index " << i
                << "; marking batch " << batch_id << " slot failed"
                << std::endl;
      batch->alloc_fail_count += 1;
      batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }
    for (IoContext* io_ctx : slot_ctxs) {
      enqueue_io_context(io_ctx, batch.get());
    }
  }

  batch->count = (int)batch->alloc_fail_count + (int)batch->io_ctxs.size();

  {
    std::lock_guard<std::mutex> lock(m_batch_map_mutex_);
    m_batch_map_[batch_id] = batch;
  }

  *batch_id_out = batch_id;
  return 0;
}

int SpdkIoEngineCore::spdk_wait_batch(uint32_t batch_id) {
  std::shared_ptr<BatchEntry> batch;
  {
    std::lock_guard<std::mutex> lock(m_batch_map_mutex_);
    auto it = m_batch_map_.find(batch_id);
    if (it == m_batch_map_.end()) {
      std::cerr << "[spdk_wait_batch] Unknown batch_id: " << batch_id
                << std::endl;
      return -1;
    }
    batch = it->second;
  }

  int total = batch->count;

  auto batch_deadline = std::chrono::steady_clock::now() + g_spdk_wait_timeout;
  while (batch->completed_count.load(std::memory_order_acquire) < total &&
         m_worker_alive_.load() &&
         std::chrono::steady_clock::now() < batch_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (batch->completed_count.load(std::memory_order_acquire) < total) {
    std::cerr << "[spdk_wait_batch] Batch " << batch_id
              << " did not settle before its deadline/worker exit; tearing the "
                 "device down to confirm it"
              << std::endl;
    fail_controller_now();
    shutdown_spdk_workers();
  }

  int status = (batch->alloc_fail_count > 0) ? -1 : 0;
  for (IoContext* io_ctx : batch->io_ctxs) {
    if (io_ctx->status != 0) {
      status = -1;
    }
    if (!io_ctx->is_completed.load(std::memory_order_acquire)) {
      std::cerr
          << "[spdk_wait_batch] Internal error: I/O still incomplete after "
             "teardown (LBA="
          << io_ctx->lba << ")" << std::endl;
      status = -1;
    }
    free_io_ctx(io_ctx);
  }

  {
    std::lock_guard<std::mutex> lock(m_batch_map_mutex_);
    m_batch_map_.erase(batch_id);
  }

  return status;
}

// ---------------------------------------------------------------------------
// C wrapper functions for ctypes FFI compatibility
// ---------------------------------------------------------------------------

extern "C" {

int core_init_spdk(void* core_ptr, const char* core_mask, int mem_size_mb) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->init_spdk(core_mask, mem_size_mb);
}

void core_deinit_spdk(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  core->deinit_spdk();
}

uintptr_t core_allocate_spdk_memory(void* core_ptr, size_t size, size_t align,
                                    int numa_id) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->allocate_spdk_memory(size, align, numa_id);
}

void core_free_spdk_memory(void* core_ptr, uintptr_t buff) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  core->free_spdk_memory(buff);
}

int core_register_external_memory(void* core_ptr, uintptr_t ptr, size_t size) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->register_external_memory(ptr, size);
}

int core_unregister_external_memory(void* core_ptr, uintptr_t ptr,
                                    size_t size) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->unregister_external_memory(ptr, size);
}

/* Wrapper functions. `op` is IO_READ (0) or IO_WRITE (1). */
int core_spdk_io(void* core_ptr, uint64_t byte_offset, uint64_t byte_count,
                 void* buffer, int op) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->spdk_io(byte_offset, byte_count, buffer, static_cast<IoOp>(op));
}

int core_launch_spdk_workers(void* core_ptr, const char* transport_type,
                             const char* addr, const char* port,
                             const char* nqn) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->launch_spdk_workers(transport_type, addr, port, nqn);
}

int core_get_device_size(void* core_ptr, uint64_t* result_size) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->get_device_size(*result_size);
}

int core_get_max_io_depth(void* core_ptr, uint32_t* depth_out) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->get_max_io_depth(*depth_out);
}

// Wrapper function. Returns the namespace MDTS (max transfer size) in bytes.
int core_get_max_transfer_size(void* core_ptr, uint32_t* transfer_out) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->get_max_transfer_size(*transfer_out);
}

// Wrapper function. `op` is IO_READ (0) or IO_WRITE (1).
int core_spdk_batch_io_submit(void* core_ptr, uint64_t* offsets,
                              uint64_t* total_lens, uint64_t* buf_ptrs,
                              int count, int op, uint32_t* batch_id_out) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->spdk_batch_io_submit(offsets, total_lens, buf_ptrs, count,
                                    static_cast<IoOp>(op), batch_id_out);
}

int core_spdk_wait_batch(void* core_ptr, uint32_t batch_id) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->spdk_wait_batch(batch_id);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Factory functions (owned by the Python FFI until free_SpdkIoEngineCore)
// ---------------------------------------------------------------------------

// Return a raw heap pointer so the Python FFI owns the object's lifetime until
// free_SpdkIoEngineCore() is called.
SpdkIoEngineCore* make_SpdkIoEngineCore() { return new SpdkIoEngineCore(); }

void free_SpdkIoEngineCore(SpdkIoEngineCore* core) { delete core; }

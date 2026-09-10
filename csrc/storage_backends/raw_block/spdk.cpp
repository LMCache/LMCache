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
#include <new>  // std::bad_alloc

// I/O ring capacity shared between the io worker and producer threads.
#define IO_RING_SIZE 4096

// Max re-enqueues of a failed submit before the I/O is cancelled.
#define MAX_RE_ENQUEUE_COUNT 100

// Completions reaped per pass in the io worker.
#define IO_WORKER_COMPLETION_REAP_SIZE 32

// Max time a synchronous (spdk_io) or batch (spdk_wait_batch) caller blocks
constexpr std::chrono::seconds g_spdk_wait_timeout{30};

// Max graceful drain time (ms) the io_worker reaps in-flight I/O at shutdown
constexpr int g_drain_timeout_ms = 1000;

static_assert(
    g_drain_timeout_ms <=
        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             g_spdk_wait_timeout)
                             .count()),
    "g_drain_timeout_ms must not exceed g_spdk_wait_timeout");

// IoContext objects preallocated for the per-adapter pool.
#define IO_CTX_POOL_SIZE IO_RING_SIZE

// Process-global EAL reference counting; the one-shot EAL is shared across
// adapters and torn down only when the last one deinitializes.
std::atomic<int> SpdkIoEngineCore::g_eal_refcount_{0};
std::mutex SpdkIoEngineCore::g_eal_mutex_;

// Allocate an IoContext from the per-adapter pool (lock-free), falling back to
// a heap allocation. Marks from_mempool so free_io_ctx uses the matching path.
IoContext* SpdkIoEngineCore::alloc_io_ctx() {
  if (m_io_ctx_pool_ != nullptr) {
    void* obj = nullptr;
    if (rte_mempool_get(m_io_ctx_pool_, &obj) == 0 && obj != nullptr) {
      IoContext* ctx = static_cast<IoContext*>(obj);
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

// Register an IoContext a caller gave up on (timed out or the worker exited)
void SpdkIoEngineCore::register_abandoned_io_ctx(IoContext* io_ctx) {
  if (io_ctx == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(m_abandoned_mutex_);
  m_abandoned_io_ctx_.push_back(io_ctx);
}

// Free every IoContext a caller abandoned after timing out, during shutdown
// drain
void SpdkIoEngineCore::drain_abandoned_io_ctxs() {
  std::vector<IoContext*> to_free;
  {
    std::lock_guard<std::mutex> lock(m_abandoned_mutex_);
    to_free.swap(m_abandoned_io_ctx_);
  }
  for (IoContext* io_ctx : to_free) {
    free_io_ctx(io_ctx);
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

  ctx->io_queue_depth = spdk_nvme_ctrlr_get_opts(ctrlr)->io_queue_size;
  ctx->io_queue_requests = spdk_nvme_ctrlr_get_opts(ctrlr)->io_queue_requests;

  uint32_t num_ns = spdk_nvme_ctrlr_get_num_ns(ctrlr);
  for (uint32_t i = 1; i <= num_ns; i++) {
    struct spdk_nvme_ns* ns = spdk_nvme_ctrlr_get_ns(ctrlr, i);
    if (ns == nullptr || !spdk_nvme_ns_is_active(ns)) {
      continue;
    }
    ctx->ns = ns;
    break;
  }

  ctx->is_connected = (ctx->ns != nullptr);
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
  if (rc != 0 || !ctx->is_connected) {
    std::cerr << "Failed to connect to NVMe-oF "
              << (transport_type ? transport_type : "") << " target"
              << std::endl;
    return -1;
  }

  return 0;
}

static int connect_local_device(const char* pcie_addr, ConnectionContext* ctx) {
  struct spdk_nvme_transport_id trid = {};

  trid.trtype = SPDK_NVME_TRANSPORT_PCIE;
  snprintf(trid.traddr, sizeof(trid.traddr), "%s", pcie_addr);

  int rc = spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, nullptr);
  if (rc != 0 || !ctx->is_connected) {
    std::cerr << "Failed to probe PCIe NVMe device at " << pcie_addr
              << std::endl;
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
  }

  context->is_completed.store(true);
  // Return this command's full child-command count.
  if (context->engine != nullptr) {
    context->engine->record_io_complete(context->command_count);
  }

  // Update batch completion count if part of a batch.
  if (context->batch != nullptr) {
    context->batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
  }
}

SpdkIoEngineCore::SpdkIoEngineCore() {}

SpdkIoEngineCore::~SpdkIoEngineCore() {
  // stop workers and release resources even if the Python FFI did not call
  // shutdown_spdk_workers()/deinit_spdk()
  if (m_io_thread_.joinable() || m_admin_thread_.joinable()) {
    m_shutdown_.store(true, std::memory_order_release);
    if (m_io_thread_.joinable()) {
      m_io_thread_.join();
    }
    if (m_admin_thread_.joinable()) {
      m_admin_thread_.join();
    }
  }
  deinit_spdk();
}

// Return a raw heap pointer so the Python FFI owns the object's lifetime until
// free_SpdkIoEngineCore() is called.
SpdkIoEngineCore* make_SpdkIoEngineCore() { return new SpdkIoEngineCore(); }

void free_SpdkIoEngineCore(SpdkIoEngineCore* core) { delete core; }

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

// C wrapper functions for ctypes FFI compatibility
extern "C" {

int core_init_spdk(void* core_ptr, const char* core_mask, int mem_size_mb) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->init_spdk(core_mask, mem_size_mb);
}

void core_deinit_spdk(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  core->deinit_spdk();
}

void core_shutdown_spdk_workers(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  core->shutdown_spdk_workers();
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
  if (!core || !result_size) {
    std::cerr << "[core_get_device_size] Invalid arguments: core_ptr or "
                 "result_size is null"
              << std::endl;
    return -1;
  }
  return core->get_device_size(*result_size);
}

int core_get_max_io_depth(void* core_ptr, uint32_t* depth_out) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  if (!core || !depth_out) {
    std::cerr << "[core_get_max_io_depth] Invalid arguments: core_ptr or "
                 "depth_out is null"
              << std::endl;
    return -1;
  }
  return core->get_max_io_depth(*depth_out);
}

}  // extern "C"

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
  m_shutdown_.store(true, std::memory_order_release);

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

class IoWorkerAliveGuard {
 public:
  IoWorkerAliveGuard(std::atomic<bool>* alive) : alive_(alive) {}
  ~IoWorkerAliveGuard() { alive_->store(false, std::memory_order_release); }

 private:
  std::atomic<bool>* alive_;
};

void SpdkIoEngineCore::io_worker() {
  IoWorkerAliveGuard alive_guard(&m_worker_alive_);
  cpu_set_t cpuset;
  int rc = 0;
  IoContext* io_ctx = nullptr;
  struct spdk_nvme_qpair* qpair = nullptr;

  if (-1 == m_io_worker_core_) {
    std::cerr << "[io_worker] core is not initialized" << std::endl;
    return;
  }
  CPU_ZERO(&cpuset);
  CPU_SET(m_io_worker_core_, &cpuset);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    std::cerr << "[io_worker] Failed to set thread affinity" << std::endl;
    return;
  }

  if (m_transport_type_ == "pcie") {
    if (m_pcie_addr_.empty()) {
      std::cerr << "[io_worker] PCIe address not configured" << std::endl;
      return;
    }
    if (connect_local_device(m_pcie_addr_.c_str(), &m_ctx_) != 0) {
      std::cerr << "[io_worker] Failed to connect to PCIe device" << std::endl;
      return;
    }
  } else {
    if (connect_remote_device(m_transport_type_.c_str(), m_target_ip_.c_str(),
                              m_target_port_.c_str(), m_target_nqn_.c_str(),
                              &m_ctx_) != 0) {
      std::cerr << "[io_worker] Failed to connect to NVMe-oF target"
                << std::endl;
      return;
    }
  }

  if (0 == m_ctx_.io_queue_depth) {
    std::cerr << "[io_worker] Invalid IO queue depth" << std::endl;
    return;
  }

  struct spdk_nvme_io_qpair_opts qpair_opts = {};
  spdk_nvme_ctrlr_get_default_io_qpair_opts(m_ctx_.ctrlr, &qpair_opts,
                                            sizeof(qpair_opts));
  /* Utilize controller reported config for queue allocation */
  qpair_opts.io_queue_size = m_ctx_.io_queue_depth;
  qpair_opts.io_queue_requests = m_ctx_.io_queue_requests;

  qpair_opts.opts_size = sizeof(qpair_opts);
  qpair = spdk_nvme_ctrlr_alloc_io_qpair(m_ctx_.ctrlr, &qpair_opts,
                                         sizeof(qpair_opts));
  if (!qpair) {
    std::cerr << "[io_worker] Failed to allocate qpair" << std::endl;
    spdk_nvme_detach(m_ctx_.ctrlr);
    return;
  }

  // Cancel an I/O that can no longer be retried
  auto cancel_io_ctx = [](IoContext* io_ctx) {
    io_ctx->status = -ECANCELED;
    io_ctx->is_completed.store(true);
    if (io_ctx->batch != nullptr) {
      io_ctx->batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
    }
  };

  while (true) {
    if (m_shutdown_.load()) {
      break;
    }

    if (rte_ring_dequeue(m_io_ring_, reinterpret_cast<void**>(&io_ctx)) != 0) {
      spdk_nvme_qpair_process_completions(qpair,
                                          IO_WORKER_COMPLETION_REAP_SIZE);
      continue;
    }

    m_in_flight_.fetch_add(io_ctx->command_count, std::memory_order_acq_rel);
    uint32_t submitted_command_count = io_ctx->command_count;

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
      // Respect the re-enqueue limit before retrying
      if (io_ctx->re_enqueue_count >= MAX_RE_ENQUEUE_COUNT) {
        std::cerr << "[io_worker] Cancelling I/O after " << MAX_RE_ENQUEUE_COUNT
                  << " failed re-enqueues (LBA=" << io_ctx->lba
                  << ", op=" << (io_ctx->op == IO_WRITE ? "write" : "read")
                  << ")"
                  << " Error code= " << rc << std::endl;
        cancel_io_ctx(io_ctx);
      } else if (rte_ring_enqueue(m_io_ring_,
                                  reinterpret_cast<void*>(io_ctx)) != 0) {
        std::cerr << "[io_worker] Ring full" << std::endl;
        cancel_io_ctx(io_ctx);
      } else {
        io_ctx->re_enqueue_count++;
        /* Handle memory pressure by aggressively reaping completions */
        if (-ENOMEM == rc) {
          uint32_t required_headroom = (2 * io_ctx->command_count);
          uint32_t required_inflight =
              (current_inflight <= required_headroom)
                  ? 0
                  : (current_inflight - required_headroom);
          while (m_in_flight_.load(std::memory_order_acquire) >
                 required_inflight) {
            spdk_nvme_qpair_process_completions(qpair,
                                                IO_WORKER_COMPLETION_REAP_SIZE);
          }
        }
      }
    }

    if (m_in_flight_.load(std::memory_order_acquire) >= 1) {
      spdk_nvme_qpair_process_completions(qpair, submitted_command_count);
    }
  }

  // Capture the drain start time to enforce the graceful drain timeout below.
  auto drain_start = std::chrono::steady_clock::now();

  // Gate out new submissions
  m_ctx_.is_connected = false;

  // Bounded drain count
  constexpr int k_max_ring_drain =
      IO_RING_SIZE + IO_WORKER_COMPLETION_REAP_SIZE;
  int ring_drained = 0;
  while (ring_drained < k_max_ring_drain &&
         rte_ring_dequeue(m_io_ring_, reinterpret_cast<void**>(&io_ctx)) == 0) {
    std::cerr << "[io_worker] Cancelling I/O from ring (LBA=" << io_ctx->lba
              << ", op=" << (io_ctx->op == IO_WRITE ? "write" : "read") << ")"
              << std::endl;
    cancel_io_ctx(io_ctx);
    ring_drained++;
  }

  while (m_in_flight_.load(std::memory_order_acquire) > 0) {
    spdk_nvme_qpair_process_completions(qpair, IO_WORKER_COMPLETION_REAP_SIZE);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - drain_start)
                       .count();
    if (elapsed >= g_drain_timeout_ms) {
      std::cerr << "[io_worker] Timeout (" << g_drain_timeout_ms
                << "ms) waiting for " << m_in_flight_.load()
                << " in-flight I/Os. Marking remaining as failed." << std::endl;
      break;
    }
    std::this_thread::yield();
  }

  spdk_nvme_ctrlr_free_io_qpair(qpair);
  spdk_nvme_detach(m_ctx_.ctrlr);

  // Release abandoned IoContext
  drain_abandoned_io_ctxs();
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

    if (!m_ctx_.is_connected) {
      std::this_thread::yield();
      continue;
    }

    spdk_nvme_ctrlr_process_admin_completions(m_ctx_.ctrlr);
    std::this_thread::yield();
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
  m_io_thread_ = std::thread(&SpdkIoEngineCore::io_worker, this);

  /* Wait for connection */
  sleep(1);

  /* Only launch admin worker for NVMe-oF */
  if (m_transport_type_ != "pcie") {
    m_admin_thread_ = std::thread(&SpdkIoEngineCore::admin_worker, this);
  }

  return 0;
}

int SpdkIoEngineCore::spdk_io(uint64_t byte_offset, uint64_t byte_count,
                              void* buffer, IoOp op) {
  if (!buffer) {
    std::cerr << "[spdk_io] Null buffer" << std::endl;
    return -1;
  }

  // Check if we have a connected namespace
  if (!m_ctx_.is_connected || m_ctx_.ns == nullptr) {
    std::cerr << "[spdk_io] No namespace connected. Call launch_spdk_workers "
                 "first."
              << std::endl;
    return -1;
  }

  uint32_t sector_size = spdk_nvme_ns_get_sector_size(m_ctx_.ns);
  uint64_t lba = byte_offset / sector_size;
  uint64_t lba_count = (byte_count + sector_size - 1) / sector_size;

  IoContext* io_ctx = alloc_io_ctx();
  if (io_ctx == nullptr) {
    std::cerr << "[spdk_io] Failed to allocate IoContext" << std::endl;
    return -1;
  }
  io_ctx->is_completed.store(false);
  io_ctx->status = 0;
  io_ctx->lba = lba;
  io_ctx->lba_count = lba_count;
  io_ctx->command_count = spdk_command_count_for(m_ctx_.ns, lba_count);
  io_ctx->op = op;
  io_ctx->buffer = buffer;
  // These two fields are not set by the caller, so initialize them explicitly:
  io_ctx->batch = nullptr;
  io_ctx->engine = this;
  io_ctx->re_enqueue_count = 0;

  int rc = rte_ring_enqueue(m_io_ring_, reinterpret_cast<void*>(io_ctx));
  if (rc != 0) {
    std::cerr << "[spdk_io] Ring full" << std::endl;
    free_io_ctx(io_ctx);
    return -1;
  }

  // Never block forever: stop spinning once the I/O completes.
  auto io_deadline = std::chrono::steady_clock::now() + g_spdk_wait_timeout;
  while (!io_ctx->is_completed.load() && m_worker_alive_.load() &&
         std::chrono::steady_clock::now() < io_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (!io_ctx->is_completed.load()) {
    std::cerr << "[spdk_io] Timed out waiting for completion (LBA="
              << io_ctx->lba << ", op=" << (op == IO_WRITE ? "write" : "read")
              << ")";
    if (!m_worker_alive_.load()) {
      std::cerr << "; worker exited";
    }
    std::cerr << "; ownership handed to io_worker for reclamation."
              << std::endl;
    // Do NOT free here: this op may still be in-flight on the device and owned
    // by the worker.
    register_abandoned_io_ctx(io_ctx);
    return -1;
  }

  if (io_ctx->status != 0) {
    free_io_ctx(io_ctx);
    return -1;
  }

  free_io_ctx(io_ctx);
  return 0;
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
  if (!m_ctx_.is_connected || m_ctx_.ns == nullptr) {
    std::cerr << "[get_device_size] No namespace connected" << std::endl;
    return -1;
  }

  uint64_t num_sectors = spdk_nvme_ns_get_num_sectors(m_ctx_.ns);
  uint32_t sector_size = spdk_nvme_ns_get_sector_size(m_ctx_.ns);
  result_size = num_sectors * sector_size;

  return 0;
}

int SpdkIoEngineCore::get_max_io_depth(uint32_t& depth_out) {
  depth_out = m_ctx_.io_queue_depth;
  return 0;
}

int SpdkIoEngineCore::spdk_batch_io_submit(uint64_t* offsets,
                                           uint64_t* total_lens,
                                           uint64_t* buf_ptrs, int count,
                                           IoOp op, uint32_t* batch_id_out) {
  if (!m_ctx_.is_connected || m_ctx_.ns == nullptr) {
    std::cerr << "[spdk_batch_io_submit] No namespace connected" << std::endl;
    return -1;
  }

  if ((batch_id_out == nullptr) || (count <= 0)) {
    std::cerr << "[spdk_batch_io_submit] Invalid batch id buffer or count"
              << std::endl;
    return -1;
  }

  uint32_t sector_size = spdk_nvme_ns_get_sector_size(m_ctx_.ns);

  // Allocate a batch entry for tracking completion
  uint32_t batch_id = m_next_batch_id_.fetch_add(1, std::memory_order_relaxed);
  auto batch = std::make_shared<BatchEntry>();
  batch->batch_id = batch_id;
  batch->count = count;
  batch->io_ctxs.reserve(count);
  batch->completed_count.store(0, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(m_batch_map_mutex_);
    m_batch_map_[batch_id] = batch;
  }

  for (int i = 0; i < count; ++i) {
    uint64_t lba = offsets[i] / sector_size;
    uint32_t lba_count = (total_lens[i] + sector_size - 1) / sector_size;
    void* buffer = reinterpret_cast<void*>(buf_ptrs[i]);

    IoContext* io_ctx = alloc_io_ctx();
    if (io_ctx == nullptr) {
      std::cerr
          << "[spdk_batch_io_submit] Failed to allocate IoContext at index "
          << i << "; marking batch " << batch_id << " slot failed" << std::endl;
      batch->alloc_fail_count += 1;
      batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }
    io_ctx->is_completed.store(false);
    io_ctx->status = 0;
    io_ctx->lba = lba;
    io_ctx->lba_count = lba_count;
    io_ctx->command_count = spdk_command_count_for(m_ctx_.ns, lba_count);
    io_ctx->op = op;
    io_ctx->buffer = buffer;
    io_ctx->batch = batch.get();
    io_ctx->engine = this;
    io_ctx->re_enqueue_count = 0;

    int rc = rte_ring_enqueue(m_io_ring_, reinterpret_cast<void*>(io_ctx));
    if (rc != 0) {
      std::cerr << "[spdk_batch_io_submit] Ring full at index " << i
                << "; marking batch " << batch_id << " slot failed"
                << std::endl;
      io_ctx->status = -ECANCELED;
      io_ctx->is_completed.store(true);
      batch->io_ctxs.push_back(io_ctx);
      batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }
    batch->io_ctxs.push_back(io_ctx);
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

  // Never block forever: stop spinning once every I/O completes.
  auto batch_deadline = std::chrono::steady_clock::now() + g_spdk_wait_timeout;
  while (batch->completed_count.load(std::memory_order_acquire) < total &&
         m_worker_alive_.load() &&
         std::chrono::steady_clock::now() < batch_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // If completion is still short, the batch did not finish in time (the worker
  // exited or the timeout elapsed); report it as a failure below.
  bool timed_out =
      batch->completed_count.load(std::memory_order_acquire) < total;

  // Overall batch status: 0 if every I/O succeeded, -1 if any failed or the
  // batch did not complete in time.
  int status = timed_out ? -1 : 0;
  if (batch->alloc_fail_count > 0) {
    status = -1;
  }
  for (IoContext* io_ctx : batch->io_ctxs) {
    if (io_ctx->status != 0) {
      status = -1;
      break;
    }
  }

  // Free only the I/Os this batch actually observed as completed (the worker no
  // longer references them). Any I/O still in-flight when the batch timed out
  // or the worker exited is owned by the worker and handed back for reclamation
  // instead of freeing it here, which would race the worker's/SPDK's completion
  // callback and cause a use-after-free.
  for (IoContext* io_ctx : batch->io_ctxs) {
    if (io_ctx->is_completed.load(std::memory_order_acquire)) {
      free_io_ctx(io_ctx);
    } else {
      register_abandoned_io_ctx(io_ctx);
    }
  }

  {
    std::lock_guard<std::mutex> lock(m_batch_map_mutex_);
    m_batch_map_.erase(batch_id);
  }

  return status;
}

// C wrapper functions for batched I/O
extern "C" {

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

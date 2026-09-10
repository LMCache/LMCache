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

// Struct to keep track of our connected remote resources
struct ConnectionContext {
  struct spdk_nvme_ctrlr* ctrlr = nullptr;
  struct spdk_nvme_ns* ns = nullptr;
  bool is_connected = false;
  uint32_t io_queue_depth = 0;
};

// Batch tracking structure.
struct BatchEntry {
  uint32_t batch_id;
  int count;                            // Number of I/Os in the batch
  std::atomic<int> completed_count{0};  // Number of I/Os that have completed
  // Owned IoContexts for this batch, in batch order.
  std::vector<IoContext*> io_ctxs;
};

// Structure to track the status of our asynchronous I/O operation
struct IoContext {
  std::atomic<bool> is_completed = false;
  int status = 0;               // To store IO completion status
  uint64_t lba;                 // Starting Logical Block Address
  uint32_t lba_count;           // Number of logical blocks
  IoOp op;                      // I/O operation type
  void* buffer = nullptr;       // Data buffer for read/write
  int re_enqueue_count = 0;     // To track re-enqueue count
  BatchEntry* batch = nullptr;  // To store batch info for batch submission
};

/* IO ring specifics shared between io worker and producer threads */
#define IO_RING_SIZE 4096
static struct rte_ring* g_io_ring = nullptr;

/* Max completions to reap in io worker */
#define MAX_COMPLETIONS_TO_WAIT (4)

/* Max times a failed submit is re-enqueued before the I/O is cancelled */
#define MAX_RE_ENQUEUE_COUNT 5

// Global batch tracking. Batch IDs are assigned sequentially starting at 1.
static std::atomic<uint32_t> g_next_batch_id{1};
static std::mutex g_batch_map_mutex;
static std::unordered_map<uint32_t, std::shared_ptr<BatchEntry>> g_batch_map;

// Global shutdown flag and worker threads
std::atomic<bool> m_shutdown_flag{false};
std::thread m_io_thread;
std::thread m_admin_thread;
ConnectionContext ctx;

// Resolved CPU core IDs for worker threads (computed during init_spdk)
static int g_io_worker_core = -1;
static int g_admin_worker_core = -1;

// In-flight I/O counter to track pending operations
static std::atomic<int> m_in_flight_count{0};

// Connection parameters (configurable from Python)
static std::string g_transport_type = "tcp";  // "tcp", "rdma" or "pcie"
static std::string g_target_ip = "127.0.0.1";
static std::string g_target_port = "4420";
static std::string g_target_nqn = "nqn.2016-06.io.spdk:cnode1";
static std::string g_pcie_addr =
    "";  // PCIe device address (e.g., "0000:01:00.0")

static bool probe_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid,
                     struct spdk_nvme_ctrlr_opts* opts) {
  // Request the maximum I/O queue depth (submission-queue size)
  opts->io_queue_size = 0xFFFFFFFFu;
  return true;
}

static void attach_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid,
                      struct spdk_nvme_ctrlr* ctrlr,
                      const struct spdk_nvme_ctrlr_opts* opts) {
  ConnectionContext* ctx = static_cast<ConnectionContext*>(cb_ctx);
  ctx->ctrlr = ctrlr;

  ctx->io_queue_depth = spdk_nvme_ctrlr_get_opts(ctrlr)->io_queue_size;

  uint32_t num_ns = spdk_nvme_ctrlr_get_num_ns(ctrlr);
  for (uint32_t i = 1; i <= num_ns; i++) {
    struct spdk_nvme_ns* ns = spdk_nvme_ctrlr_get_ns(ctrlr, i);
    if (ns == nullptr || !spdk_nvme_ns_is_active(ns)) {
      continue;
    }
    ctx->ns = ns;
    break;
  }

  ctx->is_connected = true;
}

/* Map a caller-supplied transport string ("tcp", "rdma") to the SPDK NVMf
 * transport type */
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
  m_in_flight_count.fetch_sub(1, std::memory_order_acq_rel);

  // Update batch completion count if part of a batch.
  if (context->batch != nullptr) {
    context->batch->completed_count.fetch_add(1, std::memory_order_acq_rel);
  }
}

SpdkIoEngineCore::SpdkIoEngineCore() {}

SpdkIoEngineCore::~SpdkIoEngineCore() {}

std::unique_ptr<SpdkIoEngineCore> make_SpdkIoEngineCore() {
  return std::make_unique<SpdkIoEngineCore>();
}

static int resolve_worker_cores(const char* core_mask) {
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
    g_io_worker_core = cores[0];
    g_admin_worker_core = cores[0];
  } else {
    // io_worker takes the highest core, admin_worker the second-highest.
    g_io_worker_core = cores.back();
    g_admin_worker_core = cores[cores.size() - 2];
  }

  return 0;
}

// C wrapper functions for ctypes FFI compatibility
extern "C" {

int core_init_spdk(void* core_ptr, const char* core_mask) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->init_spdk(core_mask);
}

void core_deinit_spdk(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  const_cast<SpdkIoEngineCore*>(core)->deinit_spdk();
}

void core_shutdown_spdk_workers(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  const_cast<SpdkIoEngineCore*>(core)->shutdown_spdk_workers();
}

uintptr_t core_allocate_spdk_memory(void* core_ptr, size_t size, size_t align,
                                    int numa_id) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->allocate_spdk_memory(size, align,
                                                                   numa_id);
}

void core_free_spdk_memory(void* core_ptr, uintptr_t buff) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  const_cast<SpdkIoEngineCore*>(core)->free_spdk_memory(buff);
}

int core_register_external_memory(void* core_ptr, uintptr_t ptr, size_t size) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->register_external_memory(ptr,
                                                                       size);
}

int core_unregister_external_memory(void* core_ptr, uintptr_t ptr,
                                    size_t size) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->unregister_external_memory(ptr,
                                                                         size);
}

/* Wrapper functions. `op` is IO_READ (0) or IO_WRITE (1). */
int core_spdk_io(void* core_ptr, uint64_t byte_offset, uint64_t byte_count,
                 void* buffer, int op) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->spdk_io(
      byte_offset, byte_count, buffer, static_cast<IoOp>(op));
}

int core_launch_spdk_workers(void* core_ptr, const char* transport_type,
                             const char* addr, const char* port,
                             const char* nqn) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->launch_spdk_workers(
      transport_type, addr, port, nqn);
}

int core_get_device_size(void* core_ptr, uint64_t* result_size) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  if (!core || !result_size) {
    std::cerr << "[core_get_device_size] Invalid arguments: core_ptr or "
                 "result_size is null"
              << std::endl;
    return -1;
  }
  return const_cast<SpdkIoEngineCore*>(core)->get_device_size(*result_size);
}

}  // extern "C"

int SpdkIoEngineCore::init_spdk(const char* core_mask) const {
  struct spdk_env_opts m_opts;
  int ret = -1;

  spdk_env_opts_init(&m_opts);
  m_opts.name = "lmcache_spdk";

  if ((core_mask == nullptr) || (core_mask[0] == '\0')) {
    std::cerr << "[init_spdk] Core Mask cannot be uninitialized" << std::endl;
    return -1;
  }

  // Resolve worker cores from the provided core mask; fail if it is empty.
  if (resolve_worker_cores(core_mask) != 0) {
    return -1;
  }

  // Use the resolved (possibly synthesized) core mask.
  m_opts.core_mask = core_mask;

  ret = spdk_env_init(&m_opts);
  if (ret < 0) {
    std::cerr << "[init_spdk] SPDK env init failed" << std::endl;
    return ret;
  }

  g_io_ring = rte_ring_create("g_io_ring", IO_RING_SIZE, 0, RING_F_SC_DEQ);
  if (!g_io_ring) {
    std::cerr << "[init_spdk] Failed to create rte_ring" << std::endl;
    return -1;
  }

  return ret;
}

void SpdkIoEngineCore::deinit_spdk() const {
  // Guard against a second deinit. The Python FFI __del__ also calls
  // deinit() after explicit cleanup nulls the engine reference.
  static bool deinitialized = false;
  if (deinitialized) {
    return;
  }
  deinitialized = true;

  if (g_io_ring != nullptr) {
    rte_ring_free(g_io_ring);
    g_io_ring = nullptr;
  }

  spdk_env_fini();
}

void SpdkIoEngineCore::shutdown_spdk_workers() const {
  m_shutdown_flag.store(true);

  if (m_io_thread.joinable()) {
    m_io_thread.join();
  }

  if (m_admin_thread.joinable()) {
    m_admin_thread.join();
  }
}

uintptr_t SpdkIoEngineCore::allocate_spdk_memory(size_t size, size_t align,
                                                 int numa_id) const {
  void* buffer = spdk_dma_zmalloc_socket(size, align, nullptr, numa_id);
  if (buffer == nullptr) {
    std::cerr << "[allocate_spdk_memory] Failed to allocate " << size
              << " bytes" << std::endl;
    return 0;
  }
  return reinterpret_cast<uintptr_t>(buffer);
}

void SpdkIoEngineCore::free_spdk_memory(uintptr_t buff) const {
  if (buff != 0) {
    void* buffer = reinterpret_cast<void*>(buff);
    spdk_dma_free(buffer);
  }
}

static void io_worker() {
  cpu_set_t cpuset;
  int rc = 0;
  IoContext* io_ctx = nullptr;
  struct spdk_nvme_qpair* qpair = nullptr;

  if (-1 == g_io_worker_core) {
    std::cerr << "[io_worker] core is not initialized" << std::endl;
    return;
  }
  CPU_ZERO(&cpuset);
  CPU_SET(g_io_worker_core, &cpuset);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    std::cerr << "[io_worker] Failed to set thread affinity" << std::endl;
    return;
  }

  if (g_transport_type == "pcie") {
    if (g_pcie_addr.empty()) {
      std::cerr << "[io_worker] PCIe address not configured" << std::endl;
      return;
    }
    if (connect_local_device(g_pcie_addr.c_str(), &ctx) != 0) {
      std::cerr << "[io_worker] Failed to connect to PCIe device" << std::endl;
      return;
    }
  } else {
    if (connect_remote_device(g_transport_type.c_str(), g_target_ip.c_str(),
                              g_target_port.c_str(), g_target_nqn.c_str(),
                              &ctx) != 0) {
      std::cerr << "[io_worker] Failed to connect to NVMe-oF target"
                << std::endl;
      return;
    }
  }

  if (0 == ctx.io_queue_depth) {
    std::cerr << "[io_worker] Invalid IO queue depth" << std::endl;
    return;
  }

  struct spdk_nvme_io_qpair_opts qpair_opts = {};
  spdk_nvme_ctrlr_get_default_io_qpair_opts(ctx.ctrlr, &qpair_opts,
                                            sizeof(qpair_opts));
  /* Initialize qpair size to controller queue depth to prevent re-enqueue in
   * the IO path */
  qpair_opts.io_queue_size = ctx.io_queue_depth;
  qpair_opts.io_queue_requests = ctx.io_queue_depth * 2;

  qpair_opts.opts_size = sizeof(qpair_opts);
  qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx.ctrlr, &qpair_opts,
                                         sizeof(qpair_opts));
  if (!qpair) {
    std::cerr << "[io_worker] Failed to allocate qpair" << std::endl;
    spdk_nvme_detach(ctx.ctrlr);
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
    if (m_shutdown_flag.load()) {
      break;
    }

    if (rte_ring_dequeue(g_io_ring, reinterpret_cast<void**>(&io_ctx)) != 0) {
      spdk_nvme_qpair_process_completions(qpair, MAX_COMPLETIONS_TO_WAIT);
      continue;
    }

    m_in_flight_count.fetch_add(1, std::memory_order_acq_rel);

    if (io_ctx->op == IO_WRITE) {
      rc = spdk_nvme_ns_cmd_write(ctx.ns, qpair, io_ctx->buffer, io_ctx->lba,
                                  io_ctx->lba_count, io_complete_cb, io_ctx, 0);
    } else {
      rc = spdk_nvme_ns_cmd_read(ctx.ns, qpair, io_ctx->buffer, io_ctx->lba,
                                 io_ctx->lba_count, io_complete_cb, io_ctx, 0);
    }

    if (rc != 0) {
      m_in_flight_count.fetch_sub(1, std::memory_order_acq_rel);
      // Respect the re-enqueue limit before retrying
      if (io_ctx->re_enqueue_count >= MAX_RE_ENQUEUE_COUNT) {
        std::cerr << "[io_worker] Cancelling I/O after " << MAX_RE_ENQUEUE_COUNT
                  << " failed re-enqueues (LBA=" << io_ctx->lba
                  << ", op=" << (io_ctx->op == IO_WRITE ? "write" : "read")
                  << ")" << std::endl;
        cancel_io_ctx(io_ctx);
      } else if (rte_ring_enqueue(g_io_ring, reinterpret_cast<void*>(io_ctx)) !=
                 0) {
        std::cerr << "[io_worker] Ring full" << std::endl;
        cancel_io_ctx(io_ctx);
      } else {
        io_ctx->re_enqueue_count++;
      }
    }

    if (m_in_flight_count.load(std::memory_order_acquire) >= 1) {
      spdk_nvme_qpair_process_completions(qpair, MAX_COMPLETIONS_TO_WAIT);
    }
    std::this_thread::yield();
  }

  const int k_drain_timeout_ms = 1000;  // 1 second timeout
  auto drain_start = std::chrono::steady_clock::now();

  while (rte_ring_dequeue(g_io_ring, reinterpret_cast<void**>(&io_ctx)) == 0) {
    std::cerr << "[io_worker] Cancelling I/O from ring (LBA=" << io_ctx->lba
              << ", op=" << (io_ctx->op == IO_WRITE ? "write" : "read") << ")"
              << std::endl;
    io_ctx->status = -ECANCELED;
    io_ctx->is_completed.store(true);
  }

  while (m_in_flight_count.load(std::memory_order_acquire) > 0) {
    spdk_nvme_qpair_process_completions(qpair, MAX_COMPLETIONS_TO_WAIT);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - drain_start)
                       .count();
    if (elapsed >= k_drain_timeout_ms) {
      std::cerr << "[io_worker] Timeout (" << k_drain_timeout_ms
                << "ms) waiting for " << m_in_flight_count.load()
                << " in-flight I/Os. Marking remaining as failed." << std::endl;
      break;
    }
    std::this_thread::yield();
  }

  spdk_nvme_ctrlr_free_io_qpair(qpair);
  spdk_nvme_detach(ctx.ctrlr);
}

static void admin_worker() {
  cpu_set_t cpuset;

  if (-1 == g_admin_worker_core) {
    std::cerr << "[admin_worker] core is not initialized" << std::endl;
    return;
  }

  CPU_ZERO(&cpuset);
  CPU_SET(g_admin_worker_core, &cpuset);

  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    std::cerr << "[admin_worker] Failed to set thread affinity" << std::endl;
    return;
  }

  while (true) {
    if (m_shutdown_flag.load()) {
      if (ctx.ctrlr) {
        spdk_nvme_ctrlr_process_admin_completions(ctx.ctrlr);
      }
      break;
    }

    if (!ctx.is_connected) {
      std::this_thread::yield();
      continue;
    }

    spdk_nvme_ctrlr_process_admin_completions(ctx.ctrlr);
    std::this_thread::yield();
  }
}

int SpdkIoEngineCore::set_connection_params(const char* transport_type,
                                            const char* addr, const char* port,
                                            const char* nqn) const {
  if (transport_type) {
    g_transport_type = transport_type;
  }

  if (g_transport_type == "pcie") {
    if (addr) {
      g_pcie_addr = addr;
    }
  } else {
    if (addr) g_target_ip = addr;
    if (port) g_target_port = port;
    if (nqn) g_target_nqn = nqn;
  }
  return 0;
}

int SpdkIoEngineCore::launch_spdk_workers(const char* transport_type,
                                          const char* addr, const char* port,
                                          const char* nqn) const {
  set_connection_params(transport_type, addr, port, nqn);
  m_io_thread = std::thread(io_worker);

  /* Wait for connection */
  sleep(1);

  /* Only launch admin worker for NVMe-oF */
  if (g_transport_type != "pcie") {
    m_admin_thread = std::thread(admin_worker);
  }

  return 0;
}

int SpdkIoEngineCore::spdk_io(uint64_t byte_offset, uint64_t byte_count,
                              void* buffer, IoOp op) const {
  if (!buffer) {
    std::cerr << "[spdk_io] Null buffer" << std::endl;
    return -1;
  }

  // Check if we have a connected namespace
  if (!ctx.is_connected || ctx.ns == nullptr) {
    std::cerr << "[spdk_io] No namespace connected. Call launch_spdk_workers "
                 "first."
              << std::endl;
    return -1;
  }

  uint32_t sector_size = spdk_nvme_ns_get_sector_size(ctx.ns);
  uint64_t lba = byte_offset / sector_size;
  uint64_t lba_count = (byte_count + sector_size - 1) / sector_size;

  auto io_ctx = new IoContext;
  io_ctx->is_completed.store(false);
  io_ctx->status = 0;
  io_ctx->lba = lba;
  io_ctx->lba_count = lba_count;
  io_ctx->op = op;
  io_ctx->buffer = buffer;

  int rc = rte_ring_enqueue(g_io_ring, reinterpret_cast<void*>(io_ctx));
  if (rc != 0) {
    std::cerr << "[spdk_io] Ring full" << std::endl;
    delete io_ctx;
    return -1;
  }

  while (!(io_ctx->is_completed.load()));

  if (io_ctx->status != 0) {
    delete io_ctx;
    return -1;
  }

  delete io_ctx;
  return 0;
}

int SpdkIoEngineCore::register_external_memory(uintptr_t ptr,
                                               size_t size) const {
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

int SpdkIoEngineCore::unregister_external_memory(uintptr_t ptr,
                                                 size_t size) const {
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

int SpdkIoEngineCore::get_device_size(uint64_t& result_size) const {
  if (!ctx.is_connected || ctx.ns == nullptr) {
    std::cerr << "[get_device_size] No namespace connected" << std::endl;
    return -1;
  }

  uint64_t num_sectors = spdk_nvme_ns_get_num_sectors(ctx.ns);
  uint32_t sector_size = spdk_nvme_ns_get_sector_size(ctx.ns);
  result_size = num_sectors * sector_size;

  return 0;
}

int SpdkIoEngineCore::spdk_batch_io_submit(uint64_t* offsets,
                                           uint64_t* total_lens,
                                           uint64_t* buf_ptrs, int count,
                                           IoOp op,
                                           uint32_t* batch_id_out) const {
  if (!ctx.is_connected || ctx.ns == nullptr) {
    std::cerr << "[spdk_batch_io_submit] No namespace connected" << std::endl;
    return -1;
  }

  if ((batch_id_out == nullptr) || (count <= 0)) {
    std::cerr << "[spdk_batch_io_submit] Invalid batch id buffer or count"
              << std::endl;
    return -1;
  }

  uint32_t sector_size = spdk_nvme_ns_get_sector_size(ctx.ns);

  // Allocate a batch entry for tracking completion
  uint32_t batch_id = g_next_batch_id.fetch_add(1, std::memory_order_relaxed);
  auto batch = std::make_shared<BatchEntry>();
  batch->batch_id = batch_id;
  batch->count = count;
  batch->io_ctxs.reserve(count);
  batch->completed_count.store(0, std::memory_order_relaxed);

  {
    std::lock_guard<std::mutex> lock(g_batch_map_mutex);
    g_batch_map[batch_id] = batch;
  }

  for (int i = 0; i < count; ++i) {
    uint64_t lba = offsets[i] / sector_size;
    uint32_t lba_count = (total_lens[i] + sector_size - 1) / sector_size;
    void* buffer = reinterpret_cast<void*>(buf_ptrs[i]);

    auto io_ctx = new IoContext;
    io_ctx->is_completed.store(false);
    io_ctx->status = 0;
    io_ctx->lba = lba;
    io_ctx->lba_count = lba_count;
    io_ctx->op = op;
    io_ctx->buffer = buffer;
    io_ctx->batch = batch.get();

    int rc = rte_ring_enqueue(g_io_ring, reinterpret_cast<void*>(io_ctx));
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

int SpdkIoEngineCore::spdk_wait_batch(uint32_t batch_id) const {
  std::shared_ptr<BatchEntry> batch;
  {
    std::lock_guard<std::mutex> lock(g_batch_map_mutex);
    auto it = g_batch_map.find(batch_id);
    if (it == g_batch_map.end()) {
      std::cerr << "[spdk_wait_batch] Unknown batch_id: " << batch_id
                << std::endl;
      return -1;
    }
    batch = it->second;
  }

  int total = batch->count;

  while (batch->completed_count.load(std::memory_order_acquire) < total) {
    std::this_thread::yield();
  }

  // Overall batch status: 0 if every I/O succeeded, -1 if any failed.
  int status = 0;
  for (IoContext* io_ctx : batch->io_ctxs) {
    if (io_ctx->status != 0) {
      status = -1;
      break;
    }
  }

  for (IoContext* io_ctx : batch->io_ctxs) {
    delete io_ctx;
  }

  {
    std::lock_guard<std::mutex> lock(g_batch_map_mutex);
    g_batch_map.erase(batch_id);
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
  return const_cast<SpdkIoEngineCore*>(core)->spdk_batch_io_submit(
      offsets, total_lens, buf_ptrs, count, static_cast<IoOp>(op),
      batch_id_out);
}

int core_spdk_wait_batch(void* core_ptr, uint32_t batch_id) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return core->spdk_wait_batch(batch_id);
}

}  // extern "C"

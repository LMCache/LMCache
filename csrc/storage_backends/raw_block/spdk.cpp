// SPDX-License-Identifier: Apache-2.0

#include "spdk.hpp"
#include <sched.h>
#include <pthread.h>
#include <chrono>
#include <sstream>
#include <cstring>
#include <vector>
#include <cstdlib>  // For getenv, stoi

// Struct to keep track of our connected remote resources
struct ConnectionContext {
  struct spdk_nvme_ctrlr* ctrlr = nullptr;
  struct spdk_nvme_ns* ns = nullptr;
  bool is_connected = false;
};

// Structure to track the status of our asynchronous I/O operation
struct IoContext {
  std::atomic<bool> is_completed = false;
  int status = 0;
  uint64_t lba;                   // Starting Logical Block Address
  uint32_t lba_count;             // Number of logical blocks
  enum { OP_READ, OP_WRITE } op;  // I/O operation type
  void* buffer = nullptr;         // Data buffer for read/write
};

// Lockless ring for IoContext passing between producer (spdk_write/spdk_read)
// and consumer (io_worker) Using SPDK's rte_ring for lockless, thread-safe
// enqueue/dequeue operations
#define IO_RING_SIZE 4096
static struct rte_ring* io_ring = nullptr;

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
static std::string g_transport_type = "tcp";  // "tcp" or "pcie"
static std::string g_target_ip = "127.0.0.1";
static std::string g_target_port = "4420";
static std::string g_target_nqn = "nqn.2019-04.pos:subsystem1";
static std::string g_pcie_addr =
    "";  // PCIe device address (e.g., "0000:01:00.0")

// SPDK core mask (configurable from Python)
// Default: empty string means SPDK uses all available cores
static bool probe_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid,
                     struct spdk_nvme_ctrlr_opts* opts) {
  return true;
}

static void attach_cb(void* cb_ctx, const struct spdk_nvme_transport_id* trid,
                      struct spdk_nvme_ctrlr* ctrlr,
                      const struct spdk_nvme_ctrlr_opts* opts) {
  ConnectionContext* ctx = static_cast<ConnectionContext*>(cb_ctx);
  ctx->ctrlr = ctrlr;

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

int connect_remote_tcp_ssd(const char* ip_addr, const char* port,
                           const char* nqn, ConnectionContext* ctx) {
  struct spdk_nvme_transport_id trid = {};

  trid.trtype = SPDK_NVME_TRANSPORT_TCP;
  trid.adrfam = SPDK_NVMF_ADRFAM_IPV4;

  snprintf(trid.traddr, sizeof(trid.traddr), "%s", ip_addr);
  snprintf(trid.trsvcid, sizeof(trid.trsvcid), "%s", port);
  snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", nqn);

  int rc = spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, nullptr);
  if (rc != 0 || !ctx->is_connected) {
    std::cerr << "Failed to connect to NVMe-oF TCP target" << std::endl;
    return -1;
  }

  return 0;
}

int connect_local_pcie_nvme(const char* pcie_addr, ConnectionContext* ctx) {
  struct spdk_nvme_transport_id trid = {};

  trid.trtype = SPDK_NVME_TRANSPORT_PCIE;
  trid.adrfam = SPDK_NVMF_ADRFAM_NOT_SPECIFIED;
  memset(trid.trsvcid, 0, sizeof(trid.trsvcid));
  snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);
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
}

/**
 * Write data to a remote NVMe-oF TCP SSD namespace.
 */
int write_to_remote_tcp_ssd(struct spdk_nvme_ns* ns,
                            struct spdk_nvme_qpair* qpair, void* buffer,
                            uint64_t lba, uint32_t lba_count) {
  if (!ns || !qpair || !buffer) {
    return -1;
  }

  struct IoContext context;
  context.is_completed.store(false);
  context.status = 0;
  context.lba = lba;
  context.lba_count = lba_count;
  context.op = IoContext::OP_WRITE;

  int rc = spdk_nvme_ns_cmd_write(ns, qpair, buffer, lba, lba_count,
                                  io_complete_cb, &context, 0);

  if (rc != 0) {
    return -1;
  }

  while (!(context.is_completed.load())) {
    spdk_nvme_qpair_process_completions(qpair, 4);
  }

  return context.status;
}

int read_from_remote_tcp_ssd(struct spdk_nvme_ns* ns,
                             struct spdk_nvme_qpair* qpair, void* buffer,
                             uint64_t lba, uint32_t lba_count) {
  if (!ns || !qpair || !buffer) {
    return -1;
  }

  struct IoContext context;
  context.is_completed.store(false);
  context.status = 0;
  context.lba = lba;
  context.lba_count = lba_count;
  context.op = IoContext::OP_READ;

  int rc = spdk_nvme_ns_cmd_read(ns, qpair, buffer, lba, lba_count,
                                 io_complete_cb, &context, 0);

  if (rc != 0) {
    return -1;
  }

  while (!(context.is_completed.load())) {
    spdk_nvme_qpair_process_completions(qpair, 4);
  }

  return context.status;
}

SpdkIoEngineCore::SpdkIoEngineCore() {}

SpdkIoEngineCore::~SpdkIoEngineCore() {}

std::unique_ptr<SpdkIoEngineCore> make_SpdkIoEngineCore() {
  return std::make_unique<SpdkIoEngineCore>();
}

// ========================================================================
// Global SPDK configuration (set before initialization)
// ========================================================================

// SPDK configuration options - settable from Python/storage backend
// Single core mask that controls all core assignments:
// - Cores in the mask are used for DPDK pollers
// - I/O worker and admin worker cores are derived from this mask

static std::string g_dpdk_core_mask =
    "";  // DPDK poller cores (e.g., "0x3f" for cores 0-5)

/**
 * Resolve worker core IDs from the DPDK core mask or environment variables.
 */
/**
 * Resolve worker core IDs from the DPDK core mask or environment variables.
 */
static void resolve_worker_cores(void) {
  unsigned long mask = 0;
  const char* io_core_env = std::getenv("LMCACHE_IO_WORKER_CORE");
  const char* admin_core_env = std::getenv("LMCACHE_ADMIN_WORKER_CORE");
  std::vector<int> cores;

  if (io_core_env != nullptr) {
    try {
      g_io_worker_core = std::stoi(io_core_env);
    } catch (...) {
      std::cerr << "[resolve_worker_cores] Invalid LMCACHE_IO_WORKER_CORE"
                << std::endl;
    }
  }

  if (admin_core_env != nullptr) {
    try {
      g_admin_worker_core = std::stoi(admin_core_env);
    } catch (...) {
      std::cerr << "[resolve_worker_cores] Invalid LMCACHE_ADMIN_WORKER_CORE"
                << std::endl;
    }
  }

  if (io_core_env != nullptr && admin_core_env != nullptr) {
    return;
  }

  if (!g_dpdk_core_mask.empty()) {
    try {
      mask = std::stoul(g_dpdk_core_mask, nullptr, 16);
    } catch (...) {
      std::cerr << "[resolve_worker_cores] Invalid core mask" << std::endl;
      mask = 0;
    }

    if (mask != 0) {
      for (int i = 0; i < 64; i++) {
        if (mask & (1UL << i)) {
          cores.push_back(i);
        }
      }

      if (cores.empty()) {
        std::cerr << "[resolve_worker_cores] No cores in mask" << std::endl;
      } else if (cores.size() == 1) {
        g_io_worker_core =
            (io_core_env == nullptr) ? cores[0] : g_io_worker_core;
        g_admin_worker_core =
            (admin_core_env == nullptr) ? cores[0] : g_admin_worker_core;
      } else {
        if (io_core_env == nullptr) {
          g_io_worker_core = cores.back();
        }
        if (admin_core_env == nullptr) {
          g_admin_worker_core =
              (cores.size() >= 3) ? cores[cores.size() - 2] : cores[0];
        }
      }
      return;
    }
  }

  if (io_core_env == nullptr) {
    g_io_worker_core = 23;
  }
  if (admin_core_env == nullptr) {
    g_admin_worker_core = 1;
  }

  // Warn if both workers are scheduled on the same core
  if (g_io_worker_core >= 0 && g_admin_worker_core >= 0 &&
      g_io_worker_core == g_admin_worker_core) {
    std::cout
        << "[resolve_worker_cores] WARNING: io_worker and admin_worker "
        << "threads are both scheduled on core " << g_io_worker_core
        << ". This may cause performance degradation due to thread "
        << "contention. Consider assigning them to different cores using "
        << "LMCACHE_IO_WORKER_CORE and LMCACHE_ADMIN_WORKER_CORE environment "
        << "variables." << std::endl;
  }
}

// ========================================================================
// C wrapper functions for ctypes FFI compatibility
// ========================================================================

extern "C" {

/**
 * Set the DPDK/SPDK poller core mask. Must be called before init_spdk().
 */
int core_set_dpdk_core_mask(const char* core_mask) {
  if (core_mask) {
    g_dpdk_core_mask = core_mask;
  } else {
    g_dpdk_core_mask.clear();
  }
  return 0;
}

/**
 * Get the resolved I/O worker core ID.
 */
int core_get_io_worker_core(void) { return g_io_worker_core; }

/**
 * Get the resolved admin worker core ID.
 */
int core_get_admin_worker_core(void) { return g_admin_worker_core; }

void* core_init_spdk(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return reinterpret_cast<void*>(
      const_cast<SpdkIoEngineCore*>(core)->init_spdk());
}

void core_deinit_spdk(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  const_cast<SpdkIoEngineCore*>(core)->deinit_spdk();
}

void core_shutdown_io_worker(void* core_ptr) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  const_cast<SpdkIoEngineCore*>(core)->shutdown_io_worker();
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

/* Wrapper function */
int core_spdk_write(void* core_ptr, uint64_t lba, uint32_t lba_count,
                    const uint8_t* buffer) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->spdk_write(lba, lba_count,
                                                         buffer);
}

int core_spdk_read(void* core_ptr, uint64_t lba, uint32_t lba_count,
                   uint8_t* buffer) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->spdk_read(lba, lba_count, buffer);
}

int core_spdk_write_external(void* core_ptr, uint64_t byte_offset,
                             uint64_t byte_count, const uint8_t* buffer) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->spdk_write_external(
      byte_offset, byte_count, buffer);
}

int core_spdk_read_external(void* core_ptr, uint64_t byte_offset,
                            uint64_t byte_count, uint8_t* buffer) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->spdk_read_external(
      byte_offset, byte_count, buffer);
}

int core_launch_io_worker(void* core_ptr, const char* transport_type,
                          const char* addr, const char* port, const char* nqn) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->launch_io_worker(transport_type,
                                                               addr, port, nqn);
}

int core_set_connection_params(void* core_ptr, const char* transport_type,
                               const char* addr, const char* port,
                               const char* nqn) {
  auto* core = static_cast<SpdkIoEngineCore*>(core_ptr);
  return const_cast<SpdkIoEngineCore*>(core)->set_connection_params(
      transport_type, addr, port, nqn);
}

/**
 * Get the device size in bytes from the connected NVMe namespace.
 * This queries the namespace for its number of sectors and sector size,
 * then calculates the total device size.
 *
 * @param core_ptr: Pointer to the SpdkIoEngineCore instance.
 * @param result_size: Output pointer to store the device size in bytes.
 * @return: 0 on success, -1 on failure.
 */
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

int SpdkIoEngineCore::init_spdk() const {
  struct spdk_env_opts m_opts;
  int ret = -1;

  spdk_env_opts_init(&m_opts);
  m_opts.name = "lmcache_spdk";

  resolve_worker_cores();

  if (!g_dpdk_core_mask.empty()) {
    m_opts.core_mask = g_dpdk_core_mask.c_str();
  }

  ret = spdk_env_init(&m_opts);
  if (ret < 0) {
    std::cerr << "[init_spdk] SPDK env init failed" << std::endl;
    return ret;
  }

  io_ring = rte_ring_create("io_ring", IO_RING_SIZE * sizeof(void*), 0,
                            RING_F_SP_ENQ | RING_F_SC_DEQ);
  if (!io_ring) {
    std::cerr << "[init_spdk] Failed to create rte_ring" << std::endl;
    return -1;
  }

  return ret;
}

void SpdkIoEngineCore::deinit_spdk() const { spdk_env_fini(); }

void SpdkIoEngineCore::shutdown_io_worker() const {
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
/**
 * I/O worker thread: connects to NVMe device and processes I/O operations.
 */
static void io_worker() {
  cpu_set_t cpuset;
  int rc = 0;
  IoContext* io_ctx = nullptr;
  struct spdk_nvme_qpair* qpair = nullptr;

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
    if (connect_local_pcie_nvme(g_pcie_addr.c_str(), &ctx) != 0) {
      std::cerr << "[io_worker] Failed to connect to PCIe device" << std::endl;
      return;
    }
  } else {
    if (connect_remote_tcp_ssd(g_target_ip.c_str(), g_target_port.c_str(),
                               g_target_nqn.c_str(), &ctx) != 0) {
      std::cerr << "[io_worker] Failed to connect to TCP target" << std::endl;
      return;
    }
  }

  qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx.ctrlr, nullptr, 0);
  if (!qpair) {
    std::cerr << "[io_worker] Failed to allocate qpair" << std::endl;
    spdk_nvme_detach(ctx.ctrlr);
    return;
  }

  while (true) {
    if (m_shutdown_flag.load()) {
      break;
    }

    if (rte_ring_dequeue(io_ring, reinterpret_cast<void**>(&io_ctx)) != 0) {
      spdk_nvme_qpair_process_completions(qpair, 4);
      continue;
    }

    m_in_flight_count.fetch_add(1, std::memory_order_acq_rel);

    if (io_ctx->op == IoContext::OP_WRITE) {
      rc = spdk_nvme_ns_cmd_write(ctx.ns, qpair, io_ctx->buffer, io_ctx->lba,
                                  io_ctx->lba_count, io_complete_cb, io_ctx, 0);
      if (rc != 0) {
        m_in_flight_count.fetch_sub(1, std::memory_order_acq_rel);
        rte_ring_enqueue(io_ring, reinterpret_cast<void*>(io_ctx));
      }
    } else {
      rc = spdk_nvme_ns_cmd_read(ctx.ns, qpair, io_ctx->buffer, io_ctx->lba,
                                 io_ctx->lba_count, io_complete_cb, io_ctx, 0);
      if (rc != 0) {
        m_in_flight_count.fetch_sub(1, std::memory_order_acq_rel);
        rte_ring_enqueue(io_ring, reinterpret_cast<void*>(io_ctx));
      }
    }

    if (m_in_flight_count.load(std::memory_order_acquire) > 1) {
      spdk_nvme_qpair_process_completions(qpair, 4);
    }
  }

  spdk_nvme_ctrlr_free_io_qpair(qpair);
  spdk_nvme_detach(ctx.ctrlr);
}

/**
 * Admin worker thread: processes admin command completions.
 */
static void admin_worker() {
  cpu_set_t cpuset;
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

/**
 * Set connection parameters for NVMe device (PCIe or TCP).
 */
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

/**
 * Launch I/O worker thread.
 */
int SpdkIoEngineCore::launch_io_worker(const char* transport_type,
                                       const char* addr, const char* port,
                                       const char* nqn) const {
  set_connection_params(transport_type, addr, port, nqn);
  m_io_thread = std::thread(io_worker);
  sleep(1);

  // Only launch admin worker for non-PCIe transports (NVMe-oF)
  // PCIe devices do not require the admin worker thread
  if (g_transport_type != "pcie") {
    m_admin_thread = std::thread(admin_worker);
  }

  return 0;
}

/**
 * Submit an async write operation to the I/O ring.
 */
int SpdkIoEngineCore::spdk_write(uint64_t lba, uint32_t lba_count,
                                 const uint8_t* buffer) const {
  if (!buffer) {
    std::cerr << "[spdk_write] Null buffer" << std::endl;
    return -1;
  }

  auto io_ctx = new IoContext;
  io_ctx->is_completed.store(false);
  io_ctx->status = 0;
  io_ctx->lba = lba;
  io_ctx->lba_count = lba_count;
  io_ctx->op = IoContext::OP_WRITE;
  io_ctx->buffer = reinterpret_cast<void*>(const_cast<uint8_t*>(buffer));

  int rc = rte_ring_enqueue(io_ring, reinterpret_cast<void*>(io_ctx));
  if (rc != 0) {
    std::cerr << "[spdk_write] Ring full" << std::endl;
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

/**
 * Submit an async read operation to the I/O ring.
 */
int SpdkIoEngineCore::spdk_read(uint64_t lba, uint32_t lba_count,
                                uint8_t* buffer) const {
  if (!buffer) {
    std::cerr << "[spdk_read] Null buffer" << std::endl;
    return -1;
  }

  auto io_ctx = new IoContext;
  io_ctx->is_completed.store(false);
  io_ctx->status = 0;
  io_ctx->lba = lba;
  io_ctx->lba_count = lba_count;
  io_ctx->op = IoContext::OP_READ;
  io_ctx->buffer = reinterpret_cast<void*>(buffer);

  int rc = rte_ring_enqueue(io_ring, reinterpret_cast<void*>(io_ctx));
  if (rc != 0) {
    std::cerr << "[spdk_read] Ring full" << std::endl;
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

int SpdkIoEngineCore::spdk_write_external(uint64_t byte_offset,
                                          uint64_t byte_count,
                                          const uint8_t* buffer) const {
  uint32_t sector_size;
  uint64_t lba, lba_count;

  if (!ctx.is_connected || ctx.ns == nullptr) {
    std::cerr << "[spdk_write_external] No namespace connected" << std::endl;
    return -1;
  }

  sector_size = spdk_nvme_ns_get_sector_size(ctx.ns);
  lba = byte_offset / sector_size;
  lba_count = (byte_count + sector_size - 1) / sector_size;

  return spdk_write(lba, static_cast<uint32_t>(lba_count), buffer);
}

/**
 * Get the total device size in bytes from the attached NVMe namespace.
 */
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

int SpdkIoEngineCore::spdk_read_external(uint64_t byte_offset,
                                         uint64_t byte_count,
                                         uint8_t* buffer) const {
  uint32_t sector_size;
  uint64_t lba, lba_count;

  // Check if we have a connected namespace
  if (!ctx.is_connected || ctx.ns == nullptr) {
    std::cerr << "[spdk_read_external] No namespace connected. Call "
                 "launch_io_worker first."
              << std::endl;
    return -1;
  }

  // Get sector size from the namespace and convert byte offset/count to LBA
  sector_size = spdk_nvme_ns_get_sector_size(ctx.ns);
  lba = byte_offset / sector_size;
  lba_count = (byte_count + sector_size - 1) / sector_size;  // Round up

  return spdk_read(lba, static_cast<uint32_t>(lba_count), buffer);
}

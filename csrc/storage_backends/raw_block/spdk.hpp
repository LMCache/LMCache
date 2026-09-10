// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>

#include <spdk/nvme.h>
#include <spdk/env.h>
#include <rte_ring.h>
#include <rte_mempool.h>

// Forward declarations
struct IoContext;
struct BatchEntry;
class SpdkIoEngineCore;

enum IoOp { IO_READ, IO_WRITE };

// Per-adapter connection context: the attached NVMe controller/namespace.
struct ConnectionContext {
  struct spdk_nvme_ctrlr* ctrlr = nullptr;
  struct spdk_nvme_ns* ns = nullptr;
  bool is_connected = false;
  uint32_t io_queue_depth = 0;
  uint32_t io_queue_requests = 0;
};

// Batch tracking structure.
struct BatchEntry {
  uint32_t batch_id;
  int count;                            // Number of I/Os in the batch
  std::atomic<int> completed_count{0};  // I/Os that have completed
  uint32_t alloc_fail_count{0};     // I/Os whose IoContext allocation failed
  std::vector<IoContext*> io_ctxs;  // Owned IoContexts, in batch order
};

// Tracks the status of one asynchronous I/O operation.
struct IoContext {
  std::atomic<bool> is_completed = false;
  int status = 0;               // IO completion status
  uint64_t lba = 0;             // Starting Logical Block Address
  uint32_t lba_count = 0;       // Number of logical blocks
  IoOp op = IO_READ;            // I/O operation type
  void* buffer = nullptr;       // Data buffer for read/write
  int re_enqueue_count = 0;     // Re-enqueue count
  uint32_t command_count = 0;   // NVMe child commands this key splits into
  BatchEntry* batch = nullptr;  // Batch info for batch submission
  SpdkIoEngineCore* engine = nullptr;  // Back-pointer to the owning engine
  bool from_mempool = false;  // True if this object came from the mempool
};

class SpdkIoEngineCore {
 public:
  SpdkIoEngineCore();
  ~SpdkIoEngineCore();

  /**
   * @brief Initialize the SPDK environment.
   *
   * @param core_mask Hex string of available cores (e.g., "0x3" for cores 0 and
   * 1). Must be non-empty.
   * @param mem_size_mb SPDK hugepage memory size in MB (0 = SPDK default).
   * @return 0 on success, a negative error code on failure.
   */
  int init_spdk(const char* core_mask, int mem_size_mb);
  void deinit_spdk();

  /** @brief Allocate SPDK (hugepage) memory. */
  uintptr_t allocate_spdk_memory(size_t size, size_t align = 4096,
                                 int numa_id = SPDK_ENV_SOCKET_ID_ANY);
  /** @brief Free memory allocated by allocate_spdk_memory. */
  void free_spdk_memory(uintptr_t buff);

  /**
   * @brief Synchronous read/write using externally registered memory at a byte
   * offset. Converts the byte offset/count to LBA internally (using the
   * attached NVMe namespace sector size) and blocks until the I/O completes.
   *
   * @param byte_offset Starting byte offset on the device.
   * @param byte_count Number of bytes to transfer.
   * @param buffer Data buffer (read destination or write source).
   * @param op IO_READ or IO_WRITE.
   * @return 0 on success, a negative error code on failure.
   */
  int spdk_io(uint64_t byte_offset, uint64_t byte_count, void* buffer, IoOp op);

  /**
   * @brief Release the in-flight reservation for a completed I/O.
   *
   * Called from the free-function SPDK completion callback, which holds only an
   * IoContext back-pointer to the engine, so it must go through this public
   * entry point rather than reaching into private members.
   *
   * @param command_count Number of child commands the completed I/O consumed.
   */
  void record_io_complete(uint32_t command_count);

  /** @brief Register an IoContext a caller abandoned after timing out. */
  void register_abandoned_io_ctx(IoContext* io_ctx);
  /** @brief Reclaim abandoned IoContexts during shutdown drain. */
  void drain_abandoned_io_ctxs();

  /**
   * @brief Launch io_worker as a thread pinned to a core. Connection
   * parameters are passed from Python via FFI. For NVMe-oF ("tcp"/"rdma")
   * ip_addr, port and nqn are used; for PCIe pcie_addr is used.
   */
  int launch_spdk_workers(
      const char* transport_type,  // "pcie", "tcp" or "rdma"
      const char* addr,            // IP for TCP, PCIe address for PCIe
      const char* port,            // Port for TCP (ignored for PCIe)
      const char* nqn              // NQN for TCP (ignored for PCIe)
  );

  /**
   * @brief Set connection parameters (call before launch_spdk_workers).
   *
   * transport_type is "pcie" for local NVMe, "tcp"/"rdma" for NVMe-oF. For
   * PCIe, addr = "0000:01:00.0" and port/nqn are ignored; for TCP/RDMA, addr is
   * the IP, port is the port, and nqn is the NQN.
   */
  int set_connection_params(const char* transport_type, const char* addr,
                            const char* port, const char* nqn);

  /** @brief Signal the I/O loop to shut down. */
  void shutdown_spdk_workers();

  /**
   * @brief Register externally allocated memory (e.g., hugepages) with SPDK.
   *
   * @param ptr Pointer to externally allocated memory.
   * @param size Size of the memory region in bytes.
   * @return 0 on success, -1 on failure.
   */
  int register_external_memory(uintptr_t ptr, size_t size);

  /**
   * @brief Unregister memory from SPDK.
   *
   * @param ptr Pointer to the registered memory buffer.
   * @param size Size of the memory region in bytes.
   * @return 0 on success, -1 on failure.
   */
  int unregister_external_memory(uintptr_t ptr, size_t size);

  /**
   * @brief Get the total device size in bytes from the attached NVMe namespace.
   *
   * @param result_size Reference to store the device size in bytes.
   * @return 0 on success, -1 if the namespace is not connected or on error.
   */
  int get_device_size(uint64_t& result_size);

  /**
   * @brief Get the maximum NVMe I/O queue depth for the attached controller.
   *
   * @param depth_out Output parameter that receives the queue depth (0 when the
   * controller has not connected yet).
   * @return 0 on success.
   */
  int get_max_io_depth(uint32_t& depth_out);

  /**
   * @brief Submit a batch of read/write operations asynchronously (io_uring
   * style). All operations are pushed to the io_worker ring up front and the
   * function returns immediately without waiting for completions, so the
   * device can overlap them.
   *
   * @param offsets Array of byte offsets for each operation.
   * @param total_lens Array of byte counts for each operation.
   * @param buf_ptrs Array of buffer pointers for each operation.
   * @param count Number of operations in the batch.
   * @param op IO_READ or IO_WRITE.
   * @param batch_id_out Output parameter that receives the assigned batch ID
   * on success (wait for it with spdk_wait_batch). Left untouched on failure.
   * @return 0 on success (batch_id_out is set); -1 on failure.
   */
  int spdk_batch_io_submit(uint64_t* offsets, uint64_t* total_lens,
                           uint64_t* buf_ptrs, int count, IoOp op,
                           uint32_t* batch_id_out);

  /**
   * @brief Wait for a batch of I/O operations to complete. Blocks until every
   * I/O in the batch has completed, then releases the batch's resources and
   * returns an overall status.
   *
   * @param batch_id The positive batch ID returned by spdk_batch_io_submit.
   * @return 0 if every I/O in the batch succeeded, -1 if any I/O failed or the
   * batch_id is unknown.
   */
  int spdk_wait_batch(uint32_t batch_id);

 private:
  struct rte_ring* m_io_ring_ = nullptr;
  struct rte_mempool* m_io_ctx_pool_ = nullptr;

  ConnectionContext m_ctx_;
  std::string m_transport_type_ = "tcp";  // "tcp", "rdma" or "pcie"
  std::string m_target_ip_ = "127.0.0.1";
  std::string m_target_port_ = "4420";
  std::string m_target_nqn_ = "nqn.2016-06.io.spdk:cnode1";
  std::string m_pcie_addr_;  // PCIe device address (e.g., "0000:01:00.0")

  std::atomic<bool> m_shutdown_{false};
  std::thread m_io_thread_;
  std::thread m_admin_thread_;

  int m_io_worker_core_ = -1;
  int m_admin_worker_core_ = -1;

  std::atomic<uint32_t> m_in_flight_{0};

  // List for tracking timed-out IoContexts
  std::mutex m_abandoned_mutex_;
  std::vector<IoContext*> m_abandoned_io_ctx_;

  // Flag to track io_worker existence
  std::atomic<bool> m_worker_alive_{true};

  std::atomic<uint32_t> m_next_batch_id_{1};
  std::mutex m_batch_map_mutex_;
  std::unordered_map<uint32_t, std::shared_ptr<BatchEntry>> m_batch_map_;

  // Process-global SPDK EAL reference counting. The EAL (spdk_env_init/fini) is
  // one-shot per process, so it is shared across adapters and only torn down
  // when the last one deinitializes.
  static std::atomic<int> g_eal_refcount_;
  static std::mutex g_eal_mutex_;

  int m_eal_refs_ = 0;
  std::atomic<bool> m_deinitialized_{false};

  void io_worker();
  void admin_worker();

  int resolve_worker_cores(const char* core_mask);

  IoContext* alloc_io_ctx();
  void free_io_ctx(IoContext* io_ctx);

  void release_eal_ref();

  // RAII guard that flips m_worker_alive_ to false when io_worker returns by
  // any path, so spdk_io/spdk_wait_batch waiters stop spinning via it.
  class IoWorkerAliveGuard {
   public:
    explicit IoWorkerAliveGuard(std::atomic<bool>* alive) : alive_(alive) {}
    ~IoWorkerAliveGuard() { alive_->store(false, std::memory_order_release); }

   private:
    std::atomic<bool>* alive_;
  };
};

// Factory to create a SpdkIoEngineCore instance.
//
// Returns a heap-allocated object the caller owns until free_SpdkIoEngineCore()
// is called.
extern "C" {
SpdkIoEngineCore* make_SpdkIoEngineCore();
void free_SpdkIoEngineCore(SpdkIoEngineCore* core);
}

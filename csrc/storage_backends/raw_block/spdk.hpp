// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
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
  // Written once during connection setup and read concurrently by the
  // io_worker, admin_worker, and submitter threads, so these are atomic to
  // avoid a data race.
  std::atomic<bool> is_connected = false;
  std::atomic<uint32_t> io_queue_depth = 0;
  std::atomic<uint32_t> io_queue_requests = 0;
  // Owning engine back-pointer, set once by SpdkIoEngineCore's constructor
  SpdkIoEngineCore* engine = nullptr;
};

// Batch tracking structure.
struct BatchEntry {
  uint32_t batch_id = 0;
  int count = 0;                        // Number of I/Os in the batch
  std::atomic<int> completed_count{0};  // I/Os that have completed
  uint32_t alloc_fail_count = 0;    // I/Os whose IoContext allocation failed
  std::vector<IoContext*> io_ctxs;  // Owned IoContexts, in batch order
};

// Tracks the status of one asynchronous I/O operation.
struct IoContext {
  std::atomic<bool> is_completed = false;
  int status = 0;               // IO completion status
  uint64_t lba = 0;             // Starting Logical Block Address
  uint32_t lba_count = 0;       // Number of logical blocks
  IoOp op = IO_READ;            // I/O operation type
  void* buffer = nullptr;       // Data buffer actually submitted for DMA. For
                                // a partial-sector I/O this is the owned,
                                // sector-aligned bounce buffer (see below).
  int re_enqueue_count = 0;     // Re-enqueue count
  uint32_t command_count = 0;   // NVMe child commands this key splits into
  BatchEntry* batch = nullptr;  // Batch info for batch submission
  SpdkIoEngineCore* engine = nullptr;  // Back-pointer to the owning engine
  bool from_mempool = false;      // True if this object came from the mempool
  void* caller_buffer = nullptr;  // Caller's original buffer (head base)
  void* bounce_buffer = nullptr;  // Owned, sector-sized unaligned-tail buffer
  uint64_t requested_bytes = 0;   // Read tail bytes to copy back (reads only)
};

class SpdkIoEngineCore {
 public:
  /**
   * @brief Default constructor. Leaves the engine unstarted; the caller must
   *  call init_spdk() then launch_spdk_workers() before issuing any I/O.
   */
  SpdkIoEngineCore();
  /**
   * @brief Destructor that runs the single de-initialization path, so the
   *  process-global SPDK resources are released even when deinit_spdk() was
   *  never called explicitly.
   */
  ~SpdkIoEngineCore();

  /**
   * @brief Initialize the SPDK environment.
   *
   * @param core_mask Hex string of available cores (e.g., "0x3" for cores 0 and
   * 1). Must be non-empty.
   * @param mem_size_mb SPDK hugepage memory size in MB. Must be positive; 0 or
   * negative is rejected.
   * @return 0 on success, a negative error code on failure.
   */
  int init_spdk(const char* core_mask, int mem_size_mb);
  /**
   * @brief The single de-initialization entry point: stop and join the worker
   *  threads (which tears the device down on the io_worker), then free the I/O
   *  ring and IoContext mempool and release this adapter's share of the
   *  process-global EAL (torn down when the last adapter deinitializes).
   *  Idempotent - a second call is a no-op.
   */
  void deinit_spdk();

  /**
   * @brief Allocate SPDK DMA (hugepage) memory.
   *
   * @param size Size of the memory to allocate in bytes.
   * @param align Alignment of the allocation in bytes (default 4096).
   * @param numa_id NUMA socket to allocate on, or SPDK_ENV_SOCKET_ID_ANY.
   * @return the allocated buffer as a uintptr_t, or 0 on failure.
   */
  uintptr_t allocate_spdk_memory(size_t size, size_t align = 4096,
                                 int numa_id = SPDK_ENV_SOCKET_ID_ANY);
  /**
   * @brief Free memory previously allocated by allocate_spdk_memory.
   *
   * @param buff Buffer returned by allocate_spdk_memory; a zero value is a
   * no-op.
   */
  void free_spdk_memory(uintptr_t buff);

  /**
   * @brief Synchronous read/write using externally registered memory at a byte
   * offset. Converts the byte offset/count to LBA internally (using the
   * attached NVMe namespace sector size) and blocks until the I/O completes.
   *
   * Each NVMe command is independently timed by SPDK's per-command timeout
   * callback; a command that never completes is aborted so this call fails
   * instead of blocking until the caller-side backstop.
   *
   * @param byte_offset Starting byte offset on the device.
   * @param byte_count Number of bytes to transfer.
   * @param buffer Data buffer (read destination or write source).
   * @param op IO_READ or IO_WRITE.
   * @return 0 on success, a negative error code on failure.
   */
  int spdk_io(uint64_t byte_offset, uint64_t byte_count, void* buffer, IoOp op);

  /**
   * @brief Decrement the in-flight I/O counter by the child-command count of a
   * completed I/O.
   *
   * Called from io_complete_cb, the SPDK NVMe completion callback, which only
   * has an IoContext back-pointer to the engine, so it must go through this
   * public entry point rather than reaching into private members.
   *
   * @param command_count Number of child commands the completed I/O consumed.
   */
  void record_io_complete(uint32_t command_count);

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
   * @brief Signal the I/O loop to stop and join both worker threads.
   *
   * Serialized by an internal mutex and idempotent, so the destructor, an
   * explicit close, and concurrent callers whose wait deadline expired can all
   * call it safely. It does not fail the controller itself: a caller that still
   * has an outstanding command fails it first (see fail_controller_now()), and
   * the io_worker fails it during teardown when anything is still in flight.
   *
   * Joining the io_worker guarantees that the single device teardown has
   * finished: the ring is cancelled, every submitted command has been aborted
   * or completed, the qpair is freed, and the controller is detached. Once this
   * returns, every IoContext is settled and no SPDK callback can run again, so
   * callers may free their IoContexts and recycle buffers/slots.
   */
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
   * @brief Get the maximum NVMe command transfer size (MDTS) for the attached
   *  namespace in bytes.
   *
   * @param transfer_out Output parameter that receives the transfer size in
   *  bytes (0 when the namespace has not connected yet).
   * @return 0 on success.
   */
  int get_max_transfer_size(uint32_t& transfer_out);

  /**
   * @brief Submit a batch of read/write operations asynchronously. All
   * operations are pushed to the io_worker ring up front and the function
   * returns immediately without waiting for completions, so the device can
   * overlap them. Each operation's NVMe command is independently subject to the
   * per-command timeout enforced by SPDK.
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
   * A command that exceeds the per-command timeout is aborted by SPDK and
   * completes with a failure status, so the batch settles with -1 rather than
   * relying on the caller-side backstop. A command whose abort cannot settle is
   * resolved by the single device teardown (see shutdown_spdk_workers()), which
   * this call runs when its own deadline expires.
   *
   * @param batch_id The positive batch ID returned by spdk_batch_io_submit.
   * @return 0 if every I/O in the batch succeeded; -1 if any I/O failed, the
   * batch_id is unknown, or the wait deadline expired with a command still
   * outstanding. -1 is always a confirmed outcome: before returning it, the
   * device teardown has cancelled ring entries and aborted every submitted
   * command, so buffers and slots may be recycled on 0 and on -1.
   */
  int spdk_wait_batch(uint32_t batch_id);

  /**
   * @brief Record that the controller must be failed, without failing it.
   *
   * Completion-context safe: SPDK's timeout and abort callbacks can run while
   * the controller lock is held (spdk_nvme_ctrlr_process_admin_completions()
   * takes it for the whole admin poll), so calling spdk_nvme_ctrlr_fail() from
   * them would self-deadlock. This only marks the fail as requested; the
   * io_worker observes it, stops, and performs the actual fail inside
   * teardown_io_device() -- outside any completion context.
   *
   * Called from timeout_cb (I/O or admin command) and spdk_abort_complete_cb.
   */
  void request_ctrlr_fail();

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

  // Handshake between launch_spdk_workers and the io_worker: the io_worker
  // signals this once device setup (connection + qpair allocation) finishes,
  // so the caller can wait for connection completion instead of a fixed sleep.
  std::mutex m_setup_mutex_;
  std::condition_variable m_setup_cv_;
  bool m_setup_complete_ = false;  // guarded by m_setup_mutex_
  int m_setup_result_ = 0;         // guarded by m_setup_mutex_ (0 = ready)

  // Serializes shutdown_spdk_workers() so the destructor, an explicit close,
  // and concurrent callers whose wait deadline expired never join the same
  // threads twice. The io_worker never takes it, so a joiner cannot deadlock
  // against the worker's own teardown.
  std::mutex m_teardown_mutex_;
  // Guards m_ctx_.ctrlr between fail_controller_now() (which a caller whose
  // wait deadline expired runs on its own thread) and the detach in
  // teardown_io_device(), so the controller is never failed after it is freed.
  // The io_worker does not hold m_teardown_mutex_ while taking this one, so the
  // two are never taken in opposite orders.
  std::mutex m_ctrlr_mutex_;
  // Fail protocol between SPDK completion contexts and the worker threads:
  //   Healthy       -- no fail requested.
  //   FailRequested -- a completion/timeout callback asked for a fail. Those
  //                    callbacks can run under the controller lock, which
  //                    spdk_nvme_ctrlr_fail() also takes, so they only mark it.
  //   Failed        -- spdk_nvme_ctrlr_fail() has actually run; see
  //                    fail_controller_now().
  enum class CtrlrState { Healthy, FailRequested, Failed };
  std::atomic<CtrlrState> m_ctrlr_state_{CtrlrState::Healthy};

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

  /**
   * @brief io_worker main loop.
   *
   * Pins to its core, connects the device, and allocates the I/O qpair; then
   * consumes the ring to issue reads/writes and reap completions until shutdown
   * or a controller fail, after which the single device teardown settles every
   * outstanding command.
   */
  void io_worker();
  /**
   * @brief admin_worker main loop.
   *
   * Periodically drives controller admin-completion queues (with a final flush
   * on shutdown) so out-of-band NVMe events are processed without blocking the
   * io_worker.
   */
  void admin_worker();

  /**
   * @brief Resolve the pinned worker cores from the hex core mask.
   *
   * A single core serves both workers, otherwise the highest core is the io
   * worker and the next-highest is the admin worker.
   *
   * @param core_mask Hex string of available cores (e.g., "0x3").
   * @return 0 on success, -1 if the mask is null, empty, unparsable, or has
   *         no cores set (e.g. "0x0").
   */
  int resolve_worker_cores(const char* core_mask);

  /**
   * @brief Populate the connection parameters (transport, target address/port,
   * NQN or PCIe address) the io_worker reads during device setup.
   *
   * launch_spdk_workers accepts these values from the caller and forwards them
   * here before starting the workers, so it is only ever called internally.
   *
   * @param transport_type "pcie" for local NVMe, "tcp"/"rdma" for NVMe-oF.
   * @param addr PCIe address for PCIe; IP for TCP/RDMA targets.
   * @param port Port for TCP/RDMA (ignored for PCIe).
   * @param nqn NQN for TCP/RDMA (ignored for PCIe).
   */
  int set_connection_params(const char* transport_type, const char* addr,
                            const char* port, const char* nqn);

  /**
   * @brief Allocate an IoContext for the next I/O.
   *
   * Draws from the preallocated pool (lock-free), falling back to a heap
   * allocation when the pool is exhausted.
   *
   * @return a freshly initialized IoContext, or nullptr when the pool is
   * exhausted and heap allocation fails.
   */
  IoContext* alloc_io_ctx();
  /**
   * @brief Release an IoContext back to its source.
   *
   * Returns it to the pool when it came from there, otherwise deletes it; a
   * null pointer is ignored.
   *
   * @param io_ctx The context to release.
   */
  void free_io_ctx(IoContext* io_ctx);

  /**
   * @brief Build the aligned-head and bounced-tail IoContexts for one
   * byte-range transfer and append them to ``out``.
   *
   * Allocates a sector-sized bounce buffer only when there is an unaligned
   * tail. On failure frees everything built, leaves ``out`` unchanged, and
   * returns -1. Shared by spdk_io and spdk_batch_io_submit.
   *
   * @param byte_offset Sector-aligned starting byte offset.
   * @param byte_count Number of bytes to transfer.
   * @param buffer Caller's data buffer.
   * @param op IO_READ or IO_WRITE.
   * @param out Vector to append the built contexts to.
   * @return 0 on success, -1 on allocation failure.
   */
  int append_io_contexts(uint64_t byte_offset, uint64_t byte_count,
                         void* buffer, IoOp op, std::vector<IoContext*>& out);

  /**
   * @brief Stamp one IoContext with its batch and enqueue it on the I/O ring.
   *
   * Shared by spdk_io (batch == nullptr) and spdk_batch_io_submit (batch set).
   * On a ring-full the command is marked completed with -ECANCELED; for a batch
   * it is also recorded and accounted so waiters still settle. Never blocks.
   *
   * @param io_ctx Pre-populated context to submit.
   * @param batch Owning batch, or nullptr for the standalone spdk_io path.
   * @return 0 on enqueue, -1 when the ring is full and the command was failed.
   */
  int enqueue_io_context(IoContext* io_ctx, BatchEntry* batch);

  /**
   * @brief Cancel an I/O that can no longer be retried.
   *
   * Marks the context failed/completed and accounts for it in its batch (if
   * any) so batch waiters observe the outcome.
   *
   * @param io_ctx The context to cancel.
   */
  void cancel_io_ctx(IoContext* io_ctx);

  /**
   * @brief Fail the controller now, exactly once, from a non-completion thread.
   *
   * Moves the controller state to Failed and calls spdk_nvme_ctrlr_fail() for
   * the first caller only, so a pending request_ctrlr_fail() or a concurrent
   * second caller neither skips nor repeats the fail. Must not be invoked from
   * an SPDK completion/timeout callback: those can run under the controller
   * lock and spdk_nvme_ctrlr_fail() takes that (non-recursive) lock. Callers
   * are teardown_io_device() and the spdk_io()/spdk_wait_batch() deadline
   * paths, which must unblock a worker stuck in a transport call before
   * joining.
   */
  void fail_controller_now();

  /**
   * @brief Report whether the engine has given up on the controller.
   *
   * Completion-context safe: reads the fail state without touching the
   * controller.
   *
   * @return true once a fail has been requested or performed.
   */
  bool ctrlr_failing() const;

  /**
   * @brief Tear down the I/O device on the io_worker thread.
   *
   * Single teardown routine for the device/qpair/controller layer. It runs on
   * the io_worker (qpair owner) in every case -- normal loop exit, controller
   * fail self-shutdown, and setup-failure early returns -- so it never races
   * the ring or qpair from another thread. It fails the controller when a
   * command is still outstanding or a fail was requested, signals shutdown,
   * gates out new submissions, cancels every entry still queued in the
   * single-consumer ring, joins the admin worker (leaving one poller for the
   * admin queue while the qpair is freed), frees the qpair, and detaches the
   * controller. Freeing the qpair aborts every outstanding tracker to its
   * completion callback, so every IoContext is settled when this returns.
   *
   * Tolerant of a null qpair (setup failed before allocation) and a null
   * controller.
   *
   * @param qpair The I/O qpair to free, or nullptr when setup never allocated
   *        it.
   */
  void teardown_io_device(struct spdk_nvme_qpair* qpair);
  /**
   * @brief Handle a failed NVMe submission.
   *
   * Releases the in-flight slot, then either cancels the I/O (re-enqueue limit
   * reached or the ring is full) or re-enqueues it for retry, aggressively
   * reaping completions under memory pressure (-ENOMEM).
   *
   * @param rc SPDK submission return code.
   * @param io_ctx The I/O whose submission failed.
   * @param current_inflight In-flight I/O count observed at submission time.
   * @param qpair The qpair the submission was attempted on.
   */
  void handle_submission_failure(int rc, IoContext* io_ctx,
                                 uint32_t current_inflight,
                                 struct spdk_nvme_qpair* qpair);

  /**
   * @brief Release one process-global EAL reference.
   *
   * When the last reference goes, the shared SPDK EAL is finalized.
   */
  void release_eal_ref();

  /**
   * @brief Record the io_worker setup outcome and wake launch_spdk_workers.
   *
   * Publishes the result under m_setup_mutex_ and notifies the setup condition
   * variable that launch_spdk_workers waits on.
   *
   * @param result 0 when device setup succeeded, a negative code on failure.
   */
  void signal_setup_complete(int result);

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

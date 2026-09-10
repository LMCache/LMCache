// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <chrono>

#include <spdk/nvme.h>
#include <spdk/env.h>
#include <rte_ring.h>
#include <rte_mempool.h>

// Forward declaration
struct IoContext;

enum IoOp { IO_READ, IO_WRITE };

class SpdkIoEngineCore {
 public:
  SpdkIoEngineCore();
  ~SpdkIoEngineCore();

  /**
   * Initialize the SPDK environment.
   *
   * @param core_mask: Hex string representing available cores (e.g., "0x3" for
   * cores 0 and 1). Must be non-empty
   * @return: 0 on success, negative error code on failure.
   */
  int init_spdk(const char* core_mask) const;
  void deinit_spdk() const;

  /* SPDK memory allocation/free functions */
  uintptr_t allocate_spdk_memory(size_t size, size_t align = 4096,
                                 int numa_id = SPDK_ENV_SOCKET_ID_ANY) const;
  void free_spdk_memory(uintptr_t buff) const;

  // Synchronous read or write using externally registered memory with a byte
  // offset. Converts the byte offset/count to LBA internally (using the
  // attached NVMe namespace sector size) and blocks until the I/O completes.
  //
  // @param byte_offset: Starting byte offset on the device.
  // @param byte_count: Number of bytes to transfer.
  // @param buffer: Data buffer (read destination or write source).
  // @param op: IO_READ or IO_WRITE.
  int spdk_io(uint64_t byte_offset, uint64_t byte_count, void* buffer,
              IoOp op) const;

  /**
   * Launch io_worker as a thread pinned to core 0
   * Connection parameters are passed from Python via FFI
   * For NVMe-oF ("tcp" or "rdma"): ip_addr, port, nqn are used
   * For PCIe: pcie_addr is used, other params are ignored
   */
  int launch_spdk_workers(
      const char* transport_type,  // "pcie", "tcp" or "rdma"
      const char* addr,            // IP for TCP, PCIe address for PCIe
      const char* port,            // Port for TCP (ignored for PCIe)
      const char* nqn              // NQN for TCP (ignored for PCIe)
  ) const;

  /**
   * Set connection parameters (called before launch_io_worker)
   * transport_type: "pcie" for local NVMe, "tcp" or "rdma" for NVMe-oF
   * For PCIe: addr = "0000:01:00.0", port and nqn are ignored
   * For TCP or RDMA: addr = IP, port = port, nqn = NQN
   */
  int set_connection_params(const char* transport_type, const char* addr,
                            const char* port, const char* nqn) const;

  // Shutdown signal for the I/O loop
  void shutdown_spdk_workers() const;

  /**
   * Register externally allocated memory (e.g., hugepages) with SPDK.
   *
   * @param ptr: Pointer to externally allocated memory buffer.
   * @param size: Size of the memory region in bytes.
   * @return: 0 on success, -1 on failure.
   */
  int register_external_memory(uintptr_t ptr, size_t size) const;

  /**
   * Unregister memory from SPDK.
   *
   * @param ptr: Pointer to the registered memory buffer.
   * @param size: Size of the memory region in bytes.
   * @return: 0 on success, -1 on failure.
   */
  int unregister_external_memory(uintptr_t ptr, size_t size) const;

  /**
   * Get the total device size in bytes from the attached NVMe namespace.
   *
   * @param result_size: Reference to store the device size in bytes.
   * @return: 0 on success, -1 if namespace is not connected or on error.
   */
  int get_device_size(uint64_t& result_size) const;

  /**
   * Submit a batch of read or write operations asynchronously (io_uring-style).
   *
   * All operations are submitted to the io_worker ring up front and the
   * function returns immediately without waiting for individual completions,
   * so the device can overlap them.
   *
   * @param offsets: Array of byte offsets for each operation.
   * @param total_lens: Array of byte counts for each operation.
   * @param buf_ptrs: Array of buffer pointers for each operation.
   * @param count: Number of operations in the batch.
   * @param op: IO_READ or IO_WRITE.
   * @param batch_id_out: Output parameter that receives the assigned batch ID
   *  on success (wait for it with spdk_wait_batch). Left untouched on failure.
   * @return: 0 on success (``batch_id_out`` is set); -1 on failure.
   */
  int spdk_batch_io_submit(uint64_t* offsets, uint64_t* total_lens,
                           uint64_t* buf_ptrs, int count, IoOp op,
                           uint32_t* batch_id_out) const;

  /**
   * Wait for a batch of I/O operations to complete.
   *
   * Blocks until every I/O in the batch has completed, then releases the
   * batch's resources and returns an overall status.
   *
   * @param batch_id: The positive batch ID returned by batch_io_submit.
   * @return: 0 if every I/O in the batch succeeded, -1 if any I/O failed or
   * the batch_id is unknown.
   */
  int spdk_wait_batch(uint32_t batch_id) const;
};

// Factory function to create SpdkIoEngineCore instance
// Exported with C linkage for ctypes compatibility
extern "C" {
std::unique_ptr<SpdkIoEngineCore> make_SpdkIoEngineCore();
}

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

// SPDK and DPDK headers - paths are configured via CMake -I flags
// The actual paths depend on the SPDK_ROOT and DPDK_ROOT build variables
#include <spdk/nvme.h>
#include <spdk/env.h>
#include <rte_ring.h>
#include <rte_mempool.h>

// Forward declaration
struct IoContext;

class SpdkIoEngineCore {
 public:
  SpdkIoEngineCore();
  ~SpdkIoEngineCore();

  // ========================================================================
  // SPDK Environment Configuration (must be called before init_spdk)
  // ========================================================================

  /**
   * Set the SPDK/DKDK core mask. This controls which CPU cores SPDK can use.
   * Must be called before init_spdk().
   *
   * @param core_mask: Hex string representing available cores (e.g., "0x3" for
   * cores 0 and 1) Empty string means SPDK uses all available cores.
   */
  void set_core_mask(const char* core_mask);

  /**
   * Set the SPDK memory size in MB. This controls how much hugepage
   * memory SPDK reserves during initialization via DPDK's -m flag.
   * Must be called before init_spdk().
   *
   * @param mem_size_mb: Memory size in megabytes (0 = use SPDK default).
   */
  void set_mem_size(int mem_size_mb);

  int init_spdk() const;
  void deinit_spdk() const;

  // ========================================================================
  // Worker Thread CPU Affinity Configuration
  // ========================================================================

  /**
   * Set the CPU core for the I/O worker thread.
   * The thread will be pinned to this core using pthread_setaffinity_np.
   *
   * @param io_worker_core: CPU core ID (e.g., 23)
   */
  void set_io_worker_core(int io_worker_core) const;

  /**
   * Set the CPU core for the admin worker thread.
   * The thread will be pinned to this core using pthread_setaffinity_np.
   *
   * @param admin_worker_core: CPU core ID (e.g., 1)
   */
  void set_admin_worker_core(int admin_worker_core) const;

  // Legacy SPDK memory allocation (using spdk_dma_zmalloc)
  uintptr_t allocate_spdk_memory(size_t size, size_t align = 4096,
                                 int numa_id = SPDK_ENV_SOCKET_ID_ANY) const;
  void free_spdk_memory(uintptr_t buff) const;

  // Async write: creates IoContext and pushes into the rte_ring for async I/O
  int spdk_write(uint64_t lba, uint32_t lba_count, const uint8_t* buffer) const;

  // Async read: creates IoContext and pushes into the rte_ring for async I/O
  int spdk_read(uint64_t lba, uint32_t lba_count, uint8_t* buffer) const;

  // Launch io_worker as a thread pinned to core 0
  // Connection parameters are passed from Python via FFI
  // For NVMe-oF (TCP): ip_addr, port, nqn are used
  // For PCIe: pcie_addr is used, other params are ignored
  int launch_io_worker(const char* transport_type,  // "pcie" or "tcp"
                       const char* addr,  // IP for TCP, PCIe address for PCIe
                       const char* port,  // Port for TCP (ignored for PCIe)
                       const char* nqn    // NQN for TCP (ignored for PCIe)
  ) const;

  // Set connection parameters (called before launch_io_worker)
  // transport_type: "pcie" for local NVMe, "tcp" for NVMe-oF
  // For PCIe: addr = "0000:01:00.0", port and nqn are ignored
  // For TCP: addr = IP, port = port, nqn = NQN
  int set_connection_params(const char* transport_type, const char* addr,
                            const char* port, const char* nqn) const;

  // Shutdown signal for the I/O loop
  void shutdown_io_worker() const;

  // ========================================================================
  // External Memory Registration (for hugepage-backed memory)
  // ========================================================================

  /**
   * Register externally allocated memory (e.g., hugepages) with SPDK.
   * This enables zero-copy I/O operations using SPDK's DMA engine.
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
   * Write using externally registered memory with byte offset.
   * This method accepts a byte offset and byte count, then internally
   * converts them to LBA and lba_count using the device's sector size.
   * The memory must have already been registered with SPDK.
   *
   * @param byte_offset: Starting byte offset on the device.
   * @param byte_count: Number of bytes to write.
   * @param buffer: Pointer to the data buffer (must be registered).
   * @return: 0 on success, -1 on failure.
   */
  int spdk_write_external(uint64_t byte_offset, uint64_t byte_count,
                          const uint8_t* buffer) const;

  /**
   * Read using externally registered memory with byte offset.
   * This method accepts a byte offset and byte count, then internally
   * converts them to LBA and lba_count using the device's sector size.
   * The memory must have already been registered with SPDK.
   *
   * @param byte_offset: Starting byte offset on the device.
   * @param byte_count: Number of bytes to read.
   * @param buffer: Pointer to the destination buffer (must be registered).
   * @return: 0 on success, -1 on failure.
   */
  int spdk_read_external(uint64_t byte_offset, uint64_t byte_count,
                         uint8_t* buffer) const;

  /**
   * Get the total device size in bytes from the attached NVMe namespace.
   * Queries the namespace for its number of sectors and sector size,
   * then calculates the total size.
   *
   * @param result_size: Reference to store the device size in bytes.
   * @return: 0 on success, -1 if namespace is not connected or on error.
   */
  int get_device_size(uint64_t& result_size) const;

 private:
  // Track registered memory regions for validation
  mutable std::unordered_map<uintptr_t, size_t> registered_regions_;
  mutable std::mutex regions_mutex_;
};

// Factory function to create SpdkIoEngineCore instance
// Exported with C linkage for ctypes compatibility
extern "C" {
std::unique_ptr<SpdkIoEngineCore> make_SpdkIoEngineCore();
}

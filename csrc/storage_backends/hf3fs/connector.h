// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
// Authors: Wenwen Chen <wenwen.chen@samsung.com>

#pragma once

#include "../connector_base.h"
#include "../connector_types.h"
#include <hf3fs_usrbio.h>
#include <filesystem>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// Third Party - sharded flat_hash_set for high-throughput key buffer
#include "sharded_flat_hash_set.h"

namespace lmcache {
namespace connector {

/**
 * Per-thread connection state for the 3FS connector.
 *
 * Each worker thread maintains its own:
 * - File descriptors (Linux FD + 3FS registered FD)
 * - Dual Ior (I/O rings) for read and write operations
 * - Dual Iov (I/O buffers) for read and write operations
 *
 * Design rationale:
 * - Dual Ior: 3FS Ior cannot handle read and write simultaneously
 * - Per-thread Iov: Avoids synchronization overhead and IB registration
 * complexity
 * - FD per I/O: Register on open, deregister on close for proper resource
 * management
 */
struct WorkerHf3fsConn {
  // File descriptors
  int fd = -1;              // Linux file descriptor
  bool registered = false;  // Whether hf3fs_reg_fd succeeded
  std::string file_path;    // Current file path being accessed

  // I/O Rings (separate for read and write)
  // 3FS documentation states Ior cannot handle both read and write
  // simultaneously
  hf3fs_ior read_ior;
  bool read_ior_initialized = false;
  hf3fs_ior write_ior;
  bool write_ior_initialized = false;

  // I/O Buffers (separate for read and write)
  // Per-thread buffers avoid synchronization overhead
  hf3fs_iov read_iov;
  bool read_iov_initialized = false;
  hf3fs_iov write_iov;
  bool write_iov_initialized = false;

  /**
   * Destructor - cleans up all 3FS resources.
   *
   * Cleanup order:
   * 1. Deregister FD from 3FS (only if registration succeeded and not already
   * done)
   * 2. Close Linux FD (if not already closed by close_file)
   * 3. Destroy Ior structures (read, then write)
   * 4. Destroy Iov structures (read, then write)
   */
  ~WorkerHf3fsConn() {
    if (registered) {
      hf3fs_dereg_fd(fd);
    }
    if (fd >= 0) {
      ::close(fd);
    }
    if (read_ior_initialized) {
      hf3fs_iordestroy(&read_ior);
    }
    if (write_ior_initialized) {
      hf3fs_iordestroy(&write_ior);
    }
    if (read_iov_initialized) {
      hf3fs_iovdestroy(&read_iov);
    }
    if (write_iov_initialized) {
      hf3fs_iovdestroy(&write_iov);
    }
  }
};

/**
 * Native C++ connector for 3FS (Fire-Flyer File System).
 *
 * This connector provides high-performance KV cache storage using the 3FS
 * Usrbio API. It supports both MP (multiprocess) and Non-MP modes, multi-path
 * load balancing, and per-thread I/O resources for optimal concurrency.
 *
 * Features:
 * - GIL-free operations for true concurrency
 * - Per-thread Ior/Iov for parallel read/write
 * - Multi-path load balancing based on chunk_hash
 * - FD registration per I/O operation
 * - Mock layer for SDK-independent development
 *
 * Example usage:
 * @code
 *   auto connector = std::make_unique<Hf3fsConnector>(
 *       "/mnt/3fs",                    // mount_point
 *       "/mnt/3fs/path1,/mnt/3fs/path2", // base_paths
 *       8,                             // num_workers
 *       256,                           // ior_entries
 *       0,                             // io_depth
 *       -1,                            // numa_id
 *       209715200,                     // iov_size (200MB)
 *       200                            // time_out (ms)
 *   );
 * @endcode
 */
class Hf3fsConnector : public ConnectorBase<WorkerHf3fsConn> {
 public:
  /**
   * Construct a new Hf3fsConnector.
   *
   * @param mount_point 3FS mount point directory (must exist and be valid)
   * @param base_paths Comma-separated subdirectories under mount_point
   * @param num_workers Number of worker threads (default 8)
   * @param ior_entries Max concurrent requests per Ior (default 256, range
   * [128, 1024])
   * @param io_depth Batch control parameter (default 0, range [-128, 128])
   * @param numa_id NUMA node ID (default -1 = current node)
   * @param iov_size Per-thread I/O buffer size in bytes (default 200MB, range
   * [100MB, 2GB])
   * @param time_out I/O timeout in milliseconds (default 200)
   * @param enable_key_buffer Enable in-memory key buffer for accelerated
   * exists lookups (default true)
   * @param worker_pool_config Per-operation worker lane configuration
+  * (default empty, uses shared worker pool). Maps lane keys
+  * ("lookup", "retrieve", "store", "delete") to worker counts.
   * @throws std::runtime_error if mount_point is invalid or base_paths are not
   * subdirectories
   * @throws std::runtime_error if ior_entries, io_depth, or iov_size are out of
   * range
   */
  Hf3fsConnector(std::string mount_point, std::string base_paths,
                 int num_workers, int ior_entries = 256, int io_depth = 0,
                 int numa_id = -1, size_t iov_size = 209715200,
                 int time_out = 200, bool enable_key_buffer = true,
                 WorkerPoolConfig worker_pool_config = {});

  ~Hf3fsConnector() override = default;

 protected:
  WorkerHf3fsConn create_connection() override;
  void do_single_get(WorkerHf3fsConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerHf3fsConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerHf3fsConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerHf3fsConn& conn, const std::string& key) override;
  void do_batch_exists(WorkerHf3fsConn& conn, const Request& req) override;
  void shutdown_connections() override;

 private:
  // Configuration
  std::string mount_point_;
  std::vector<std::string> base_paths_;
  int ior_entries_;
  int io_depth_;
  int numa_id_;
  size_t iov_size_;
  int time_out_;

  // Key Buffer (for accelerated exists lookups)
  // Uses sharded flat_hash_set with per-shard locks for high write throughput
  concurrent_hash::sharded_flat_hash_set<std::string> key_buffer_;
  const bool buffer_enabled_;

  void scan_and_build_buffer_();
  void buffer_scan_worker_(const std::string& base_path,
                           std::unordered_set<std::string>& local_set);
  void buffer_add_(const std::string& key);
  void buffer_remove_(const std::string& key);
  bool buffer_contains_(const std::string& key) const;

  // Path selection and key encoding
  const std::string& select_base_path(const std::string& key) const;
  static std::string key_to_filename(const std::string& key);
  std::string key_to_path(const std::string& key);

  // Ior/Iov initialization
  void init_read_ior(hf3fs_ior& ior);
  void init_write_ior(hf3fs_ior& ior);
  void init_read_iov(hf3fs_iov& iov);
  void init_write_iov(hf3fs_iov& iov);

  // File operations
  void open_file(WorkerHf3fsConn& conn, const std::string& file_path,
                 bool for_write);
  void close_file(WorkerHf3fsConn& conn);
  void read_file(WorkerHf3fsConn& conn, hf3fs_ior& ior, hf3fs_iov& iov,
                 void* buf, size_t len);
  void write_file(WorkerHf3fsConn& conn, hf3fs_ior& ior, hf3fs_iov& iov,
                  const void* buf, size_t len);
};

}  // namespace connector
}  // namespace lmcache

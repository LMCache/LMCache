// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <blkio.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "../connector_base.h"

namespace lmcache {
namespace connector {

// Per-worker connection state for the blkio connector.
// Each worker gets its own blkio handle (io_uring instance) so
// I/O submissions never contend on a shared queue.
struct BlkioWorkerConn {
  struct blkio* handle = nullptr;
  struct blkioq* queue = nullptr;

  ~BlkioWorkerConn() {
    if (handle) {
      blkio_destroy(&handle);
    }
  }

  // Move-only: the blkio handle is not copyable.
  BlkioWorkerConn() = default;
  BlkioWorkerConn(BlkioWorkerConn&& o) noexcept
      : handle(o.handle), queue(o.queue) {
    o.handle = nullptr;
    o.queue = nullptr;
  }
  BlkioWorkerConn& operator=(BlkioWorkerConn&& o) noexcept {
    if (this != &o) {
      if (handle) blkio_destroy(&handle);
      handle = o.handle;
      queue = o.queue;
      o.handle = nullptr;
      o.queue = nullptr;
    }
    return *this;
  }
  BlkioWorkerConn(const BlkioWorkerConn&) = delete;
  BlkioWorkerConn& operator=(const BlkioWorkerConn&) = delete;
};

class BlkioConnector : public ConnectorBase<BlkioWorkerConn> {
 public:
  /**
   * Construct a blkio connector for block device I/O.
   *
   * @param device_path  Path to the block device (e.g. /dev/nvme0n1).
   * @param num_workers  Number of worker threads (each gets its own
   *                     io_uring instance via libblkio).
   * @param direct_io    Enable O_DIRECT (default true).
   */
  BlkioConnector(std::string device_path, int num_workers,
                 bool direct_io = true);
  ~BlkioConnector() override;

 protected:
  BlkioWorkerConn create_connection() override;
  void do_single_get(BlkioWorkerConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(BlkioWorkerConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(BlkioWorkerConn& conn, const std::string& key) override;
  void shutdown_connections() override;

 private:
  // Parse a key string to extract the byte offset on the block device.
  // Expected key format: "...@<offset_hex>" where the last '@'-delimited
  // field is the hex-encoded byte offset.
  static uint64_t key_to_offset(const std::string& key);

  // Map (register) a DRAM buffer with a blkio handle, perform I/O, then
  // unmap.  This mirrors the NIXL pattern of per-I/O buffer registration.
  static void map_do_io_unmap(struct blkio* handle, struct blkioq* queue,
                              bool is_read, uint64_t device_offset, void* buf,
                              size_t len);

  std::string device_path_;
  bool direct_io_;
};

}  // namespace connector
}  // namespace lmcache

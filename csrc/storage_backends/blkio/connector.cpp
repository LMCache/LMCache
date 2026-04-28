// SPDX-License-Identifier: Apache-2.0

#include "connector.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace lmcache {
namespace connector {

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

uint64_t BlkioConnector::key_to_offset(const std::string& key) {
  // Key format from NativeConnectorL2Adapter:
  //   "{model}@{kv_rank:08x}@{chunk_hash_hex}"
  //
  // For the blkio backend the chunk_hash_hex field encodes the
  // byte offset on the block device (written as a hex string by
  // the Python layer).
  size_t last_sep = key.rfind('@');
  if (last_sep == std::string::npos || last_sep + 1 >= key.size()) {
    throw std::runtime_error("blkio: invalid key format (no '@'): " + key);
  }
  const char* hex_start = key.c_str() + last_sep + 1;
  char* end = nullptr;
  errno = 0;
  uint64_t offset = std::strtoull(hex_start, &end, 16);
  if (errno != 0 || end == hex_start) {
    throw std::runtime_error("blkio: bad hex offset in key: " + key);
  }
  return offset;
}

void BlkioConnector::map_do_io_unmap(struct blkio* handle, struct blkioq* queue,
                                     bool is_read, uint64_t device_offset,
                                     void* buf, size_t len) {
  // O_DIRECT and io_uring buffer registration require page-aligned
  // buffers.  When the caller's buffer is not aligned we use a
  // page-aligned bounce buffer and copy data to/from it.
  static constexpr size_t kPageSize = 4096;
  const bool aligned = (reinterpret_cast<uintptr_t>(buf) % kPageSize == 0) &&
                       (len % kPageSize == 0);

  void* io_buf = buf;
  void* bounce_buf = nullptr;
  const size_t original_len = len;

  if (!aligned) {
    // Round len up to page boundary for the bounce allocation so
    // the region itself satisfies alignment requirements.
    size_t alloc_len = (len + kPageSize - 1) & ~(kPageSize - 1);
    int rc = posix_memalign(&bounce_buf, kPageSize, alloc_len);
    if (rc != 0) {
      throw std::runtime_error(std::string("blkio: posix_memalign failed: ") +
                               strerror(rc));
    }
    io_buf = bounce_buf;
    if (!is_read) {
      std::memcpy(bounce_buf, buf, len);
      // Zero the padding so we don't write uninitialised bytes.
      if (alloc_len > len) {
        std::memset(static_cast<char*>(bounce_buf) + len, 0, alloc_len - len);
      }
    }
    // Use the padded length for the region registration and I/O.
    len = alloc_len;
  }

  // 1. Register the DRAM buffer with the blkio instance.
  struct blkio_mem_region region;
  region.addr = io_buf;
  region.len = len;
  region.fd = -1;
  region.fd_offset = 0;

  int ret = blkio_map_mem_region(handle, &region);
  if (ret < 0) {
    free(bounce_buf);
    throw std::runtime_error(
        std::string("blkio: blkio_map_mem_region failed: ") + strerror(-ret));
  }

  // 2. Submit the I/O and wait for completion.
  struct blkio_completion comp;
  if (is_read) {
    blkioq_read(queue, device_offset, io_buf, static_cast<uint32_t>(len), &comp,
                0);
  } else {
    blkioq_write(queue, device_offset, io_buf, static_cast<uint32_t>(len),
                 &comp, 0);
  }

  ret = blkioq_do_io(queue, &comp, 1, 1, nullptr);
  if (ret < 0) {
    blkio_unmap_mem_region(handle, &region);
    free(bounce_buf);
    throw std::runtime_error(std::string("blkio: blkioq_do_io failed: ") +
                             strerror(-ret));
  }

  if (comp.ret < 0) {
    blkio_unmap_mem_region(handle, &region);
    free(bounce_buf);
    throw std::runtime_error(std::string("blkio: I/O completion error: ") +
                             strerror(-comp.ret));
  }

  // 3. Unmap the buffer.
  blkio_unmap_mem_region(handle, &region);

  // 4. Copy read data back and free the bounce buffer.
  if (bounce_buf) {
    if (is_read) {
      std::memcpy(buf, bounce_buf, original_len);
    }
    free(bounce_buf);
  }
}

// ---------------------------------------------------------------
// BlkioConnector
// ---------------------------------------------------------------

BlkioConnector::BlkioConnector(std::string device_path, int num_workers,
                               bool direct_io)
    : ConnectorBase(num_workers),
      device_path_(std::move(device_path)),
      direct_io_(direct_io) {
  if (device_path_.empty()) {
    throw std::runtime_error("blkio: device_path must not be empty");
  }

  start_workers();  // IMPORTANT: call at END of constructor
}

BlkioConnector::~BlkioConnector() { close(); }

BlkioWorkerConn BlkioConnector::create_connection() {
  BlkioWorkerConn conn;

  // Create a new blkio instance using the io_uring driver.
  int ret = blkio_create("io_uring", &conn.handle);
  if (ret < 0) {
    throw std::runtime_error(
        std::string("blkio: blkio_create(io_uring) failed: ") + strerror(-ret));
  }

  // Set the block device path.
  ret = blkio_set_str(conn.handle, "path", device_path_.c_str());
  if (ret < 0) {
    std::string err = std::string("blkio: failed to set path '") +
                      device_path_ + "': " + strerror(-ret);
    blkio_destroy(&conn.handle);
    throw std::runtime_error(err);
  }

  // Enable direct I/O if requested.
  if (direct_io_) {
    ret = blkio_set_bool(conn.handle, "direct", true);
    if (ret < 0) {
      std::string err =
          std::string("blkio: failed to enable direct I/O: ") + strerror(-ret);
      blkio_destroy(&conn.handle);
      throw std::runtime_error(err);
    }
  }

  // Connect and start the io_uring instance.
  ret = blkio_connect(conn.handle);
  if (ret < 0) {
    std::string err =
        std::string("blkio: blkio_connect failed: ") + strerror(-ret);
    blkio_destroy(&conn.handle);
    throw std::runtime_error(err);
  }

  ret = blkio_start(conn.handle);
  if (ret < 0) {
    std::string err =
        std::string("blkio: blkio_start failed: ") + strerror(-ret);
    blkio_destroy(&conn.handle);
    throw std::runtime_error(err);
  }

  // Get queue 0 for this handle.
  conn.queue = blkio_get_queue(conn.handle, 0);
  if (!conn.queue) {
    std::string err = "blkio: blkio_get_queue(0) returned null";
    blkio_destroy(&conn.handle);
    throw std::runtime_error(err);
  }

  return conn;
}

void BlkioConnector::do_single_get(BlkioWorkerConn& conn,
                                   const std::string& key, void* buf,
                                   size_t len, size_t chunk_size) {
  uint64_t offset = key_to_offset(key);
  map_do_io_unmap(conn.handle, conn.queue, /*is_read=*/true, offset, buf, len);
}

void BlkioConnector::do_single_set(BlkioWorkerConn& conn,
                                   const std::string& key, const void* buf,
                                   size_t len, size_t chunk_size) {
  uint64_t offset = key_to_offset(key);
  // blkioq_write expects a void* (non-const) per the libblkio API.
  map_do_io_unmap(conn.handle, conn.queue, /*is_read=*/false, offset,
                  const_cast<void*>(buf), len);
}

bool BlkioConnector::do_single_exists(BlkioWorkerConn& conn,
                                      const std::string& key) {
  // Block devices don't have a native "key exists" concept.
  // The Python layer is responsible for tracking which offsets
  // have been written (via its index/metadata).  At the C++
  // level we always return false and let the Python side
  // handle existence through its own bookkeeping.
  (void)conn;
  (void)key;
  return false;
}

void BlkioConnector::shutdown_connections() {
  // BlkioWorkerConn destructor handles blkio_destroy.
  // Nothing extra needed here.
}

}  // namespace connector
}  // namespace lmcache

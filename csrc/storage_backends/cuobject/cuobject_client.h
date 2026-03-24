// SPDX-License-Identifier: Apache-2.0
//
// C++ wrapper around the NVIDIA cuObjClient library.
//
// Linked against libcuobjclient.so at build time.  The cuObjClient C++
// class (from <cuobjclient.h>) is constructed directly -- no
// dlopen/dlsym.
//
// Callback model (per official cuObjClient API v1.0.0)
// ----------------------------------------------------
// cuObjClient uses a CUObjIOOps / CUObjOps_t struct with GET and PUT
// callbacks.  During cuObjGet() / cuObjPut(), the library invokes these
// callbacks one or more times (for buffers that exceed
// MaxRequestCallbackSize).  Each callback receives a cufileRDMAInfo_t*
// containing the RDMA descriptor string (desc_str) that serves as the
// x-amz-rdma-token.
//
// This wrapper captures the descriptor from the callback into a
// std::string that is returned to the Python layer.
//
// PUT token size patching
// -----------------------
// For PUT operations, the RDMA descriptor's size field (2nd colon-
// delimited field) is patched with the actual payload size.  This
// matches the behaviour of the EMCECS/aws-c-s3 CRT cuObject plugin.
//
// Thread safety
// -------------
// prepare_put / prepare_get are serialised with an internal mutex so
// that the callback-captured descriptor cannot be clobbered by a
// concurrent call.  register_pool / deregister_pool are expected to
// be called only during init / shutdown.

#pragma once

#include "cuobject_api.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace lmcache {
namespace cuobject {

class CuObjectClient {
 public:
  /// Create a cuObject client with the given transport protocol.
  ///
  /// Constructs a cuObjClient instance (build-time linked).
  ///
  /// @param proto  Transport protocol (default: CUOBJ_PROTO_RDMA_DC_V1).
  /// @throws std::runtime_error on client creation failure.
  explicit CuObjectClient(int proto = CUOBJ_PROTO_RDMA_DC_V1);

  ~CuObjectClient();

  // Non-copyable, non-movable (owns cuObjClient instance).
  CuObjectClient(const CuObjectClient &) = delete;
  CuObjectClient &operator=(const CuObjectClient &) = delete;
  CuObjectClient(CuObjectClient &&) = delete;
  CuObjectClient &operator=(CuObjectClient &&) = delete;

  /// Register a contiguous memory region for RDMA.
  ///
  /// Wraps cuMemObjGetDescriptor(ptr, size).  Supports system memory,
  /// CUDA managed memory, and CUDA device memory.
  ///
  /// @param ptr   Start address of the memory region.
  /// @param size  Byte size.  Must be < CUOBJ_MAX_MEMORY_REG_SIZE (4 GiB).
  /// @returns (ptr, size) pair that serves as the registration handle.
  /// @throws std::runtime_error on registration failure or size overflow.
  std::pair<uintptr_t, size_t> register_pool(uintptr_t ptr, size_t size);

  /// Deregister a previously registered memory region.
  ///
  /// Wraps cuMemObjPutDescriptor(ptr).  Must be called after all I/O
  /// operations on this memory have completed.
  ///
  /// @param ptr  Must match the address used in register_pool().
  /// @returns 0 on success, error code otherwise.  Does **not** throw.
  int deregister_pool(uintptr_t ptr) noexcept;

  /// Prepare an RDMA-accelerated PUT operation.
  ///
  /// Internally calls cuObjPut which invokes the PUT callback with the
  /// RDMA descriptor.  The descriptor's size field is patched with the
  /// actual payload size (matching CRT plugin behaviour).
  ///
  /// @param ptr         Data pointer within the registered pool.
  /// @param size        Byte size of the data.
  /// @param offset      Object offset (reserved, default 0).
  /// @param buf_offset  Buffer offset from base (default 0).
  /// @returns The x-amz-rdma-token header value.
  /// @throws std::runtime_error on failure or if no descriptor received.
  std::string prepare_put(uintptr_t ptr, size_t size, int64_t offset = 0,
                          int64_t buf_offset = 0);

  /// Prepare an RDMA-accelerated GET operation.
  ///
  /// Internally calls cuObjGet which invokes the GET callback with the
  /// RDMA descriptor.
  ///
  /// @param ptr         Destination pointer within the registered pool.
  /// @param size        Expected byte size of the data.
  /// @param offset      Object offset (reserved, default 0).
  /// @param buf_offset  Buffer offset from base (default 0).
  /// @returns The x-amz-rdma-token header value.
  /// @throws std::runtime_error on failure or if no descriptor received.
  std::string prepare_get(uintptr_t ptr, size_t size, int64_t offset = 0,
                          int64_t buf_offset = 0);

  /// Check if the client is connected and ready for operations.
  ///
  /// @returns true if connected, false otherwise.
  bool is_connected() const;

  /// Get the maximum callback chunk size for registered memory.
  ///
  /// If an I/O request exceeds this size, the callback will be invoked
  /// multiple times (once per chunk).
  ///
  /// @param ptr  Start address of registered memory.
  /// @returns Maximum callback size in bytes, or -1 on error.
  ssize_t get_max_callback_size(uintptr_t ptr) const;

  /// Destroy the cuObject client.  Safe to call multiple times.
  ///
  /// @returns 0 on success (or if already closed).  Does **not** throw.
  int close() noexcept;

 private:
  /// Static GET callback handed to the cuObject library.
  /// Extracts the RDMA descriptor from cufileRDMAInfo_t.
  static ssize_t get_callback(const void *handle, char *ptr, size_t size,
                               loff_t offset,
                               const cufileRDMAInfo_t *rdma_info);

  /// Static PUT callback handed to the cuObject library.
  /// Extracts the RDMA descriptor from cufileRDMAInfo_t.
  static ssize_t put_callback(const void *handle, const char *ptr,
                               size_t size, loff_t offset,
                               const cufileRDMAInfo_t *rdma_info);

  /// Patch the size field in an RDMA descriptor for PUT operations.
  ///
  /// The RDMA token format is "<proto>:<size_hex>:<rdma_fields...>".
  /// This overwrites the size field (2nd colon-delimited field) with
  /// zero-padded lowercase hex of the actual payload size, matching
  /// the CRT cuObject plugin's behaviour.
  static void patch_put_token_size(std::string &token, size_t payload_size);

  std::unique_ptr<cuObjClient> client_;  ///< cuObject client instance
  CUObjOps_t ops_{};  ///< Callback struct (must outlive client)

  /// Captured RDMA descriptor from the most recent callback invocation.
  std::string captured_descriptor_;
  /// Whether a callback was received during the last I/O operation.
  bool descriptor_received_ = false;

  std::mutex mutex_;  ///< Serialises prepare_put / prepare_get
};

}  // namespace cuobject
}  // namespace lmcache

// SPDX-License-Identifier: Apache-2.0
//
// C++ wrapper around the NVIDIA cuObjClient library.
//
// Linked against libcuobjclient.so at build time.  The cuObjClient C++
// class (from <cuobjclient.h>) is constructed directly.
//
// Token generation model (per official cuObjClient API v1.0.0)
// ------------------------------------------------------------
// RDMA tokens are generated via cuMemObjGetRDMAToken(), which returns
// an opaque descriptor string encoding the RDMA memory address, keys,
// and connection info.  The caller injects this string as the
// x-amz-rdma-token HTTP header in the S3 PUT/GET request.
//
// After copying the descriptor string, cuMemObjPutRDMAToken() frees the
// library-allocated memory.  The token content remains valid because the
// underlying RDMA memory registration (via cuMemObjGetDescriptor) is
// still active.
//
// Pool tracking
// -------------
// register_pool() stores the base address and size of the registered
// memory pool.  prepare_put / prepare_get accept a data pointer
// anywhere within the pool and automatically compute the buffer_offset
// as (data_ptr - pool_base) for cuMemObjGetRDMAToken().
//
// Thread safety
// -------------
// cuMemObjGetRDMAToken() writes to a caller-provided output pointer, so
// concurrent calls from different threads each get their own descriptor
// allocation.  No internal mutex is needed.

#pragma once

#include "cuobject_api.h"

#include <cstdint>
#include <memory>
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
  /// Also stores the pool base address and size so that prepare_put /
  /// prepare_get can compute the buffer_offset automatically.
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

  /// Generate an RDMA token for a PUT operation.
  ///
  /// Calls cuMemObjGetRDMAToken() with CUOBJ_PUT.  The returned token
  /// string is the value for the x-amz-rdma-token HTTP header.
  ///
  /// @param data_ptr  Data pointer within the registered pool.
  /// @param size      Byte size of the data to upload.
  /// @returns The x-amz-rdma-token header value.
  /// @throws std::runtime_error on failure or if data_ptr is outside pool.
  std::string prepare_put(uintptr_t data_ptr, size_t size);

  /// Generate an RDMA token for a GET operation.
  ///
  /// Calls cuMemObjGetRDMAToken() with CUOBJ_GET.  The returned token
  /// string is the value for the x-amz-rdma-token HTTP header.
  ///
  /// @param data_ptr  Destination pointer within the registered pool.
  /// @param size      Expected byte size of the data to download.
  /// @returns The x-amz-rdma-token header value.
  /// @throws std::runtime_error on failure or if data_ptr is outside pool.
  std::string prepare_get(uintptr_t data_ptr, size_t size);

  /// Check if the client is connected and ready for operations.
  ///
  /// @returns true if connected, false otherwise.
  bool is_connected() const;

  /// Destroy the cuObject client.  Safe to call multiple times.
  ///
  /// @returns 0 on success (or if already closed).  Does **not** throw.
  int close() noexcept;

 private:
  /// Generate an RDMA token for the given operation type.
  ///
  /// Calls cuMemObjGetRDMAToken(), copies the descriptor string, then
  /// frees the library allocation via cuMemObjPutRDMAToken().
  ///
  /// @param data_ptr  Pointer within the registered pool.
  /// @param size      Byte size of the operation.
  /// @param op_type   CUOBJ_GET or CUOBJ_PUT.
  /// @returns The RDMA token string.
  /// @throws std::runtime_error on failure.
  std::string generate_token(uintptr_t data_ptr, size_t size, int op_type);

  std::unique_ptr<cuObjClient> client_;  ///< cuObject client instance
  CUObjOps_t ops_{};  ///< Callback struct (required by constructor, unused)

  uintptr_t pool_base_ = 0;  ///< Base address of the registered pool
  size_t pool_size_ = 0;     ///< Byte size of the registered pool
};

}  // namespace cuobject
}  // namespace lmcache

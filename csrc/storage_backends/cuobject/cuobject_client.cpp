// SPDX-License-Identifier: Apache-2.0
#include "cuobject_client.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lmcache {
namespace cuobject {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

CuObjectClient::CuObjectClient(int proto) {
  // The constructor requires a CUObjOps_t reference.  Since we use
  // cuMemObjGetRDMAToken() for token generation (not cuObjPut/cuObjGet),
  // the callbacks are unused.  Zero-initialise the struct.
  std::memset(&ops_, 0, sizeof(ops_));

  client_ = std::make_unique<cuObjClient>(
      ops_, static_cast<cuObjProto_t>(proto));
}

CuObjectClient::~CuObjectClient() { close(); }

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::pair<uintptr_t, size_t> CuObjectClient::register_pool(uintptr_t ptr,
                                                            size_t size) {
  if (size >= CUOBJ_MAX_MEMORY_REG_SIZE) {
    std::ostringstream oss;
    oss << "Memory registration size " << size
        << " exceeds CUOBJ_MAX_MEMORY_REG_SIZE (4 GiB)";
    throw std::runtime_error(oss.str());
  }

  cuObjErr_t rc = client_->cuMemObjGetDescriptor(
      reinterpret_cast<void *>(ptr), size);
  if (static_cast<int>(rc) != CU_OBJ_SUCCESS) {
    std::ostringstream oss;
    oss << "cuMemObjGetDescriptor failed (ptr=0x" << std::hex << ptr
        << ", size=" << std::dec << size
        << "): error " << static_cast<int>(rc);
    throw std::runtime_error(oss.str());
  }

  // Store pool bounds for buffer_offset computation in generate_token().
  pool_base_ = ptr;
  pool_size_ = size;

  return {ptr, size};
}

int CuObjectClient::deregister_pool(uintptr_t ptr) noexcept {
  if (!client_) {
    // Already closed — nothing to deregister.
    pool_base_ = 0;
    pool_size_ = 0;
    return CU_OBJ_SUCCESS;
  }
  cuObjErr_t rc =
      client_->cuMemObjPutDescriptor(reinterpret_cast<void *>(ptr));
  if (ptr == pool_base_) {
    pool_base_ = 0;
    pool_size_ = 0;
  }
  return static_cast<int>(rc);
}

std::string CuObjectClient::prepare_put(uintptr_t data_ptr, size_t size) {
  return generate_token(data_ptr, size, CUOBJ_OP_PUT);
}

std::string CuObjectClient::prepare_get(uintptr_t data_ptr, size_t size) {
  return generate_token(data_ptr, size, CUOBJ_OP_GET);
}

bool CuObjectClient::is_connected() const {
  if (!client_) return false;
  return client_->isConnected();
}

int CuObjectClient::close() noexcept {
  if (!client_) return CU_OBJ_SUCCESS;

  // Deregister any active pool before destroying the client to avoid
  // leaking the RDMA memory registration (cuMemObjPutDescriptor).
  int rc = CU_OBJ_SUCCESS;
  if (pool_base_ != 0) {
    rc = deregister_pool(pool_base_);
  }

  client_.reset();
  return rc;
}

// ---------------------------------------------------------------------------
// Private: RDMA token generation
//
// cuMemObjGetRDMAToken() generates an RDMA descriptor string for a
// sub-region of the registered memory pool.  The descriptor encodes the
// RDMA memory address, keys, and connection info that the server uses
// to perform RDMA_READ (PUT) or RDMA_WRITE (GET).
//
// The buffer_offset is computed as (data_ptr - pool_base_) so that the
// Python caller can simply pass the MemoryObj's data pointer without
// needing to know the pool base address.
//
// After copying the descriptor string, cuMemObjPutRDMAToken() frees
// the library-allocated memory.  The token content is still valid for
// the S3 request because the underlying RDMA memory registration
// (from cuMemObjGetDescriptor) remains active.
// ---------------------------------------------------------------------------

std::string CuObjectClient::generate_token(uintptr_t data_ptr, size_t size,
                                           int op_type) {
  if (pool_base_ == 0 || pool_size_ == 0) {
    throw std::runtime_error(
        "No memory pool registered.  Call register_pool() first.");
  }
  if (data_ptr < pool_base_ || data_ptr + size > pool_base_ + pool_size_) {
    std::ostringstream oss;
    oss << "Data region [0x" << std::hex << data_ptr << ", 0x"
        << (data_ptr + size) << ") is outside registered pool [0x"
        << pool_base_ << ", 0x" << (pool_base_ + pool_size_) << ")";
    throw std::runtime_error(oss.str());
  }

  size_t buffer_offset = data_ptr - pool_base_;
  char *desc_str = nullptr;

  cuObjErr_t rc = client_->cuMemObjGetRDMAToken(
      reinterpret_cast<void *>(pool_base_), size, buffer_offset,
      static_cast<cuObjOpType_t>(op_type), &desc_str);

  if (static_cast<int>(rc) != CU_OBJ_SUCCESS || desc_str == nullptr) {
    std::ostringstream oss;
    oss << "cuMemObjGetRDMAToken failed (ptr=0x" << std::hex << data_ptr
        << ", size=" << std::dec << size
        << ", offset=" << buffer_offset
        << ", op=" << (op_type == CUOBJ_OP_PUT ? "PUT" : "GET")
        << "): error " << static_cast<int>(rc);
    throw std::runtime_error(oss.str());
  }

  // Copy the descriptor string and free the library allocation.
  std::string token(desc_str);
  client_->cuMemObjPutRDMAToken(desc_str);

  return token;
}

}  // namespace cuobject
}  // namespace lmcache

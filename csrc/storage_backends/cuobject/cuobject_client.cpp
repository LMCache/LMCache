// SPDX-License-Identifier: Apache-2.0
#include "cuobject_client.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lmcache {
namespace cuobject {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

CuObjectClient::CuObjectClient(int proto) {
  // Set up GET/PUT callbacks per the official cuObjClient API.
  // The callbacks extract RDMA descriptors from cufileRDMAInfo_t and
  // store them in captured_descriptor_ for the calling prepare_* method.
  ops_.get = &CuObjectClient::get_callback;
  ops_.put = &CuObjectClient::put_callback;

  // Construct the cuObjClient instance directly (build-time linked).
  client_ = std::make_unique<cuObjClient>(
      ops_, static_cast<cuObjProto_t>(proto));
}

CuObjectClient::~CuObjectClient() { close(); }

// ---------------------------------------------------------------------------
// GET/PUT callbacks
//
// Called synchronously by cuObjGet() / cuObjPut().  Each callback
// receives the cufileRDMAInfo_t containing the RDMA descriptor string.
// We copy desc_str into captured_descriptor_ for the calling
// prepare_put / prepare_get method.
//
// The handle parameter is an opaque cookie.  We pass 'this' as the
// ctx argument to cuObjGet/cuObjPut; the library wraps it in handle.
// cuObjClient::getCtx(handle) unwraps it.
//
// Note: For large transfers, callbacks may be invoked multiple times.
// We capture only the last descriptor.  For LMCache's KV cache
// workload, transfers are single-chunk (within MaxRequestCallbackSize).
// ---------------------------------------------------------------------------

ssize_t CuObjectClient::get_callback(const void *handle, char * /*ptr*/,
                                     size_t size, loff_t /*offset*/,
                                     const cufileRDMAInfo_t *rdma_info) {
  if (!rdma_info || rdma_info->desc_len <= 0 || !rdma_info->desc_str) {
    return -1;
  }

  auto *self =
      static_cast<CuObjectClient *>(cuObjClient::getCtx(handle));
  if (!self) return -1;

  size_t copy_len = static_cast<size_t>(rdma_info->desc_len);
  // Strip trailing NUL if the library includes it in desc_len
  // (some versions do, some don't -- the CRT plugin handles both).
  if (copy_len > 0 && rdma_info->desc_str[copy_len - 1] == '\0') {
    --copy_len;
  }

  self->captured_descriptor_.assign(rdma_info->desc_str, copy_len);
  self->descriptor_received_ = true;
  return static_cast<ssize_t>(size);
}

ssize_t CuObjectClient::put_callback(const void *handle,
                                     const char * /*ptr*/, size_t size,
                                     loff_t /*offset*/,
                                     const cufileRDMAInfo_t *rdma_info) {
  if (!rdma_info || rdma_info->desc_len <= 0 || !rdma_info->desc_str) {
    return -1;
  }

  auto *self =
      static_cast<CuObjectClient *>(cuObjClient::getCtx(handle));
  if (!self) return -1;

  size_t copy_len = static_cast<size_t>(rdma_info->desc_len);
  if (copy_len > 0 && rdma_info->desc_str[copy_len - 1] == '\0') {
    --copy_len;
  }

  self->captured_descriptor_.assign(rdma_info->desc_str, copy_len);
  self->descriptor_received_ = true;
  return static_cast<ssize_t>(size);
}

// ---------------------------------------------------------------------------
// PUT token size patching
//
// The RDMA descriptor format is "<proto>:<size_hex>:<rdma_fields...>".
// For PUT operations the CRT cuObject plugin patches the 2nd colon-
// delimited field with the actual payload size, zero-padded to the
// original width in lowercase hex.  We replicate this behaviour.
// ---------------------------------------------------------------------------

void CuObjectClient::patch_put_token_size(std::string &token,
                                          size_t payload_size) {
  if (token.empty()) return;

  // Locate the first two ':' delimiters.
  size_t first_colon = token.find(':');
  if (first_colon == std::string::npos) return;

  size_t second_colon = token.find(':', first_colon + 1);
  if (second_colon == std::string::npos) return;

  size_t field_start = first_colon + 1;
  size_t width = second_colon - field_start;
  if (width == 0 || width > 16) return;  // sanity bound

  char buf[17];
  int n = std::snprintf(buf, sizeof(buf), "%0*zx",
                        static_cast<int>(width), payload_size);
  if (n > 0 && static_cast<size_t>(n) >= width) {
    token.replace(field_start, width, buf, width);
  }
}

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
  return {ptr, size};
}

int CuObjectClient::deregister_pool(uintptr_t ptr) noexcept {
  // Official API: cuMemObjPutDescriptor(ptr) -- no size argument.
  cuObjErr_t rc =
      client_->cuMemObjPutDescriptor(reinterpret_cast<void *>(ptr));
  return static_cast<int>(rc);
}

std::string CuObjectClient::prepare_put(uintptr_t ptr, size_t size,
                                        int64_t offset, int64_t buf_offset) {
  std::lock_guard<std::mutex> lock(mutex_);
  captured_descriptor_.clear();
  descriptor_received_ = false;

  // Pass 'this' as the user context so our PUT callback can store
  // the RDMA descriptor back into this instance.
  // Official API: ssize_t cuObjPut(ctx, ptr, size, offset, buf_offset)
  ssize_t rc = client_->cuObjPut(
      static_cast<void *>(this), reinterpret_cast<void *>(ptr), size,
      static_cast<loff_t>(offset), static_cast<loff_t>(buf_offset));
  if (rc < 0) {
    std::ostringstream oss;
    oss << "cuObjPut failed (ptr=0x" << std::hex << ptr
        << ", size=" << std::dec << size << "): error " << rc;
    throw std::runtime_error(oss.str());
  }
  if (!descriptor_received_ || captured_descriptor_.empty()) {
    throw std::runtime_error(
        "cuObjPut succeeded but RDMA descriptor callback was not invoked");
  }

  // Patch the size field in the RDMA descriptor to match the actual
  // payload size.  Required for PUT per the CRT plugin convention.
  patch_put_token_size(captured_descriptor_, size);

  return captured_descriptor_;
}

std::string CuObjectClient::prepare_get(uintptr_t ptr, size_t size,
                                        int64_t offset, int64_t buf_offset) {
  std::lock_guard<std::mutex> lock(mutex_);
  captured_descriptor_.clear();
  descriptor_received_ = false;

  // Official API: ssize_t cuObjGet(ctx, ptr, size, offset, buf_offset)
  ssize_t rc = client_->cuObjGet(
      static_cast<void *>(this), reinterpret_cast<void *>(ptr), size,
      static_cast<loff_t>(offset), static_cast<loff_t>(buf_offset));
  if (rc < 0) {
    std::ostringstream oss;
    oss << "cuObjGet failed (ptr=0x" << std::hex << ptr
        << ", size=" << std::dec << size << "): error " << rc;
    throw std::runtime_error(oss.str());
  }
  if (!descriptor_received_ || captured_descriptor_.empty()) {
    throw std::runtime_error(
        "cuObjGet succeeded but RDMA descriptor callback was not invoked");
  }

  return captured_descriptor_;
}

bool CuObjectClient::is_connected() const {
  if (!client_) return false;
  return client_->isConnected();
}

ssize_t CuObjectClient::get_max_callback_size(uintptr_t ptr) const {
  if (!client_) return -1;
  return client_->cuMemObjGetMaxRequestCallbackSize(
      reinterpret_cast<void *>(ptr));
}

int CuObjectClient::close() noexcept {
  client_.reset();
  return CU_OBJ_SUCCESS;
}

}  // namespace cuobject
}  // namespace lmcache

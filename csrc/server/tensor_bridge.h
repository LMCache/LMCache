// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — ATen ↔ raw CUDA bridge
//
// Lightweight helpers so existing CUDA kernels (which take torch::Tensor)
// work with our raw pointers.  This is the ONLY header that includes
// <torch/torch.h>, keeping ATen out of the rest of the codebase.

#pragma once

#include <cstddef>
#include <cstdint>

// CUDA runtime types
#include <cuda_runtime_api.h>

// ATen forward declarations — full includes in tensor_bridge.cu
namespace at {
class Tensor;
}  // namespace at

// ScalarType must be included (not forward-declared) to avoid creating
// a distinct at::ScalarType from c10::ScalarType in PyTorch 2.10+.
#include <c10/core/ScalarType.h>

#include "types.h"

namespace lmcache {
namespace server {

// ============================================================================
// Tensor wrapping — raw pointer → non-owning ATen tensor
// ============================================================================

/// Wrap a raw device/host pointer as a non-owning ATen tensor.
/// The caller is responsible for lifetime: the tensor does NOT free the memory.
///
/// @param ptr        Raw pointer (device or host)
/// @param shape      Tensor dimensions
/// @param dtype      Our lightweight DType enum
/// @param device_idx CUDA device index (-1 for CPU)
/// @return           Non-owning ATen tensor
at::Tensor wrap_as_tensor(void* ptr, const std::vector<int64_t>& shape,
                          DType dtype, int device_idx = -1);

// ============================================================================
// CUDA IPC tensor operations
// ============================================================================

/// Open a cudaIpcMemHandle and create an ATen tensor view over it.
///
/// @param desc       CudaIpcTensorDesc with handle, shape, stride, dtype
/// @param device_idx Target CUDA device index
/// @return           ATen tensor backed by IPC-shared memory
at::Tensor open_ipc_tensor(const CudaIpcTensorDesc& desc, int device_idx);

/// Close a previously opened IPC memory handle.
void close_ipc_tensor(void* dev_ptr);

// ============================================================================
// CUDA IPC event helpers
// ============================================================================

/// Open a CUDA event from an IPC handle (bytes).
/// @param handle_bytes  Serialised cudaIpcEventHandle_t (64 bytes)
/// @return              CUDA event
cudaEvent_t open_ipc_event(const uint8_t* handle_bytes);

/// Create an interprocess CUDA event and return its IPC handle.
/// @param[out] event   Created CUDA event
/// @return             IPC handle bytes (64 bytes)
std::vector<uint8_t> create_ipc_event(cudaEvent_t& event);

// ============================================================================
// DType ↔ at::ScalarType conversion
// ============================================================================

/// Convert our DType enum to ATen ScalarType.
at::ScalarType dtype_to_scalar_type(DType dt);

/// Convert ATen ScalarType to our DType enum.
DType scalar_type_to_dtype(at::ScalarType st);

}  // namespace server
}  // namespace lmcache

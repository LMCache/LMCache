// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — ATen <-> raw CUDA bridge implementation

#include "tensor_bridge.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

// Use torch/all.h which correctly sets up the c10/at namespace aliasing
// in PyTorch 2.10+. Per-operator includes (ATen/ops/*.h) cause
// c10::ScalarType vs at::ScalarType mismatches.
#include <torch/all.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace lmcache {
namespace server {

// ============================================================================
// DType <-> at::ScalarType conversion
// ============================================================================

at::ScalarType dtype_to_scalar_type(DType dt) {
  switch (dt) {
    case DType::Float16:
      return at::ScalarType::Half;
    case DType::BFloat16:
      return at::ScalarType::BFloat16;
    case DType::Float32:
      return at::ScalarType::Float;
    case DType::Float8E4M3FN:
      return at::ScalarType::Float8_e4m3fn;
    case DType::Float8E5M2:
      return at::ScalarType::Float8_e5m2;
    case DType::Int8:
      return at::ScalarType::Byte;  // uint8 on wire (fp8 KV cache)
    case DType::Int32:
      return at::ScalarType::Int;
    case DType::Int64:
      return at::ScalarType::Long;
  }
  throw std::runtime_error("Unknown DType value: " +
                           std::to_string(static_cast<int>(dt)));
}

DType scalar_type_to_dtype(at::ScalarType st) {
  switch (st) {
    case at::ScalarType::Half:
      return DType::Float16;
    case at::ScalarType::BFloat16:
      return DType::BFloat16;
    case at::ScalarType::Float:
      return DType::Float32;
    case at::ScalarType::Float8_e4m3fn:
      return DType::Float8E4M3FN;
    case at::ScalarType::Float8_e5m2:
      return DType::Float8E5M2;
    case at::ScalarType::Char:
      return DType::Int8;
    case at::ScalarType::Byte:
      return DType::Int8;
    case at::ScalarType::Int:
      return DType::Int32;
    case at::ScalarType::Long:
      return DType::Int64;
    default:
      throw std::runtime_error(
          "Unsupported at::ScalarType for DType conversion: " +
          std::to_string(static_cast<int>(st)));
  }
}

// ============================================================================
// Tensor wrapping
// ============================================================================

at::Tensor wrap_as_tensor(void* ptr, const std::vector<int64_t>& shape,
                          DType dtype, int device_idx) {
  auto scalar_type = dtype_to_scalar_type(dtype);
  auto options = at::TensorOptions().dtype(scalar_type);

  if (device_idx >= 0) {
    options = options.device(at::Device(at::kCUDA, device_idx));
  } else {
    options = options.device(at::kCPU);
  }

  // Non-owning: at::from_blob does not take ownership by default.
  return at::from_blob(ptr, shape, options);
}

// ============================================================================
// CUDA IPC tensor operations
// ============================================================================

at::Tensor open_ipc_tensor(const CudaIpcTensorDesc& desc, int device_idx) {
  // Set the target device
  cudaError_t err = cudaSetDevice(device_idx);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                             cudaGetErrorString(err));
  }

  // Use libtorch's CUDACachingAllocator to open the IPC handle.
  // This handles PyTorch 2.10+ expanded handle format (66+ bytes)
  // as well as legacy 64-byte raw cudaIpcMemHandle_t.
  //
  // Python equivalent: torch.UntypedStorage._new_shared_cuda(device,
  // *handle[1:]) which internally calls getIpcDevPtr(handle_blob_as_string).
  std::shared_ptr<void> dev_ptr;
  try {
    std::string handle_str(
        reinterpret_cast<const char*>(desc.ipc_handle_blob.data()),
        desc.ipc_handle_blob.size());
    dev_ptr =
        c10::cuda::CUDACachingAllocator::getIpcDevPtr(std::move(handle_str));
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "[open_ipc_tensor] getIpcDevPtr FAILED on device %d: %s\n"
                 "  ipc_handle_blob size: %zu\n"
                 "  device_uuid: %s\n"
                 "  shape: [",
                 device_idx, e.what(), desc.ipc_handle_blob.size(),
                 desc.device_uuid.c_str());
    for (size_t i = 0; i < desc.shape.size(); ++i) {
      if (i > 0) std::fprintf(stderr, ",");
      std::fprintf(stderr, "%ld", desc.shape[i]);
    }
    std::fprintf(stderr, "]\n");
    throw;
  }

  void* base_ptr = dev_ptr.get();

  // Apply storage_offset
  auto scalar_type = dtype_to_scalar_type(desc.dtype);
  size_t elem_size = dtype_size(desc.dtype);
  void* offset_ptr =
      static_cast<char*>(base_ptr) + desc.storage_offset * elem_size;

  auto options = at::TensorOptions()
                     .dtype(scalar_type)
                     .device(at::Device(at::kCUDA, device_idx));

  // Create tensor backed by IPC storage.
  // We need to ensure the shared_ptr stays alive as long as the tensor exists.
  // Use from_blob with a custom deleter that holds the shared_ptr.
  auto ref = std::make_shared<std::shared_ptr<void>>(std::move(dev_ptr));
  auto deleter = [ref](void*) { /* ref drops when tensor dies */ };

  at::Tensor tensor;
  if (!desc.stride.empty()) {
    tensor =
        at::from_blob(offset_ptr, desc.shape, desc.stride, deleter, options);
  } else {
    tensor = at::from_blob(offset_ptr, desc.shape, deleter, options);
  }

  return tensor;
}

void close_ipc_tensor(void* dev_ptr) {
  // With getIpcDevPtr, cleanup is handled by the shared_ptr returned
  // from getIpcDevPtr (captured in the tensor's deleter).
  // This function is kept for API compatibility but is now a no-op.
  (void)dev_ptr;
}

// ============================================================================
// CUDA IPC event helpers
// ============================================================================

cudaEvent_t open_ipc_event(const uint8_t* handle_bytes) {
  cudaIpcEventHandle_t handle;
  static_assert(sizeof(handle) == 64, "cudaIpcEventHandle_t must be 64 bytes");
  std::memcpy(&handle, handle_bytes, sizeof(handle));

  cudaEvent_t event;
  cudaError_t err = cudaIpcOpenEventHandle(&event, handle);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaIpcOpenEventHandle failed: ") +
                             cudaGetErrorString(err));
  }
  return event;
}

std::vector<uint8_t> create_ipc_event(cudaEvent_t& event) {
  cudaError_t err = cudaEventCreateWithFlags(
      &event, cudaEventInterprocess | cudaEventDisableTiming);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaEventCreate failed: ") +
                             cudaGetErrorString(err));
  }

  cudaIpcEventHandle_t handle;
  err = cudaIpcGetEventHandle(&handle, event);
  if (err != cudaSuccess) {
    cudaEventDestroy(event);
    throw std::runtime_error(std::string("cudaIpcGetEventHandle failed: ") +
                             cudaGetErrorString(err));
  }

  std::vector<uint8_t> bytes(sizeof(handle));
  std::memcpy(bytes.data(), &handle, sizeof(handle));
  return bytes;
}

}  // namespace server
}  // namespace lmcache

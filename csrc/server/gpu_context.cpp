// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — GPUContext implementation
//
// Mirrors Python GPUCacheContext: opens IPC handles to vLLM's KV cache,
// discovers GPU KV format, creates CUDA streams, and manages transfer buffers.

#include "gpu_context.h"
#include "tensor_bridge.h"

#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

// Use torch/all.h which correctly sets up the c10/at namespace aliasing
// in PyTorch 2.10+. Per-operator includes (ATen/ops/*.h) cause
// c10::ScalarType vs at::ScalarType mismatches.
#include <torch/all.h>

// Use GPUKVFormat from mem_kernels.cuh (included via gpu_context.h →
// mp_mem_kernels.cuh)
namespace {

/// Discover GPU KV format from per-layer tensor shapes (vLLM only).
GPUKVFormat discover_format(
    const std::vector<lmcache::server::CudaIpcTensorDesc>& descs) {
  if (descs.empty()) {
    throw std::runtime_error("discover_format: empty kv_cache descriptors");
  }

  const auto& first = descs[0];
  int tensor_dim = static_cast<int>(first.shape.size());

  if (tensor_dim == 5) {
    // 5-D: either [2, NB, BS, NH, HS] (flash attn) or [NB, 2, BS, NH, HS]
    // (flash infer)
    if (first.shape[0] == 2) {
      return GPUKVFormat::NL_X_TWO_NB_BS_NH_HS;
    } else if (first.shape[1] == 2) {
      return GPUKVFormat::NL_X_NB_TWO_BS_NH_HS;
    }
    throw std::runtime_error(
        "discover_format: 5-D tensor but neither dim[0]==2 nor dim[1]==2");
  } else if (tensor_dim == 3) {
    // 3-D: [NB, BS, HS] → MLA
    return GPUKVFormat::NL_X_NB_BS_HS;
  }

  throw std::runtime_error(
      "discover_format: unsupported tensor_dim=" + std::to_string(tensor_dim) +
      " for vLLM (expected 3 or 5)");
}

bool format_is_mla(GPUKVFormat fmt) {
  return fmt == GPUKVFormat::NL_X_NB_BS_HS ||
         fmt == GPUKVFormat::NL_X_NBBS_ONE_HS;
}

}  // anonymous namespace

namespace lmcache {
namespace server {

// ============================================================================
// Constructor
// ============================================================================

GPUContext::GPUContext(const std::vector<CudaIpcTensorDesc>& kv_cache_descs,
                       int chunk_size) {
  if (kv_cache_descs.empty()) {
    throw std::runtime_error("GPUContext: empty kv_cache_descs");
  }

  // ---- Determine device index ----
  // We need to figure out which GPU these tensors belong to.
  // Use the first descriptor's device_uuid to find the device index.
  // For now, we discover by iterating CUDA devices and matching UUID.
  {
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    device_index_ = -1;
    for (int i = 0; i < num_devices; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      // Build UUID string like Python:
      // "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
      char uuid_str[80];
      const auto& uuid = prop.uuid;
      std::snprintf(
          uuid_str, sizeof(uuid_str),
          "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%"
          "02x%02x",
          (unsigned char)uuid.bytes[0], (unsigned char)uuid.bytes[1],
          (unsigned char)uuid.bytes[2], (unsigned char)uuid.bytes[3],
          (unsigned char)uuid.bytes[4], (unsigned char)uuid.bytes[5],
          (unsigned char)uuid.bytes[6], (unsigned char)uuid.bytes[7],
          (unsigned char)uuid.bytes[8], (unsigned char)uuid.bytes[9],
          (unsigned char)uuid.bytes[10], (unsigned char)uuid.bytes[11],
          (unsigned char)uuid.bytes[12], (unsigned char)uuid.bytes[13],
          (unsigned char)uuid.bytes[14], (unsigned char)uuid.bytes[15]);
      if (kv_cache_descs[0].device_uuid == uuid_str) {
        device_index_ = i;
        break;
      }
    }
    if (device_index_ < 0) {
      // Fallback: use device 0 with a warning
      std::fprintf(stderr,
                   "WARNING: GPUContext could not match device UUID '%s', "
                   "falling back to device 0\n",
                   kv_cache_descs[0].device_uuid.c_str());
      device_index_ = 0;
    }
  }

  cudaError_t err = cudaSetDevice(device_index_);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("GPUContext: cudaSetDevice failed: ") +
                             cudaGetErrorString(err));
  }

  // ---- Open IPC tensors and extract raw pointers ----
  num_layers_ = static_cast<int>(kv_cache_descs.size());
  kv_cache_ptrs_.resize(num_layers_);

  // Keep IPC tensors alive for the lifetime of this GPUContext.
  // getIpcDevPtr() caches via weak_ptr — if we don't hold the shared_ptr
  // (via the tensor's storage), the IPC handle is closed and kv_cache_ptrs_
  // become dangling pointers.
  kv_cache_ipc_tensors_.reserve(num_layers_);

  for (int i = 0; i < num_layers_; ++i) {
    at::Tensor t = open_ipc_tensor(kv_cache_descs[i], device_index_);
    kv_cache_ptrs_[i] = t.data_ptr();
    kv_cache_ipc_tensors_.push_back(std::move(t));
  }

  // ---- Discover GPU KV format ----
  auto fmt = discover_format(kv_cache_descs);
  gpu_kv_format_ = static_cast<int>(fmt);
  is_mla_ = format_is_mla(fmt);

  // ---- Extract shape parameters from the first tensor ----
  const auto& shape0 = kv_cache_descs[0].shape;

  switch (fmt) {
    case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
      // shape: [2, NB, BS, NH, HS]
      num_blocks_ = static_cast<int>(shape0[1]);
      block_size_ = static_cast<int>(shape0[2]);
      num_heads_ = static_cast<int>(shape0[3]);
      head_size_ = static_cast<int>(shape0[4]);
      hidden_dim_size_ = num_heads_ * head_size_;  // NH * HS
      break;

    case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
      // shape: [NB, 2, BS, NH, HS]
      num_blocks_ = static_cast<int>(shape0[0]);
      block_size_ = static_cast<int>(shape0[2]);
      num_heads_ = static_cast<int>(shape0[3]);
      head_size_ = static_cast<int>(shape0[4]);
      hidden_dim_size_ = num_heads_ * head_size_;  // NH * HS
      break;

    case GPUKVFormat::NL_X_NB_BS_HS:
      // shape: [NB, BS, HS] — MLA: num_heads not applicable
      num_blocks_ = static_cast<int>(shape0[0]);
      block_size_ = static_cast<int>(shape0[1]);
      head_size_ = static_cast<int>(shape0[2]);
      num_heads_ = 1;                 // MLA: single fused head
      hidden_dim_size_ = head_size_;  // HS
      break;

    default:
      throw std::runtime_error("GPUContext: unhandled GPU KV format " +
                               std::to_string(gpu_kv_format_));
  }

  // DType from first descriptor
  dtype_ = kv_cache_descs[0].dtype;

  // Build PageBufferShapeDesc for the block-level kernel
  shape_desc_.kv_size = is_mla_ ? 1 : 2;
  shape_desc_.nl = num_layers_;
  shape_desc_.nb = num_blocks_;
  shape_desc_.bs = block_size_;
  shape_desc_.nh = num_heads_;
  shape_desc_.hs = head_size_;
  shape_desc_.element_size = static_cast<int>(dtype_size(dtype_));

  std::fprintf(stderr,
               "GPUContext: device=%d, format=%d, mla=%d, layers=%d, "
               "blocks=%d, block_size=%d, hidden_dim=%d, num_heads=%d, "
               "head_size=%d, element_size=%d\n",
               device_index_, gpu_kv_format_, is_mla_ ? 1 : 0, num_layers_,
               num_blocks_, block_size_, hidden_dim_size_, num_heads_,
               head_size_, shape_desc_.element_size);

  // ---- Upload KV cache pointers to GPU (int64 tensor) ----
  {
    size_t ptr_bytes = num_layers_ * sizeof(int64_t);
    err = cudaMalloc(&kv_pointers_gpu_, ptr_bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaMalloc kv_pointers failed: ") +
          cudaGetErrorString(err));
    }

    // Reinterpret void* pointers as int64 for upload
    std::vector<int64_t> ptrs_as_int64(num_layers_);
    for (int i = 0; i < num_layers_; ++i) {
      ptrs_as_int64[i] = reinterpret_cast<int64_t>(kv_cache_ptrs_[i]);
    }
    err = cudaMemcpy(kv_pointers_gpu_, ptrs_as_int64.data(), ptr_bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaMemcpy kv_pointers failed: ") +
          cudaGetErrorString(err));
    }
  }

  // ---- Pre-compute slot mapping on GPU: shape [num_blocks, block_size] ----
  // slot_mapping[b][s] = b * block_size + s
  {
    size_t num_slots = static_cast<size_t>(num_blocks_) * block_size_;
    size_t slot_bytes = num_slots * sizeof(int64_t);
    err = cudaMalloc(&slot_mapping_gpu_, slot_bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaMalloc slot_mapping failed: ") +
          cudaGetErrorString(err));
    }

    // Build on host then upload
    std::vector<int64_t> slot_mapping_host(num_slots);
    for (int b = 0; b < num_blocks_; ++b) {
      for (int s = 0; s < block_size_; ++s) {
        slot_mapping_host[b * block_size_ + s] =
            static_cast<int64_t>(b) * block_size_ + s;
      }
    }
    err = cudaMemcpy(slot_mapping_gpu_, slot_mapping_host.data(), slot_bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaMemcpy slot_mapping failed: ") +
          cudaGetErrorString(err));
    }
  }

  // ---- Allocate temporary GPU buffer (sized for kMaxBatchSize chunks) ----
  {
    auto buf_shape = get_kv_buffer_shape(chunk_size * kMaxBatchSize);
    size_t num_elems = 1;
    for (auto d : buf_shape) num_elems *= static_cast<size_t>(d);
    tmp_gpu_buffer_bytes_ = num_elems * dtype_size(dtype_);

    err = cudaMalloc(&tmp_gpu_buffer_, tmp_gpu_buffer_bytes_);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaMalloc tmp_buffer failed: ") +
          cudaGetErrorString(err));
    }
  }

  // ---- Allocate pre-allocated GPU buffer for block IDs ----
  {
    size_t block_ids_bytes = kMaxBlockIds * sizeof(int64_t);
    err = cudaMalloc(&block_ids_buffer_, block_ids_bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaMalloc block_ids_buffer failed: ") +
          cudaGetErrorString(err));
    }
  }

  // ---- Create CUDA streams ----
  {
    // Normal-priority stream
    err = cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, 0);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaStreamCreate (normal) failed: ") +
          cudaGetErrorString(err));
    }

    // High-priority stream — use the highest priority available
    int least_priority = 0, greatest_priority = 0;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    err = cudaStreamCreateWithPriority(
        &high_priority_stream_, cudaStreamNonBlocking, greatest_priority);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("GPUContext: cudaStreamCreate (high) failed: ") +
          cudaGetErrorString(err));
    }
  }

  std::fprintf(stderr, "GPUContext: initialized on device %d\n", device_index_);
}

// ============================================================================
// Destructor
// ============================================================================

GPUContext::~GPUContext() {
  // Best-effort cleanup — don't throw from destructors
  cudaSetDevice(device_index_);

  if (stream_) cudaStreamDestroy(stream_);
  if (high_priority_stream_) cudaStreamDestroy(high_priority_stream_);

  if (kv_pointers_gpu_) cudaFree(kv_pointers_gpu_);
  if (slot_mapping_gpu_) cudaFree(slot_mapping_gpu_);
  if (tmp_gpu_buffer_) cudaFree(tmp_gpu_buffer_);
  if (block_ids_buffer_) cudaFree(block_ids_buffer_);

  // Close IPC handles — the base pointer is needed (before storage_offset),
  // but open_ipc_tensor may have adjusted it. We stored data_ptr() which
  // includes the offset, so we need the original base.
  // Actually, cudaIpcCloseMemHandle requires the exact pointer returned by
  // cudaIpcOpenMemHandle. Since we stored data_ptr (which includes offset),
  // and the offset might be non-zero, we can't close cleanly here.
  //
  // For correctness, IPC handles should be closed with the base pointer.
  // In practice, the server process lifetime matches the IPC handle lifetime,
  // so the OS cleans up on exit. We skip explicit close to avoid errors.
  //
  // If precise cleanup is needed, we'd need to store the base pointers
  // separately from the offset pointers.
}

// ============================================================================
// Tensor accessors
// ============================================================================

at::Tensor GPUContext::kv_pointers() const {
  return wrap_as_tensor(kv_pointers_gpu_, {num_layers_}, DType::Int64,
                        device_index_);
}

at::Tensor GPUContext::get_slot_mapping_tensor(
    const std::vector<int32_t>& gpu_block_ids) const {
  // The precomputed slot_mapping_gpu_ has shape [num_blocks, block_size].
  // We need to index into it with the given block IDs and flatten.
  //
  // Equivalent Python:
  //   gpu_block_ids_tensor = torch.tensor(gpu_block_ids, device=...,
  //   dtype=torch.long) return
  //   self.slot_mapping_tensor_[gpu_block_ids_tensor].flatten().contiguous()

  int num_blocks_requested = static_cast<int>(gpu_block_ids.size());

  // Convert int32 block IDs to int64
  std::vector<int64_t> block_ids_i64(gpu_block_ids.begin(),
                                     gpu_block_ids.end());

  // Allocate GPU buffer for block IDs and copy synchronously
  void* idx_gpu = nullptr;
  size_t idx_bytes = num_blocks_requested * sizeof(int64_t);
  cudaError_t err = cudaMalloc(&idx_gpu, idx_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("get_slot_mapping_tensor: cudaMalloc failed: ") +
        cudaGetErrorString(err));
  }
  err = cudaMemcpy(idx_gpu, block_ids_i64.data(), idx_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(idx_gpu);
    throw std::runtime_error(
        std::string("get_slot_mapping_tensor: cudaMemcpy failed: ") +
        cudaGetErrorString(err));
  }

  // Wrap as ATen tensors
  at::Tensor slot_mapping =
      wrap_as_tensor(slot_mapping_gpu_, {num_blocks_, block_size_},
                     DType::Int64, device_index_);

  // Wrap GPU index buffer as non-owning tensor, then free after use
  at::Tensor idx = wrap_as_tensor(idx_gpu, {num_blocks_requested}, DType::Int64,
                                  device_index_);

  // Index and flatten (all on GPU, same stream via CUDAStreamGuard)
  at::Tensor result = slot_mapping.index_select(0, idx).flatten().contiguous();

  // Free the temporary index buffer (safe: index_select copies the data)
  cudaFree(idx_gpu);

  return result;
}

at::Tensor GPUContext::get_tmp_gpu_buffer(int num_tokens) const {
  // Full shape: [1or2, num_layers, chunk_size, hidden_dim_size]
  // Slice to:   [1or2, num_layers, num_tokens, hidden_dim_size]
  auto full_shape = get_kv_buffer_shape(num_tokens);

  // Compute number of elements for the requested slice
  size_t num_elems = 1;
  for (auto d : full_shape) num_elems *= static_cast<size_t>(d);
  size_t needed_bytes = num_elems * dtype_size(dtype_);

  if (needed_bytes > tmp_gpu_buffer_bytes_) {
    throw std::runtime_error("GPUContext::get_tmp_gpu_buffer: requested " +
                             std::to_string(needed_bytes) +
                             " bytes but buffer has " +
                             std::to_string(tmp_gpu_buffer_bytes_));
  }

  return wrap_as_tensor(tmp_gpu_buffer_, full_shape, dtype_, device_index_);
}

std::vector<int64_t> GPUContext::get_kv_buffer_shape(int num_tokens) const {
  if (is_mla_) {
    return {1, num_layers_, num_tokens, hidden_dim_size_};
  } else {
    return {2, num_layers_, num_tokens, hidden_dim_size_};
  }
}

std::vector<at::Tensor> GPUContext::get_tmp_gpu_buffer_batched(
    int num_tokens, int batch_size) const {
  if (batch_size > kMaxBatchSize) {
    throw std::runtime_error(
        "GPUContext::get_tmp_gpu_buffer_batched: batch_size " +
        std::to_string(batch_size) + " exceeds max " +
        std::to_string(kMaxBatchSize));
  }

  auto single_shape = get_kv_buffer_shape(num_tokens);
  size_t single_elems = 1;
  for (auto d : single_shape) single_elems *= static_cast<size_t>(d);
  size_t elem_bytes = dtype_size(dtype_);

  std::vector<at::Tensor> result;
  result.reserve(batch_size);
  size_t offset = 0;
  for (int i = 0; i < batch_size; ++i) {
    void* ptr = static_cast<uint8_t*>(tmp_gpu_buffer_) + offset;
    result.push_back(wrap_as_tensor(ptr, single_shape, dtype_, device_index_));
    offset += single_elems * elem_bytes;
  }

  if (offset > tmp_gpu_buffer_bytes_) {
    throw std::runtime_error("GPUContext::get_tmp_gpu_buffer_batched: total " +
                             std::to_string(offset) + " bytes exceeds buffer " +
                             std::to_string(tmp_gpu_buffer_bytes_));
  }

  return result;
}

at::Tensor GPUContext::stage_block_ids(
    const std::vector<int32_t>& block_ids) const {
  int n = static_cast<int>(block_ids.size());
  if (n > kMaxBlockIds) {
    throw std::runtime_error(
        "GPUContext::stage_block_ids: " + std::to_string(n) +
        " block_ids exceeds max " + std::to_string(kMaxBlockIds));
  }

  // Convert int32 → int64
  std::vector<int64_t> ids_i64(block_ids.begin(), block_ids.end());

  // Copy to pre-allocated GPU buffer
  cudaMemcpyAsync(block_ids_buffer_, ids_i64.data(), n * sizeof(int64_t),
                  cudaMemcpyHostToDevice, stream_);

  return wrap_as_tensor(block_ids_buffer_, {n}, DType::Int64, device_index_);
}

// ============================================================================
// CUDA stream helpers
// ============================================================================

void GPUContext::launch_host_func(cudaHostFn_t fn, void* user_data) {
  cudaError_t err = cudaLaunchHostFunc(stream_, fn, user_data);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("GPUContext::launch_host_func failed: ") +
        cudaGetErrorString(err));
  }
}

}  // namespace server
}  // namespace lmcache

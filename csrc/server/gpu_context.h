// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — Per-GPU state management
//
// Mirrors Python GPUCacheContext: holds raw CUDA pointers to vLLM's
// KV cache tensors, CUDA streams, slot mapping, and transfer buffers.
// Uses the raw CUDA runtime API (no cupy).

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <cuda_runtime_api.h>

#include <torch/all.h>

#include "types.h"
#include "mp_mem_kernels.cuh"  // PageBufferShapeDesc

namespace lmcache {
namespace server {

class GPUContext {
 public:
  /// Construct from IPC tensor descriptors + chunk_size.
  /// Opens IPC handles, discovers KV format, creates streams.
  ///
  /// @param kv_cache_descs   Per-layer CudaIpcTensorDesc from REGISTER_KV_CACHE
  /// @param chunk_size       LMCache chunk size (e.g. 256)
  GPUContext(const std::vector<CudaIpcTensorDesc>& kv_cache_descs,
             int chunk_size);

  ~GPUContext();

  // Non-copyable, non-movable (owns CUDA resources)
  GPUContext(const GPUContext&) = delete;
  GPUContext& operator=(const GPUContext&) = delete;

  // ---- Properties ----

  int device_index() const { return device_index_; }
  DType dtype() const { return dtype_; }
  int num_layers() const { return num_layers_; }
  int num_blocks() const { return num_blocks_; }
  int block_size() const { return block_size_; }
  int hidden_dim_size() const { return hidden_dim_size_; }
  int num_heads() const { return num_heads_; }
  int head_size() const { return head_size_; }
  bool is_mla() const { return is_mla_; }
  int gpu_kv_format() const { return gpu_kv_format_; }

  /// Pre-built PageBufferShapeDesc for the block-level kernel
  PageBufferShapeDesc shape_desc() const { return shape_desc_; }

  // ---- Tensor accessors (ATen wrappers over raw pointers) ----

  /// GPU tensor of int64 KV cache base pointers, shape [num_layers]
  at::Tensor kv_pointers() const;

  /// Compute slot mapping tensor for the given block IDs.
  /// Returns a flattened int64 tensor on the GPU.
  at::Tensor get_slot_mapping_tensor(
      const std::vector<int32_t>& gpu_block_ids) const;

  /// Temporary GPU buffer for D2H/H2D transfers, shape [1or2, L, T, D]
  at::Tensor get_tmp_gpu_buffer(int num_tokens) const;

  /// Shape of the KV buffer for the given number of tokens.
  /// MLA:     [1, num_layers, num_tokens, hidden_dim_size]
  /// Non-MLA: [2, num_layers, num_tokens, hidden_dim_size]
  std::vector<int64_t> get_kv_buffer_shape(int num_tokens) const;

  /// Temporary GPU buffers for batched transfers (batch_size views into
  /// the pre-allocated buffer).
  std::vector<at::Tensor> get_tmp_gpu_buffer_batched(int num_tokens,
                                                     int batch_size) const;

  /// Copy block_ids to pre-allocated GPU buffer and return a view.
  at::Tensor stage_block_ids(const std::vector<int32_t>& block_ids) const;

  // ---- CUDA streams ----

  /// Normal-priority stream for store operations
  cudaStream_t stream() const { return stream_; }

  /// High-priority stream for retrieve operations
  cudaStream_t high_priority_stream() const { return high_priority_stream_; }

  /// Schedule a host callback on the normal-priority stream.
  /// Replaces cupy's launch_host_func with cudaLaunchHostFunc.
  void launch_host_func(cudaHostFn_t fn, void* user_data);

  // ---- Serialisation ----

  /// Per-device lock to serialise GPU↔CPU data transfers.
  /// Prevents GIL ↔ CUDA driver lock order inversion.
  std::mutex& transfer_lock() { return transfer_lock_; }

 private:
  int device_index_;
  DType dtype_;
  int num_layers_;
  int num_blocks_;
  int block_size_;
  int hidden_dim_size_;
  int num_heads_;
  int head_size_;
  bool is_mla_;
  int gpu_kv_format_;  // Cast of GPUKVFormat from mem_kernels.cuh
  PageBufferShapeDesc shape_desc_;

  static constexpr int kMaxBatchSize = 4;
  static constexpr int kMaxBlockIds = 1000000;

  // Raw KV cache device pointers (opened via IPC)
  std::vector<void*> kv_cache_ptrs_;

  // ATen tensors wrapping IPC handles — MUST stay alive to keep IPC memory
  // valid. getIpcDevPtr() returns shared_ptr<void> cached via weak_ptr;
  // destroying these tensors releases the last shared_ptr, closing the IPC
  // handle and invalidating kv_cache_ptrs_.
  std::vector<at::Tensor> kv_cache_ipc_tensors_;

  // GPU tensor of kv_cache_ptrs_ as int64, shape [num_layers]
  void* kv_pointers_gpu_;  // device memory

  // Pre-computed slot mapping, shape [num_blocks, block_size]
  void* slot_mapping_gpu_;  // device memory

  // Temporary GPU buffer for transfers
  void* tmp_gpu_buffer_;  // device memory
  size_t tmp_gpu_buffer_bytes_;

  // Pre-allocated GPU buffer for block IDs (up to kMaxBlockIds int64 elements)
  void* block_ids_buffer_;  // device memory

  // CUDA streams
  cudaStream_t stream_;
  cudaStream_t high_priority_stream_;

  // Per-device transfer lock
  std::mutex transfer_lock_;
};

}  // namespace server
}  // namespace lmcache

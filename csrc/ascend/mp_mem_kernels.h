// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine_kv_format.h"
#include "kv_transfer_types.h"
#include <cstdint>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

// Compile-time shape descriptor, field-for-field aligned with upstream
// PageBufferShapeDesc (LMCache csrc/mp_mem_kernels.cuh:11). block_stride_elems
// honours engine-side dim-0 padding when > 0, else falls back to the tight
// bs * nh * hs stride.
struct PageBufferShapeDesc {
  int kv_size;
  int nl;
  int nb;
  int bs;
  int nh;
  int hs;
  int element_size;
  int block_stride_elems;
  // Ascend-only plane count. Formats 16/17 set it at registration.
  // prepare_group sets it to 2 for format 2. Unset 16/17 stays 0 and is
  // rejected at group prepare. For format 13, prepare_group derives 1 when
  // unset.
  int32_t num_planes = 0;
  // Per-plane payload bytes per token slot (64-bit byte addressing). The
  // engine-side token step is derived from the payload (dense token rows).
  int64_t plane_slot_bytes[4] = {0, 0, 0, 0};
  // Per-plane per-block dim-0 step in bytes (stride(0) * itemsize).
  int64_t plane_block_stride_bytes[4] = {0, 0, 0, 0};
  // Format 2 only: bytes from each layer's base data_ptr() to plane p.
  // Zero for formats whose pointer table already has one entry per plane.
  int64_t plane_engine_base_offset_bytes[4] = {0, 0, 0, 0};

  template <typename ScalarType>
  inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

// Object-group transfer plan types (upstream mp_mem_kernels.cuh:80-116).
struct StagingCopy {
  uintptr_t dest;
  uintptr_t src;
  size_t nbytes;
  size_t host_offset;
};

struct LaunchVar {
  int group_idx;
  int64_t block_ids_offset;
  int total_blocks;
  int num_objects;
  int skip_prefix_n_blocks;
};

struct BatchStep {
  std::vector<StagingCopy> staging;
  std::vector<LaunchVar> launches;
};

struct KernelGroupSpec {
  uintptr_t paged_buffer_ptrs;
  std::vector<int64_t> lmcache_objects_ptrs;
  PageBufferShapeDesc shape_desc;
  int lmcache_chunk_size;
  EngineKVFormat engine_kv_format;
  uintptr_t block_ids_base;
  int64_t block_ids_capacity;
};

// Plan executor: enqueues every staging copy and kernel launch described by
// batch_steps within a single GIL release (configured at the pybind layer).
// Mirrors upstream execute_object_group_transfer (mp_mem_kernels.cuh:133).
void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps);

// Block-level multi-layer KV transfer between vLLM paged buffers and LMCache
// contiguous memory objects. Mirrors upstream multi_layer_block_kv_transfer
// (mp_mem_kernels.cuh:155). Phase 1 loops over the object batch inside this
// entry point (one kernel launch per object).
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks);

// Async host<->device copy chunked at the host buffer's alignment boundary
// (port of upstream lmcache_memcpy_async, mem_kernels.cu). Used by the plan
// executor's staging step and by the Python fallback path.
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

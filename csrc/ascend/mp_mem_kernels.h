// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kv_transfer_plan_types.h"
#include <cstdint>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

// PageBufferShapeDesc and KernelGroupSpec are the shared types from
// kv_transfer_plan_types.h. lmcache_native registers both with pybind11 and
// the server builds its plans through lmcache.v1.common.ops_types, so this
// module must consume the very same C++ types instead of registering copies.
// The registration is only visible to extensions that share pybind11
// internals with lmcache_native; see CMakeLists.txt for how that is ensured.

// Staged object-group plan descriptors, with the same layout as the CUDA
// backend's csrc/cuda/transfer_plan_types.cuh.

// One asynchronous host<->device copy. host_offset is the host-side virtual
// offset in the lmcache allocator (source for H2D, destination for D2H).
struct StagingCopy {
  uintptr_t dest;
  uintptr_t src;
  size_t nbytes;
  size_t host_offset;
};

// One kernel launch within a batch step. The batch-invariant arguments live
// in the referenced KernelGroupSpec; only these vary per (batch, group).
struct LaunchVar {
  int group_idx;             // index into the plan's kernel_group_specs
  int64_t block_ids_offset;  // element offset into the group's block_ids_base
  int total_blocks;          // number of block ids for this launch
  int num_objects;           // chunks in this batch (1-4)
  int skip_prefix_n_blocks;
};

// One batch: its staging copies and kernel launches. For H2D the staging runs
// before the launches, for D2H after; the executor preserves this ordering.
struct BatchStep {
  std::vector<StagingCopy> staging;
  std::vector<LaunchVar> launches;
};

// Plan executor: enqueues every staging copy and kernel launch described by
// batch_steps within a single GIL release (configured at the pybind layer).
void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps);

// Block-level multi-layer KV transfer between vLLM paged buffers and LMCache
// contiguous memory objects. Loops over the object batch inside this entry
// point (one kernel launch per object).
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<uintptr_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks);

// Async host<->device copy chunked at the host buffer's alignment boundary.
// Used by the plan executor's staging step and by the Python fallback path.
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

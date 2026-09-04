// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/cuda/CUDAGuard.h>

#include "kv_transfer_plan_types.h"
#include "mem_kernels.cuh"
#include "transfer_plan_types.cuh"

template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;  // 0 - 4
};

// ---------------------------------------------------------------------------
// Object-group transfer plan.
//
// A whole object group's transfer (all staging copies + all kernel launches) is
// described as a plan on the Python side, then executed in a single native call
// (execute_object_group_transfer) that releases the GIL once for the entire
// burst instead of once per copy/launch. See the design in
// docs/design/v1/multiprocess/modules/ and lmcache_driven_transfer.py.
// ---------------------------------------------------------------------------

/**
 * Execute one object group's transfer plan on the current CUDA stream.
 *
 * Enqueues every staging copy and kernel launch described by `batch_steps`
 * within a single GIL release (configured at the pybind layer), eliminating the
 * per-copy/per-launch GIL handoffs of the equivalent Python loop. The device
 * guard and stream are set once for the whole plan.
 *
 * @param direction            H2D (retrieve) or D2H (store), applied to all ops
 * @param device               CUDA device of the transfer
 * @param host_buffer_alignment Host buffer alignment for staging copies
 *                              (power of two)
 * @param kernel_group_specs   Per-kernel-group invariants
 * @param batch_steps          Ordered per-batch staging + launch work
 */
void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps);

/**
 * Block-level multi-layer KV transfer between vLLM paged buffers and
 * LMCache contiguous memory objects.
 *
 * @param paged_buffer_ptrs_tensor  GPU int64 tensor of data pointers into
 *                                  vLLM paged buffers (one per tensor)
 * @param lmcache_objects_ptrs      Raw pointers to LMCache memory objects
 * @param block_ids                 GPU int64 tensor of block indices in vLLM
 *                                  paged buffer
 * @param device                    CUDA device of vLLM tensors
 * @param direction                 H2D (LMCache->vLLM) or D2H (vLLM->LMCache)
 * @param shape_desc                Shape descriptor for the paged buffer
 * @param lmcache_chunk_size        Tokens per LMCache memory object
 * @param engine_kv_format             EngineKVFormat identifier
 * @param skip_prefix_n_blocks      Number of blocks to skip at the beginning
 */
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks);

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

// ---------------------------------------------------------------------------
// Direct copy-engine transfer (cudaMemcpyBatchAsync).
//
// Alternative to the staged plan above for layouts whose paged block is one
// contiguous run identical to LMCache's [bs, nh*hs] rows: every (kv, layer,
// block) of an object becomes one batch entry between the pinned host object
// and the paged buffer, so no staging buffer and no SM kernel are involved.
// One cudaMemcpyBatchAsync call is issued per object; the batch is stream
// ordered as a whole. Requires CUDA runtime and driver >= 12.8 (no HIP).
// ---------------------------------------------------------------------------

/**
 * Whether cudaMemcpyBatchAsync can be used in this process.
 *
 * True when the extension was compiled against CUDA >= 12.8 and both the
 * runtime and the driver report >= 12.8. Cached after the first call.
 */
bool batch_memcpy_supported();

/**
 * Whether the direct copy path can address blocks of `engine_kv_format`.
 *
 * True for token-major formats whose paged block is one contiguous
 * [bs, nh, hs] run (the affine formats of resolve_block_addressing). HND,
 * blocked-scale and per-layer (K, V)-tuple layouts return false.
 */
bool direct_copy_format_supported(EngineKVFormat engine_kv_format);

/**
 * Execute one object group's transfer through the copy engine.
 *
 * For each object, expands every (kv plane, layer, block >= skip) of every
 * group into a (host, device, tight block bytes) entry, splits entries at
 * `host_buffer_alignment` boundaries of the allocator's virtual offset (a
 * copy may not span two cudaHostRegister regions), and issues one
 * cudaMemcpyBatchAsync on the current stream of `device`.
 *
 * @param direction             H2D (retrieve) or D2H (store)
 * @param device                CUDA device of the paged buffers
 * @param host_buffer_alignment Pin-chunk granularity of the host allocator
 *                              (power of two)
 * @param group_specs           Per-kernel-group invariants
 * @param objects               Memory objects to copy, in stream order
 *
 * @throws c10::Error if batch_memcpy_supported() is false, a format is not
 *         eligible, a block id or host range is out of bounds, or the CUDA
 *         call fails.
 */
void execute_direct_copy_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<DirectCopyGroupSpec>& group_specs,
    const std::vector<DirectCopyObject>& objects);

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

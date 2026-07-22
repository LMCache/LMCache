// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mem_kernels.cuh"  // TransferDirection, EngineKVFormat

#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>

struct PageBufferShapeDesc {
  int kv_size;       // 1 or 2
  int nl;            // num layers
  int nb;            // num blocks
  int bs;            // block size
  int nh;            // num heads
  int hs;            // head size
  int element_size;  // bytes (1 or 2)
  // Physical per-block stride in source-dtype element units, used by
  // formats whose dim-0 is the block axis to step over padding bytes
  // (e.g. DeepSeek V4 compressor / indexer caches sharing a vLLM KV
  // pool with larger attn groups, whose rows are padded up to the
  // pool's max row width). 0 means "unset — fall back to the
  // format-specific tight stride".
  //
  // CONTRACT: pass ``tensor.stride(0)`` verbatim. PyTorch stride
  // semantics already absorb every inner-dim extent (including
  // ``kv_size``), so DO NOT pre-multiply by any inner dim.
  //
  // Honoured today only by NL_X_NB_BS_HS (per-layer [NB, BS, HS],
  // MLA). NL_X_NB_TWO_BS_NH_HS is restricted to the tight form
  // upstream and leaves this field at 0; all other formats either
  // pack non-block info into dim-0 or do not support dim-0 padding,
  // and ignore this field.
  int block_stride_elems;

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  // Per (K or V) block step along dim-0, expressed in ``ScalarType``
  // element units (the kernel's working dtype, e.g. uint4 / uint32_t /
  // uint16_t). Returns the tight ``bs * nh * hs`` by default, or the
  // physical ``block_stride_elems`` when dim-0 carries padding (today
  // only NL_X_NB_BS_HS, see ``block_stride_elems`` above). Every
  // ``calculate_engine_global_offset`` branch uses this as the dim-0
  // step, so honouring padding here propagates to all formats without
  // per-branch changes.
  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

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

// One asynchronous host<->device copy. `host_offset` is the host-side virtual
// offset in the lmcache allocator (source for H2D, destination for D2H).
struct StagingCopy {
  uintptr_t dest;
  uintptr_t src;
  size_t nbytes;
  size_t host_offset;
};

// One kernel launch within a batch step. The batch-invariant arguments live in
// the referenced KernelGroupSpec; only these vary per (batch, kernel group).
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

// Per-kernel-group invariants, resolved once on the Python side.
struct KernelGroupSpec {
  uintptr_t paged_buffer_ptrs;                // device ptr-array base address
  std::vector<int64_t> lmcache_objects_ptrs;  // temp GPU buffer ptr per slot
  PageBufferShapeDesc shape_desc;
  int lmcache_chunk_size;
  EngineKVFormat engine_kv_format;
  uintptr_t block_ids_base;  // device int64* base; sliced via block_ids_offset
  int64_t block_ids_capacity;  // total int64 elements behind block_ids_base;
                               // bounds-checks each slice in the executor
};

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
// CacheBlend retrieve plan: plan-then-execute like the object-group transfer
// above, plus K-only re-RoPE and a per-token scatter (CB matches are not
// block-aligned). One GIL release per request.
// ---------------------------------------------------------------------------

// Per-kernel-group invariants of a CB retrieve plan.
struct CBGroupSpec {
  uintptr_t paged_kv_ptrs;  // device ptr-array base (per-layer paged ptrs)
  std::vector<int64_t> temp_buffer_ptrs;  // temp GPU buffer base per tmp slot
  // Scatter geometry: the tmp slot buffer is [kv_size, num_layers,
  // slot_tokens, hidden_elems] of element_size-byte scalars.
  int num_layers;
  int slot_tokens;   // token capacity of one tmp slot (slots per chunk)
  int hidden_elems;  // scalars per token per layer per K/V plane
  int element_size;
  EngineKVFormat engine_kv_format;
  int page_buffer_size;
  int block_size;
  int head_size;                  // scatter kernel head_size (element units)
  uintptr_t slot_mapping_base;    // device int64*, whole-request slot mapping
  int64_t slot_mapping_capacity;  // int64 elements behind slot_mapping_base
  // Re-RoPE (cos_sin_cache == 0 disables rope for this group).
  uintptr_t cos_sin_cache;  // device ptr, [max_position, rot_dim] scalars
  int rot_dim;
  int rope_num_kv_heads;
  int64_t rope_head_size;
  int64_t rope_head_stride;  // == head_size, or 2*head_size for fused packed
  int key_scalar_type;       // at::ScalarType of the KV data
  bool is_neox;
};

// One K-only re-RoPE launch: rotate tmp slot `slot_idx` of group `group_idx`
// in place from stored position `old_st` to new position `cur_st`.
struct CBRopeVar {
  int group_idx;
  int slot_idx;
  int64_t old_st;
  int64_t cur_st;
};

// One per-token scatter launch: write `n_tok` tokens of tmp slot `slot_idx`
// to the paged KV slots at `slot_mapping_base + slot_mapping_offset`.
struct CBScatterVar {
  int group_idx;
  int slot_idx;
  int64_t slot_mapping_offset;
  int n_tok;
};

// One batch of tmp slots: H2D staging, then re-RoPE, then scatter. The order
// is load-bearing (slots are reused by the next step).
struct CBRetrieveStep {
  std::vector<StagingCopy> staging;
  std::vector<CBRopeVar> ropes;
  std::vector<CBScatterVar> scatters;
};

/**
 * Enqueue a whole CB retrieve plan (staging + rope + scatter per step) in one
 * GIL release; staging overlaps the previous step's kernels on a pool stream.
 */
void execute_cb_retrieve_plan(const torch::Device& device,
                              size_t host_buffer_alignment,
                              const std::vector<CBGroupSpec>& group_specs,
                              const std::vector<CBRetrieveStep>& steps);

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

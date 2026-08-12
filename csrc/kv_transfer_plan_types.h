// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kv_transfer_types.h"

// Shared native descriptors for blocked KV transfers and object-group transfer
// plans. These are backend-agnostic value types: they describe geometry,
// pointers, and per-launch metadata without depending on any vendor runtime.

// __host__ __device__ under CUDA/HIP so kernels can call the inline helpers;
// otherwise keep the header toolchain-agnostic for the common C++ extension.
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define LMC_TRANSFER_PLAN_HD __host__ __device__
#else
  #define LMC_TRANSFER_PLAN_HD
#endif

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
  LMC_TRANSFER_PLAN_HD inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  LMC_TRANSFER_PLAN_HD inline size_t scalars_per_token() const {
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
  LMC_TRANSFER_PLAN_HD inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

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

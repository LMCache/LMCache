// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kv_transfer_plan_types.h"  // PageBufferShapeDesc, EngineKVFormat

// CUDA-specific plan descriptors for staged object-group transfers. These stay
// local to the CUDA backend because different accelerators may need different
// batching / launch metadata layouts.

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

// ---------------------------------------------------------------------------
// Direct copy-engine plan (no GPU staging buffer, no SM kernel).
//
// Every (kv plane, layer, block) of one memory object is one entry of a
// cudaMemcpyBatchAsync call between the pinned host object and the paged
// buffer. Only token-major formats whose block is one contiguous
// [bs, nh, hs] run are eligible (see direct_copy_format_supported); HND and
// blocked-scale layouts stay on the kernel path.
// ---------------------------------------------------------------------------

// Per-kernel-group invariants for the direct copy path, resolved once per
// object group on the Python side.
struct DirectCopyGroupSpec {
  // Host copy of the group's device layer pointers, in the same order as the
  // kernel's paged_buffer_ptrs array (per layer; K layers then V layers for
  // the SGLang two-list formats; a single base for cross-layer formats).
  std::vector<uintptr_t> paged_layer_ptrs;
  PageBufferShapeDesc shape_desc;
  EngineKVFormat engine_kv_format;
  int slots_per_chunk;             // tokens per object for this group
  size_t byte_offset_in_object;    // start of this group's [kv, nl, slots,
                                   // nh*hs] region inside the memory object
  std::vector<int64_t> block_ids;  // host block ids after window downsample,
                                   // slots_per_chunk / bs entries per chunk
};

// One pinned host memory object (chunk) to scatter/gather.
struct DirectCopyObject {
  uintptr_t host_ptr;
  size_t host_offset;  // allocator virtual offset (pin-chunk boundary split)
  size_t nbytes;       // object size; every host range is bounds-checked
  int chunk_idx;       // index into each group's block_ids (x blocks/chunk)
  std::vector<int> skip_prefix_n_blocks;  // per group_specs index
};

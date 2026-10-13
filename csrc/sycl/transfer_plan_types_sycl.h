// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../kv_transfer_plan_types.h"  // KernelGroupSpec, PageBufferShapeDesc

// SYCL plan descriptors for staged object-group transfers, consumed by
// execute_object_group_transfer. Field-for-field the same as the CUDA ones in
// csrc/cuda/transfer_plan_types.cuh, since the Python planner builds both.

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
  int num_objects;           // chunks in this batch
  int skip_prefix_n_blocks;
};

// One batch: its staging copies and kernel launches. For H2D the staging runs
// before the launches, for D2H after; the executor preserves this ordering.
struct BatchStep {
  std::vector<StagingCopy> staging;
  std::vector<LaunchVar> launches;
};

// SPDX-License-Identifier: Apache-2.0

#include <sycl/sycl.hpp>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "mem_kernels_sycl.h"

namespace {

constexpr size_t kWorkGroupSize = 256;
constexpr size_t kObjectsPerLaunch = 4;
// Split large pages across work-groups rather than underutilizing the GPU
// when a request has only a handful of large blocks.
constexpr size_t kTileBytes = 16384;

struct BlockLayout {
  size_t block_stride;
  size_t layer_stride;
  size_t kv_stride;
  int pointer_layer_stride;
  int pointer_kv_stride;
  bool hnd;
  bool blocked_scale;
};

BlockLayout resolve_layout(PageBufferShapeDesc d, EngineKVFormat format) {
  const auto facts = format_facts(format);
  TORCH_CHECK(format != EngineKVFormat::NL_X_NP_X_NB_BS_ONE_HS,
              "Variable-width plane tuples require tensor-form transfer");
  const int expected_kv = (facts.is_mla || facts.is_fused_packed ||
                           format == EngineKVFormat::NL_X_NB_BS_NH_HS)
                              ? 1
                              : 2;
  TORCH_CHECK(d.kv_size == expected_kv, "kv_size does not match format");
  TORCH_CHECK(!facts.is_mla || d.nh == 1, "MLA requires nh == 1");
  const size_t block_bytes =
      static_cast<size_t>(d.bs) * d.nh * d.hs * d.element_size;
  BlockLayout layout{block_bytes,
                     0,
                     0,
                     1,
                     0,
                     facts.is_hnd,
                     format == EngineKVFormat::NL_X_NB_BSV_BSS};
  if (facts.is_cross_layer) {
    layout.block_stride = d.nl * d.kv_size * block_bytes;
    layout.layer_stride = d.kv_size * block_bytes;
    layout.kv_stride = block_bytes;
    layout.pointer_layer_stride = 0;
  } else if (facts.is_kv_list) {
    layout.pointer_kv_stride = d.nl;
  } else if (facts.is_kv_second_tuple) {
    layout.pointer_layer_stride = 2;
    layout.pointer_kv_stride = 1;
  } else if (facts.is_two_major) {
    layout.kv_stride = d.nb * block_bytes;
  } else if (d.kv_size == 2) {
    layout.block_stride = d.kv_size * block_bytes;
    layout.kv_stride = block_bytes;
  }
  if (d.block_stride_elems != 0) {
    TORCH_CHECK(format == EngineKVFormat::NL_X_NB_BS_HS,
                "block_stride_elems is supported only for NL_X_NB_BS_HS");
    const size_t stride =
        static_cast<size_t>(d.block_stride_elems) * d.element_size;
    TORCH_CHECK(stride >= block_bytes, "block stride is smaller than a block");
    layout.block_stride = stride;
  }
  if (layout.blocked_scale) {
    TORCH_CHECK(d.element_size == 1 && d.hs > 4,
                "Blocked-scale rows require byte elements and hs > 4");
  }
  return layout;
}

template <typename Unit, bool ToEngine>
void submit_blocks(sycl::queue& queue, const uintptr_t* paged_ptrs,
                   std::array<uintptr_t, kObjectsPerLaunch> objects,
                   const int64_t* ids, size_t first_block, size_t end_block,
                   size_t first_object, size_t blocks_per_object,
                   PageBufferShapeDesc d, int chunk_tokens,
                   BlockLayout layout) {
  const size_t head_units =
      static_cast<size_t>(d.hs) * d.element_size / sizeof(Unit);
  const size_t row_units = d.nh * head_units;
  const size_t block_units = d.bs * row_units;
  const size_t tile_units = kTileBytes / sizeof(Unit);
  const size_t tiles = (block_units + tile_units - 1) / tile_units;
  const size_t blocks = end_block - first_block;
  const size_t groups = static_cast<size_t>(d.kv_size) * d.nl * blocks * tiles;
  queue.parallel_for(
      sycl::nd_range<1>(groups * kWorkGroupSize, kWorkGroupSize),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        size_t group = item.get_group(0);
        const size_t tile = group % tiles;
        group /= tiles;
        const size_t flat_block = first_block + group % blocks;
        group /= blocks;
        const size_t layer = group % d.nl;
        const size_t kv = group / d.nl;
        const int64_t engine_block = ids[flat_block];
        // Bad device-side indices must fail, not silently omit data.
        if (engine_block < 0 || engine_block >= d.nb) __builtin_trap();
        const size_t object_idx = flat_block / blocks_per_object - first_object;
        const size_t block_in_object = flat_block % blocks_per_object;
        const size_t pointer_idx =
            layer * layout.pointer_layer_stride + kv * layout.pointer_kv_stride;
        auto* engine = reinterpret_cast<uint8_t*>(paged_ptrs[pointer_idx]) +
                       engine_block * layout.block_stride +
                       layer * layout.layer_stride + kv * layout.kv_stride;
        auto* object = reinterpret_cast<uint8_t*>(objects[object_idx]) +
                       ((kv * d.nl + layer) * chunk_tokens * row_units +
                        block_in_object * block_units) *
                           sizeof(Unit);
        const size_t end = sycl::min(block_units, (tile + 1) * tile_units);
        for (size_t i = tile * tile_units + item.get_local_id(0); i < end;
             i += kWorkGroupSize) {
          size_t engine_i = i;
          if (layout.hnd) {
            const size_t token = i / row_units;
            const size_t head = (i % row_units) / head_units;
            engine_i = (head * d.bs + token) * head_units + i % head_units;
          } else if (layout.blocked_scale) {
            const size_t token = i / row_units;
            const size_t column = i % row_units;
            const size_t value_units = row_units - 4 / sizeof(Unit);
            engine_i = column < value_units
                           ? token * value_units + column
                           : d.bs * value_units + token * (4 / sizeof(Unit)) +
                                 column - value_units;
          }
          uint8_t* engine_addr = engine + engine_i * sizeof(Unit);
          uint8_t* object_addr = object + i * sizeof(Unit);
          uint8_t* dst = ToEngine ? engine_addr : object_addr;
          const uint8_t* src = ToEngine ? object_addr : engine_addr;
          if ((reinterpret_cast<uintptr_t>(dst) |
               reinterpret_cast<uintptr_t>(src)) %
                  alignof(Unit) ==
              0) {
            *reinterpret_cast<Unit*>(dst) = *reinterpret_cast<const Unit*>(src);
          } else {
            for (size_t b = 0; b < sizeof(Unit); ++b) dst[b] = src[b];
          }
        }
      });
}

template <typename Unit>
void launch_blocks(sycl::queue& queue, const uintptr_t* paged_ptrs,
                   const uintptr_t* objects, size_t num_objects,
                   const int64_t* ids, size_t num_ids,
                   TransferDirection direction, PageBufferShapeDesc d,
                   int chunk_tokens, BlockLayout layout, size_t skip) {
  const size_t blocks_per_object = chunk_tokens / d.bs;
  for (size_t start = 0; start < num_objects; start += kObjectsPerLaunch) {
    const size_t count = std::min(kObjectsPerLaunch, num_objects - start);
    const size_t first = std::max(skip, start * blocks_per_object);
    const size_t end = std::min(num_ids, (start + count) * blocks_per_object);
    if (first >= end) continue;
    std::array<uintptr_t, kObjectsPerLaunch> batch{};
    std::copy_n(objects + start, count, batch.begin());
    if (direction == TransferDirection::H2D) {
      submit_blocks<Unit, true>(queue, paged_ptrs, batch, ids, first, end,
                                start, blocks_per_object, d, chunk_tokens,
                                layout);
    } else {
      submit_blocks<Unit, false>(queue, paged_ptrs, batch, ids, first, end,
                                 start, blocks_per_object, d, chunk_tokens,
                                 layout);
    }
  }
}

// Host-visible argument checks shared by the tensor entry point and the plan
// executor.
void check_transfer_args(TransferDirection direction, PageBufferShapeDesc d,
                         int lmcache_chunk_size) {
  TORCH_CHECK(direction == TransferDirection::H2D ||
                  direction == TransferDirection::D2H,
              "Unsupported transfer direction");
  TORCH_CHECK(d.nl > 0 && d.nb > 0 && d.bs > 0 && d.nh > 0 && d.hs > 0 &&
                  d.block_stride_elems >= 0,
              "Shape dimensions must be positive and block stride "
              "non-negative");
  TORCH_CHECK(d.element_size == 1 || d.element_size == 2 || d.element_size == 4,
              "element_size must be 1, 2, or 4");
  TORCH_CHECK(lmcache_chunk_size > 0 && lmcache_chunk_size % d.bs == 0,
              "chunk size must be a positive multiple of block size");
}

void check_object_ptrs(sycl::queue& queue, const uintptr_t* ptrs, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    TORCH_CHECK(ptrs[i] != 0, "LMCache object pointer must not be null");
    auto kind = sycl::get_pointer_type(reinterpret_cast<void*>(ptrs[i]),
                                       queue.get_context());
    TORCH_CHECK(
        kind != sycl::usm::alloc::unknown,
        "Object pointers must be XPU USM device/shared/host allocations; "
        "stage pageable host memory before native transfer");
  }
}

// Number of paged-pointer entries the kernel indexes for this geometry.
int64_t expected_paged_ptr_count(PageBufferShapeDesc d, BlockLayout layout) {
  return static_cast<int64_t>(d.nl - 1) * layout.pointer_layer_stride +
         static_cast<int64_t>(d.kv_size - 1) * layout.pointer_kv_stride + 1;
}

// Pick the widest copy unit the row width allows and enqueue the transfer.
// All arguments must already be validated.
void dispatch_transfer(sycl::queue& queue, const uintptr_t* paged_ptrs,
                       const uintptr_t* objects, size_t num_objects,
                       const int64_t* ids, size_t num_ids,
                       TransferDirection direction, PageBufferShapeDesc d,
                       int chunk_tokens, BlockLayout layout, size_t skip) {
  const size_t head_bytes = static_cast<size_t>(d.hs) * d.element_size;
#define LAUNCH(Unit)                                                         \
  launch_blocks<Unit>(queue, paged_ptrs, objects, num_objects, ids, num_ids, \
                      direction, d, chunk_tokens, layout, skip)
  if (layout.blocked_scale) {
    if (head_bytes % 4 == 0) {
      LAUNCH(uint32_t);
    } else {
      LAUNCH(uint8_t);
    }
  } else if (head_bytes % 16 == 0) {
    LAUNCH(sycl::uint4);
  } else if (head_bytes % 4 == 0) {
    LAUNCH(uint32_t);
  } else if (head_bytes % 2 == 0) {
    LAUNCH(uint16_t);
  } else {
    LAUNCH(uint8_t);
  }
#undef LAUNCH
}

// Whether ptr is device-accessible USM memory the kernel may index.
bool is_device_usm(sycl::queue& queue, uintptr_t ptr) {
  const auto kind =
      sycl::get_pointer_type(reinterpret_cast<void*>(ptr), queue.get_context());
  return kind == sycl::usm::alloc::device || kind == sycl::usm::alloc::shared;
}

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    const std::vector<uintptr_t>& lmcache_objects_ptrs,
    const torch::Tensor& block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, EngineKVFormat engine_kv_format,
    int skip_prefix_n_blocks) {
  TORCH_CHECK(device.is_xpu(), "device must be XPU");
  const c10::DeviceGuard guard(device);
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  check_transfer_args(direction, shape_desc, lmcache_chunk_size);
  TORCH_CHECK(skip_prefix_n_blocks >= 0, "skip_prefix_n_blocks must be >= 0");
  const BlockLayout layout = resolve_layout(shape_desc, engine_kv_format);
  TORCH_CHECK(
      paged_buffer_ptrs_tensor.is_xpu() &&
          paged_buffer_ptrs_tensor.get_device() == c10::xpu::current_device() &&
          paged_buffer_ptrs_tensor.dim() == 1 &&
          paged_buffer_ptrs_tensor.is_contiguous() &&
          (paged_buffer_ptrs_tensor.scalar_type() == at::kLong ||
           paged_buffer_ptrs_tensor.scalar_type() == at::kUInt64),
      "paged pointers must be a contiguous int64/uint64 vector on device");
  TORCH_CHECK(paged_buffer_ptrs_tensor.numel() ==
                  expected_paged_ptr_count(shape_desc, layout),
              "paged pointer count does not match shape/format");
  TORCH_CHECK(block_ids.is_xpu() &&
                  block_ids.get_device() == c10::xpu::current_device() &&
                  block_ids.dim() == 1 && block_ids.is_contiguous() &&
                  block_ids.scalar_type() == at::kLong,
              "block_ids must be a contiguous int64 vector on device");
  TORCH_CHECK(
      static_cast<size_t>(block_ids.numel()) <=
          lmcache_objects_ptrs.size() * (lmcache_chunk_size / shape_desc.bs),
      "block_ids exceed object capacity");
  check_object_ptrs(queue, lmcache_objects_ptrs.data(),
                    lmcache_objects_ptrs.size());
  dispatch_transfer(
      queue, static_cast<const uintptr_t*>(paged_buffer_ptrs_tensor.data_ptr()),
      lmcache_objects_ptrs.data(), lmcache_objects_ptrs.size(),
      block_ids.data_ptr<int64_t>(), static_cast<size_t>(block_ids.numel()),
      direction, shape_desc, lmcache_chunk_size, layout,
      static_cast<size_t>(skip_prefix_n_blocks));
}

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  TORCH_CHECK(device.is_xpu(), "device must be XPU");
  // One guard for the whole plan: every staging copy and launch below goes to
  // this device's current (in-order) queue, in plan order.
  const c10::DeviceGuard guard(device);
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  TORCH_CHECK(direction == TransferDirection::H2D ||
                  direction == TransferDirection::D2H,
              "Unsupported transfer direction");
  TORCH_CHECK(host_buffer_alignment != 0 &&
                  (host_buffer_alignment & (host_buffer_alignment - 1)) == 0,
              "host_buffer_alignment must be a power of two");

  // Validate each kernel group once instead of once per launch: the
  // get_pointer_type queries are Level Zero calls, and every launch of a
  // group reuses the same paged pointers, block ids and temp buffers.
  std::vector<BlockLayout> layouts;
  layouts.reserve(kernel_group_specs.size());
  for (const auto& group : kernel_group_specs) {
    check_transfer_args(direction, group.shape_desc, group.lmcache_chunk_size);
    layouts.push_back(resolve_layout(group.shape_desc, group.engine_kv_format));
    TORCH_CHECK(group.paged_buffer_ptrs != 0 &&
                    is_device_usm(queue, group.paged_buffer_ptrs),
                "KernelGroupSpec.paged_buffer_ptrs must be XPU device memory");
    TORCH_CHECK(group.block_ids_capacity >= 0,
                "KernelGroupSpec.block_ids_capacity must be non-negative");
    TORCH_CHECK(group.block_ids_capacity == 0 ||
                    (group.block_ids_base != 0 &&
                     is_device_usm(queue, group.block_ids_base)),
                "KernelGroupSpec.block_ids_base must be XPU device memory");
    check_object_ptrs(queue, group.lmcache_objects_ptrs.data(),
                      group.lmcache_objects_ptrs.size());
  }

  // Validate every launch before enqueuing anything, so a bad plan fails
  // without having moved part of the object group.
  for (const auto& step : batch_steps) {
    for (const auto& launch : step.launches) {
      TORCH_CHECK(
          launch.group_idx >= 0 &&
              launch.group_idx < static_cast<int>(kernel_group_specs.size()),
          "LaunchVar.group_idx out of range: ", launch.group_idx);
      const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
      TORCH_CHECK(launch.num_objects >= 1 &&
                      launch.num_objects <=
                          static_cast<int>(group.lmcache_objects_ptrs.size()),
                  "LaunchVar.num_objects (", launch.num_objects,
                  ") exceeds available temp buffers (",
                  group.lmcache_objects_ptrs.size(), ")");
      TORCH_CHECK(launch.block_ids_offset >= 0,
                  "LaunchVar.block_ids_offset must be non-negative, got ",
                  launch.block_ids_offset);
      TORCH_CHECK(launch.total_blocks >= 0,
                  "LaunchVar.total_blocks must be non-negative, got ",
                  launch.total_blocks);
      TORCH_CHECK(launch.block_ids_offset + launch.total_blocks <=
                      group.block_ids_capacity,
                  "LaunchVar block_ids slice [", launch.block_ids_offset, ", ",
                  launch.block_ids_offset + launch.total_blocks,
                  ") exceeds block_ids capacity ", group.block_ids_capacity);
      TORCH_CHECK(static_cast<int64_t>(launch.total_blocks) <=
                      static_cast<int64_t>(launch.num_objects) *
                          (group.lmcache_chunk_size / group.shape_desc.bs),
                  "LaunchVar.total_blocks exceeds object capacity");
      TORCH_CHECK(launch.skip_prefix_n_blocks >= 0,
                  "LaunchVar.skip_prefix_n_blocks must be >= 0");
    }
  }

  const bool is_h2d = direction == TransferDirection::H2D;
  const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
    for (const auto& copy : staging) {
      lmcache_memcpy_async(copy.dest, copy.src, copy.nbytes, direction,
                           copy.host_offset, host_buffer_alignment);
    }
  };

  for (const auto& step : batch_steps) {
    // H2D fills the temp buffers before the kernels read them; D2H drains
    // them after the kernels write. Temp buffers are reused across steps, and
    // the in-order queue serializes each step against the next.
    if (is_h2d) do_staging(step.staging);
    for (const auto& launch : step.launches) {
      const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
      const auto* ids = reinterpret_cast<const int64_t*>(group.block_ids_base) +
                        launch.block_ids_offset;
      dispatch_transfer(
          queue, reinterpret_cast<const uintptr_t*>(group.paged_buffer_ptrs),
          group.lmcache_objects_ptrs.data(),
          static_cast<size_t>(launch.num_objects), ids,
          static_cast<size_t>(launch.total_blocks), direction, group.shape_desc,
          group.lmcache_chunk_size, layouts[launch.group_idx],
          static_cast<size_t>(launch.skip_prefix_n_blocks));
    }
    if (!is_h2d) do_staging(step.staging);
  }
}

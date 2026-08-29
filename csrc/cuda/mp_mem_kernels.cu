// SPDX-License-Identifier: Apache-2.0

#include "mp_mem_kernels_device.cuh"
#include "mp_mem_kernels_layerwise.cuh"

namespace {

using namespace lmcache_mp;

// Per-chunk kernel: MemoryObj4 passed by value (zero GPU alloc, <=4 objects)
template <typename ScalarType, bool lmcache_to_engine, EngineKVFormat format>
__global__ void multi_layer_block_transfer_kernel(
    MemoryObj4<ScalarType> lmcache_objects,
    ScalarType** __restrict__ paged_buffer_ptrs,
    const int64_t* engine_block_ids,
    const int num_blocks_per_object,  // e.g. 16 for lmcache chunk size =
                                      // 256 and block size = 16
    const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size,  // e.g., 256, used to calculate global offset
                                   // in LMCache object
    const int skip_prefix_n_blocks) {
  // blockIdx.y spans all blocks across all objects (total_blocks).
  // Derive which object and local block index from the flat index.
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;
  const int engine_block_idx = engine_block_ids[flat_block_idx];
  multi_layer_block_transfer_single_block<ScalarType, lmcache_to_engine,
                                          format>(
      lmcache_objects.objects[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs,  // offset in LMCache object
      shape_desc, lmcache_chunk_size);
}

template <typename ScalarType>
void multi_layer_block_kv_transfer_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);

  const MultiLayerLaunchConfig launch_cfg =
      make_multi_layer_launch_config<ScalarType>(
          block_ids, num_objects, shape_desc, lmcache_chunk_size);
  const int num_blocks_per_object = launch_cfg.num_blocks_per_object;

  // --- Build MemoryObj4 ---
  MemoryObj4<ScalarType> lmcache_obj4;
  lmcache_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    lmcache_obj4.objects[i] =
        (i < num_objects)
            ? reinterpret_cast<ScalarType*>(lmcache_objects_ptrs[i])
            : nullptr;
  }

  // --- Build paged buffer pointer array ---
  ScalarType** paged_buffer_ptrs =
      reinterpret_cast<ScalarType**>(paged_buffer_ptrs_tensor.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // --- block_ids is a GPU int64 tensor, read directly ---
  const int64_t* block_ids_ptr = block_ids.data_ptr<int64_t>();

  // --- Grid and block dimensions (shared with the other path) ---
  const dim3 block = launch_cfg.block;
  const dim3 grid = launch_cfg.grid;

  if (direction == TransferDirection::H2D) {
    DISPATCH_FORMAT(multi_layer_block_transfer_kernel, true, lmcache_obj4);
  } else {
    DISPATCH_FORMAT(multi_layer_block_transfer_kernel, false, lmcache_obj4);
  }
}

#undef DISPATCH_FORMAT
#undef LAUNCH_KERNEL

}  // namespace

#define LAUNCH_TEMPLATED(type)                                             \
  do {                                                                     \
    multi_layer_block_kv_transfer_templated<type>(                         \
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs, block_ids, device, \
        direction, shape_desc, lmcache_chunk_size, engine_kv_format,       \
        skip_prefix_n_blocks);                                             \
  } while (0)

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks, bool layerwise) {
  if (layerwise) {
    // Layer-wise kernels live in their own translation unit; see
    // mp_mem_kernels_layerwise.cu.  Everything below this branch is the
    // untouched per-chunk path.
    multi_layer_block_kv_transfer_layerwise(
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs.data(),
        static_cast<int>(lmcache_objects_ptrs.size()), block_ids, device,
        direction, shape_desc, lmcache_chunk_size, engine_kv_format,
        skip_prefix_n_blocks);
    return;
  }

  // Transfer-unit selection. Deliberately duplicated rather than shared:
  // the two paths expand different launch macros. Keep in sync with
  // multi_layer_block_kv_transfer_layerwise() in mp_mem_kernels_layerwise.cu
  // -- a new EngineKVFormat or a change to the vectorization rule must be
  // applied in both places, or the layer-wise path silently keeps the old
  // width.
  int head_bytes = shape_desc.hs * shape_desc.element_size;
  TORCH_CHECK(head_bytes % sizeof(uint16_t) == 0, "head_size * element_size (",
              head_bytes, ") must be divisible by 2 for vectorized access");

  if (engine_kv_format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    // Blocked-scale indexer cache: the per-token fp32 scale must be a whole
    // number of transfer units, so pin 4-byte units regardless of row width.
    TORCH_CHECK(head_bytes % sizeof(uint32_t) == 0,
                "NL_X_NB_BSV_BSS row bytes (", head_bytes,
                ") must be divisible by 4");
    LAUNCH_TEMPLATED(uint32_t);
    return;
  }

  if (head_bytes % sizeof(uint4) == 0) {
    LAUNCH_TEMPLATED(uint4);  // 16 bytes per copy
  } else if (head_bytes % sizeof(uint32_t) == 0) {
    LAUNCH_TEMPLATED(uint32_t);  // 4 bytes per copy
  } else {
    LAUNCH_TEMPLATED(uint16_t);  // 2 bytes per copy (minimum granularity)
  }
}

#undef LAUNCH_TEMPLATED

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps, bool layerwise) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in order.
  const at::cuda::OptionalCUDAGuard device_guard(device);
  const bool is_h2d = (direction == TransferDirection::H2D);
  const auto int64_opts = at::TensorOptions().dtype(at::kLong).device(device);

  const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
    for (const auto& copy : staging) {
      lmcache_memcpy_async(copy.dest, copy.src, copy.nbytes, direction,
                           copy.host_offset, host_buffer_alignment);
    }
  };

  for (const auto& step : batch_steps) {
    // H2D stages CPU->GPU temp buffers before the kernel reads them; D2H stages
    // GPU->CPU after the kernel writes them. The per-step ordering must be
    // preserved because temp buffers are reused across steps.
    if (is_h2d) {
      do_staging(step.staging);
    }
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
      // Bounds-check the block_ids slice before the kernel dereferences it on
      // device: an out-of-range offset/length would otherwise be a silent
      // out-of-bounds device read (CUDA fault or garbage), not a clean error.
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

      // Wrap the plan's pre-resolved raw device addresses as non-owning tensor
      // views so we can reuse the existing multi_layer_block_kv_transfer entry
      // point without touching any of its code. The backing storage is owned by
      // the caller's tensors (kept alive for the duration of this call); these
      // views only carry the pointer/shape each launch needs. Downstream only
      // reads paged_buffer_ptrs_tensor.data_ptr() and block_ids.{data_ptr,
      // size(0)}.
      const uintptr_t block_ids_addr =
          group.block_ids_base +
          static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t);
      const at::Tensor paged_buffer_ptrs_tensor = at::from_blob(
          reinterpret_cast<void*>(group.paged_buffer_ptrs), {1}, int64_opts);
      const at::Tensor block_ids = at::from_blob(
          reinterpret_cast<void*>(block_ids_addr),
          {static_cast<int64_t>(launch.total_blocks)}, int64_opts);
      std::vector<int64_t> lmcache_objects_ptrs(
          group.lmcache_objects_ptrs.begin(),
          group.lmcache_objects_ptrs.begin() + launch.num_objects);

      multi_layer_block_kv_transfer(
          paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs), block_ids,
          device, direction, group.shape_desc, group.lmcache_chunk_size,
          group.engine_kv_format, launch.skip_prefix_n_blocks, layerwise);
    }
    if (!is_h2d) {
      do_staging(step.staging);
    }
  }
}

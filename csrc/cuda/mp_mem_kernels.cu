// SPDX-License-Identifier: Apache-2.0

#include "mp_mem_kernels_common.cuh"
#include "mp_mem_kernels.cuh"

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

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  dispatch_by_transfer_unit(
      shape_desc, engine_kv_format, [&](auto transfer_unit) {
        using ScalarType = typename decltype(transfer_unit)::type;
        multi_layer_block_kv_transfer_templated<ScalarType>(
            paged_buffer_ptrs_tensor, lmcache_objects_ptrs, block_ids, device,
            direction, shape_desc, lmcache_chunk_size, engine_kv_format,
            skip_prefix_n_blocks);
      });
}

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  execute_object_group_transfer_common(
      direction, device, host_buffer_alignment, kernel_group_specs, batch_steps,
      [&](const KernelGroupSpec& group, const LaunchVar& launch,
          const at::Tensor& paged_buffer_ptrs_tensor,
          const at::Tensor& block_ids) {
        std::vector<int64_t> lmcache_objects_ptrs(
            group.lmcache_objects_ptrs.begin(),
            group.lmcache_objects_ptrs.begin() + launch.num_objects);
        multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs),
            block_ids, device, direction, group.shape_desc,
            group.lmcache_chunk_size, group.engine_kv_format,
            launch.skip_prefix_n_blocks);
      });
}

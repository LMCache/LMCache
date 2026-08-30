// SPDX-License-Identifier: Apache-2.0
//
// Layer-wise multi-layer block KV transfer.
//
// Kept in its own translation unit so the layer-wise kernel templates are
// instantiated exactly once, independently of the per-chunk ones in
// mp_mem_kernels.cu. What the two paths share -- the device-side helpers and
// the plan-walking driver -- lives in mp_mem_kernels_common.cuh.

#include "mp_mem_kernels_common.cuh"
#include "mp_mem_kernels_layerwise.cuh"

#include <cstring>

namespace {

using namespace lmcache_mp;

// Per-layer kernel: ScalarType** via GPU device pointer array (arbitrary N)
template <typename ScalarType, bool lmcache_to_engine, EngineKVFormat format>
__global__ void multi_layer_block_transfer_kernel_layerwise(
    ScalarType** __restrict__ lmcache_object_ptrs,
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
  multi_layer_block_transfer_single_block<ScalarType, lmcache_to_engine, format,
                                          /*allow_interleaved=*/true>(
      lmcache_object_ptrs[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs,  // offset in LMCache object
      shape_desc, lmcache_chunk_size);
}

template <typename ScalarType>
void multi_layer_block_kv_transfer_layerwise_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    const int64_t* lmcache_objects_ptrs, int num_objects,
    const torch::Tensor& block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, EngineKVFormat engine_kv_format,
    int skip_prefix_n_blocks) {
  // --- Validation ---
  TORCH_CHECK(num_objects >= 1, "Expected at least 1 LMCache object, got ",
              num_objects);

  const MultiLayerLaunchConfig launch_cfg =
      make_multi_layer_launch_config<ScalarType>(
          block_ids, num_objects, shape_desc, lmcache_chunk_size);
  const int num_blocks_per_object = launch_cfg.num_blocks_per_object;

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

  // Per-layer path: reusable pinned+device buffer for pointer upload.
  // Avoids per-launch torch::empty (caching allocator overhead) and
  // pageable copy_ (which forces a CPU-blocking bounce buffer in the
  // CUDA driver).  Fixed 1024-element buffers (8 KB each) are allocated
  // once on first use and reused forever; the pinned→device copy is
  // truly async with zero CPU stall.
  static constexpr int kMaxObjects = 1024;
  static thread_local int64_t* pinned_host_ptr = nullptr;
  static thread_local torch::Tensor dev_buf_tensor;
  static thread_local int dev_buf_device_index = -1;

  TORCH_CHECK(num_objects <= kMaxObjects, "Layerwise path supports at most ",
              kMaxObjects, " objects, got ", num_objects);

  const int dev_idx = device.index();

  // One-time allocation of pinned host buffer
  if (!pinned_host_ptr) {
    auto err =
        cudaHostAlloc(reinterpret_cast<void**>(&pinned_host_ptr),
                      kMaxObjects * sizeof(int64_t), cudaHostAllocDefault);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  // One-time allocation of device buffer (or on device change)
  if (dev_buf_device_index != dev_idx) {
    dev_buf_tensor = torch::empty(
        {kMaxObjects},
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    dev_buf_device_index = dev_idx;
  }

  // pinned staging → device (truly async, zero CPU stall)
  std::memcpy(pinned_host_ptr, lmcache_objects_ptrs,
              num_objects * sizeof(int64_t));
  cudaMemcpyAsync(dev_buf_tensor.data_ptr(), pinned_host_ptr,
                  num_objects * sizeof(int64_t), cudaMemcpyHostToDevice,
                  stream);

  ScalarType** lmcache_ptrs_dev =
      reinterpret_cast<ScalarType**>(dev_buf_tensor.data_ptr());
  if (direction == TransferDirection::H2D) {
    DISPATCH_FORMAT(multi_layer_block_transfer_kernel_layerwise, true,
                    lmcache_ptrs_dev);
  } else {
    DISPATCH_FORMAT(multi_layer_block_transfer_kernel_layerwise, false,
                    lmcache_ptrs_dev);
  }
}

#undef DISPATCH_FORMAT
#undef LAUNCH_KERNEL

}  // namespace

void multi_layer_block_kv_transfer_layerwise(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    const int64_t* lmcache_objects_ptrs, int num_objects,
    const torch::Tensor& block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, EngineKVFormat engine_kv_format,
    int skip_prefix_n_blocks) {
  dispatch_by_transfer_unit(
      shape_desc, engine_kv_format, [&](auto transfer_unit) {
        using ScalarType = typename decltype(transfer_unit)::type;
        multi_layer_block_kv_transfer_layerwise_templated<ScalarType>(
            paged_buffer_ptrs_tensor, lmcache_objects_ptrs, num_objects,
            block_ids, device, direction, shape_desc, lmcache_chunk_size,
            engine_kv_format, skip_prefix_n_blocks);
      });
}

void execute_object_group_transfer_layerwise(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  execute_object_group_transfer_common(
      direction, device, host_buffer_alignment, kernel_group_specs, batch_steps,
      [&](const KernelGroupSpec& group, const LaunchVar& launch,
          const at::Tensor& paged_buffer_ptrs_tensor,
          const at::Tensor& block_ids) {
        // The layer-wise kernel reads the first num_objects entries directly,
        // so no slice copy is needed (num_objects is bounds-checked above).
        multi_layer_block_kv_transfer_layerwise(
            paged_buffer_ptrs_tensor, group.lmcache_objects_ptrs.data(),
            launch.num_objects, block_ids, device, direction, group.shape_desc,
            group.lmcache_chunk_size, group.engine_kv_format,
            launch.skip_prefix_n_blocks);
      });
}

// SPDX-License-Identifier: Apache-2.0
//
// Layer-wise multi-layer block KV transfer.
//
// Kept in its own translation unit so the layer-wise kernel templates are
// instantiated exactly once, independently of the per-chunk ones in
// mp_mem_kernels.cu. What the two paths share -- the device-side helpers and
// the plan-walking driver -- lives in mp_mem_kernels_detail.cuh.

#include "mp_mem_kernels_detail.cuh"
#include "mp_mem_kernels_layerwise.cuh"

#include <array>
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

// Upper bound on LMCache objects per layer-wise launch.  The pointer array is
// far too large for kernel parameter space, so it is staged through pinned
// host memory and uploaded to the device.
constexpr int kMaxObjects = 1024;

// Independent staging slots kept per thread.  One slot is occupied per
// distinct pointer array, so this need only cover the number of kernel groups
// a thread interleaves -- in practice one or two.
constexpr int kUploadSlots = 4;

// Uploads num_objects LMCache object pointers and returns their device address.
//
// The host side of an in-flight cudaMemcpyAsync must not be rewritten before
// the copy actually runs, and nothing on this path synchronises with the
// stream.  A single shared staging buffer would therefore let one launch
// overwrite pointers that an earlier launch has queued but not yet copied.
// Slots are handed out per distinct pointer array instead: a kernel group's
// staging-slot pointers are identical for every layer, so repeat launches find
// their array already resident, reuse that slot and upload nothing, while a
// different kernel group lands in a slot of its own.  Wrapping past
// kUploadSlots distinct arrays waits on that slot's previous upload, so
// correctness never depends on the host staying behind the device.
int64_t* upload_object_ptrs(const int64_t* lmcache_objects_ptrs,
                            int num_objects, const torch::Device& device,
                            cudaStream_t stream) {
  TORCH_CHECK(num_objects <= kMaxObjects, "Layerwise path supports at most ",
              kMaxObjects, " objects, got ", num_objects);

  static thread_local int64_t* pinned_host_ptr = nullptr;
  static thread_local torch::Tensor dev_buf_tensor;
  static thread_local int dev_buf_device_index = -1;
  static thread_local std::array<cudaEvent_t, kUploadSlots> slot_uploaded{};
  static thread_local std::array<cudaStream_t, kUploadSlots> slot_stream{};
  static thread_local std::array<int, kUploadSlots> slot_len{};
  static thread_local int next_slot = 0;

  const size_t nbytes = static_cast<size_t>(num_objects) * sizeof(int64_t);

  // One-time allocation of the pinned host buffer and the per-slot events.
  if (!pinned_host_ptr) {
    auto err = cudaHostAlloc(
        reinterpret_cast<void**>(&pinned_host_ptr),
        static_cast<size_t>(kMaxObjects) * kUploadSlots * sizeof(int64_t),
        cudaHostAllocDefault);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostAlloc failed: ", cudaGetErrorString(err));
    for (int i = 0; i < kUploadSlots; ++i) {
      err = cudaEventCreateWithFlags(&slot_uploaded[i], cudaEventDisableTiming);
      TORCH_CHECK(err == cudaSuccess,
                  "cudaEventCreateWithFlags failed: ", cudaGetErrorString(err));
    }
  }

  // One-time allocation of the device buffer (or on device change).  Resident
  // copies do not survive the reallocation, so the slot state is dropped.
  if (dev_buf_device_index != device.index()) {
    dev_buf_tensor = torch::empty(
        {static_cast<int64_t>(kMaxObjects) * kUploadSlots},
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    dev_buf_device_index = device.index();
    slot_len.fill(0);
    next_slot = 0;
  }

  int64_t* dev_base = static_cast<int64_t*>(dev_buf_tensor.data_ptr());

  // Already resident from an earlier launch on this stream: reuse it and skip
  // the upload entirely.  Same stream, so the earlier copy is ordered ahead of
  // the kernel about to be launched.  A longer resident array also serves a
  // shorter one, since the kernel only reads the first num_objects entries --
  // that is the trailing partial pass of a chunk sweep, whose pointers are a
  // prefix of the full passes before it.
  for (int i = 0; i < kUploadSlots; ++i) {
    if (slot_len[i] >= num_objects && slot_stream[i] == stream &&
        std::memcmp(pinned_host_ptr + static_cast<size_t>(i) * kMaxObjects,
                    lmcache_objects_ptrs, nbytes) == 0) {
      return dev_base + static_cast<size_t>(i) * kMaxObjects;
    }
  }

  const int slot = next_slot;
  next_slot = (next_slot + 1) % kUploadSlots;
  if (slot_len[slot] != 0) {
    // Reclaiming a slot: its previous upload must have drained the host side
    // before we overwrite it.
    cudaEventSynchronize(slot_uploaded[slot]);
  }

  int64_t* host_slot =
      pinned_host_ptr + static_cast<size_t>(slot) * kMaxObjects;
  int64_t* dev_slot = dev_base + static_cast<size_t>(slot) * kMaxObjects;
  std::memcpy(host_slot, lmcache_objects_ptrs, nbytes);
  cudaMemcpyAsync(dev_slot, host_slot, nbytes, cudaMemcpyHostToDevice, stream);
  cudaEventRecord(slot_uploaded[slot], stream);
  slot_stream[slot] = stream;
  slot_len[slot] = num_objects;
  return dev_slot;
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

  ScalarType** lmcache_ptrs_dev = reinterpret_cast<ScalarType**>(
      upload_object_ptrs(lmcache_objects_ptrs, num_objects, device, stream));

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

// SPDX-License-Identifier: Apache-2.0

#include "mp_mem_kernels.cuh"

namespace {

/**
 * Key logic in the kernel implementation:
 * 1. Each thread block is for (BS, NH, HS) part (i.e., a single block in the
 * paged buffer)
 * 2. Within a thread block, each warp is for a single head. Number of warps
 * in a thread block is equal to the number of heads (NH).
 * 3. Within a thread block, we do the loop over the BS (i.e., number of tokens
 * in the block) dimension.
 * 4. The grid will take over (2, NB, NL) dimensions. No matter what the actual
 * layout in memory is, we will calculate the global offset for the start of the
 * block
 * 5. For LMCache, we assume it is always using 2LTD layout, e.g.,
 * [2, L, 256, NH * HS], where 256 means that 256 tokens
 */

/**
 * Calculate the offset for the current block in the paged buffer
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_engine_global_offset(
    const int k_or_v, const int engine_block_idx, const int layer_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_block = shape_desc.scalars_per_block<ScalarType>();
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
    // Cross-layer: single tensor [NB, NL, 2, BS, NH, HS]
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  } else if constexpr (format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    // Normal: L tensors [2, NB, BS, NH, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    // Flash Infer: L tensors [NB, 2, BS, NH, HS]
    return engine_block_idx * shape_desc.kv_size * scalars_per_block +
           k_or_v * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NB_BS_HS) {
    // MLA: L tensors [NB, BS, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA: 2L tensors [NBBS, NH, HS] — K/V via separate tensor ptrs
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NBBS_ONE_HS) {
    // SGLang MLA: L tensors [NBBS, 1, HS]
    return engine_block_idx * scalars_per_block;
  }
}

/**
 * Calculate the offset for the current token against the start
 * of the block in the paged buffer.
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_engine_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  // for now, everything is BS, NH, HS, so we only have a single case
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

/**
 * Calculate the global offset for the current `block` in the LMCache object.
 * The `block` here is the memory region corresponding to a thread-block.
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_lmcache_global_offset(
    const int k_or_v,
    const int
        token_offset_in_lmcache_object,  // 0~255 if LMCache chunk size is 256
    const int layer_idx,
    const int lmcache_chunk_size,  // e.g., 256
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  // LMCache is using 2LTD all the times
  return token_offset_in_lmcache_object * scalars_per_token +
         layer_idx * lmcache_chunk_size * scalars_per_token +
         k_or_v * shape_desc.nl * lmcache_chunk_size * scalars_per_token;
}

/**
 * Calculate the local offset for the current token against the start of the
 * block in the LMCache object.
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_lmcache_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

__device__ inline uint4 ld_cs(const uint4* addr) {
#ifdef __CUDA_ARCH__
  uint4 val;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
#else
  return *addr;
#endif
}

__device__ inline void st_cs(uint4* addr, uint4 val) {
#ifdef __CUDA_ARCH__
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
  *addr = val;
#endif
}

template <typename ScalarType>
__device__ inline void warp_copy(ScalarType* __restrict__ dst,
                                 const ScalarType* __restrict__ src,
                                 size_t num_elements) {
  int idx = threadIdx.x;
  int stride = blockDim.x;
  if constexpr (std::is_same_v<ScalarType, uint4>) {
    for (size_t i = idx; i < num_elements; i += stride) {
      st_cs(dst + i, ld_cs(src + i));
    }
  } else {
    for (size_t i = idx; i < num_elements; i += stride) {
      dst[i] = src[i];
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, GPUKVFormat format>
__device__ void multi_layer_block_transfer_single_block(
    ScalarType* __restrict__ lmcache_object,
    ScalarType** __restrict__ paged_buffer_ptrs, const int engine_block_idx,
    const int offset_in_lmcache_block, const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size  // e.g., 256, used to calculate global offset
                                  // in LMCache object
) {
  const int head_idx = threadIdx.y;
  const int k_or_v = blockIdx.x;
  const int layer_idx = blockIdx.z;

  const size_t engine_global_offset =
      calculate_engine_global_offset<ScalarType, format>(
          k_or_v, engine_block_idx, layer_idx, shape_desc);
  const size_t lmcache_global_offset =
      calculate_lmcache_global_offset<ScalarType, format>(
          k_or_v, offset_in_lmcache_block, layer_idx, lmcache_chunk_size,
          shape_desc);
  ScalarType* paged_buffer_layer_ptr;
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[0];
  } else if constexpr (format == GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA: ptrs[0..NL-1] = K per layer, ptrs[NL..2NL-1] = V per layer
    paged_buffer_layer_ptr =
        (ScalarType*)paged_buffer_ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[layer_idx];
  }

  for (int token_offset = 0; token_offset < shape_desc.bs; ++token_offset) {
    const size_t engine_local_offset =
        calculate_engine_local_offset<ScalarType, format>(token_offset,
                                                          head_idx, shape_desc);
    const size_t lmcache_local_offset =
        calculate_lmcache_local_offset<ScalarType, format>(
            token_offset, head_idx, shape_desc);
    ScalarType* engine_ptr =
        paged_buffer_layer_ptr + engine_global_offset + engine_local_offset;
    ScalarType* lmcache_ptr =
        lmcache_object + lmcache_global_offset + lmcache_local_offset;
    if constexpr (lmcache_to_engine) {
      warp_copy<ScalarType>(engine_ptr, lmcache_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    } else {
      warp_copy<ScalarType>(lmcache_ptr, engine_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, GPUKVFormat format>
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

#define LAUNCH_BLOCK_KERNEL_WITH_FORMAT(DIRECTION, FORMAT)               \
  multi_layer_block_transfer_kernel<uint4, DIRECTION, FORMAT>            \
      <<<grid, block, 0, stream>>>(lmcache_obj4, paged_buffer_ptrs,      \
                                   block_ids_ptr, num_blocks_per_object, \
                                   shape_desc, lmcache_chunk_size,       \
                                   skip_prefix_n_blocks);                \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);

  int total_blocks = block_ids.size(0);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (", num_objects,
              ")");
  int num_blocks_per_object = total_blocks / num_objects;

  TORCH_CHECK(num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");

  TORCH_CHECK(
      shape_desc.hs * shape_desc.element_size % sizeof(uint4) == 0,
      "head_size * element_size (", shape_desc.hs * shape_desc.element_size,
      ") must be divisible by ", sizeof(uint4), " for uint4 vectorized access");

  // --- Build MemoryObj4 ---
  MemoryObj4<uint4> lmcache_obj4;
  lmcache_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    lmcache_obj4.objects[i] =
        (i < num_objects) ? reinterpret_cast<uint4*>(lmcache_objects_ptrs[i])
                          : nullptr;
  }

  // --- Build paged buffer pointer array ---
  uint4** paged_buffer_ptrs =
      reinterpret_cast<uint4**>(paged_buffer_ptrs_tensor.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // --- block_ids is a GPU int64 tensor, read directly ---
  const int64_t* block_ids_ptr = block_ids.data_ptr<int64_t>();

  // --- Grid and block dimensions ---
  int elements_per_head =
      shape_desc.hs * shape_desc.element_size / sizeof(uint4);
  int thread_dim_x = std::min(elements_per_head, 32);
  int thread_dim_y = shape_desc.nh;

  dim3 block(thread_dim_x, thread_dim_y);
  dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

  // --- Dispatch on direction x format ---
  if (direction == TransferDirection::H2D) {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true,
                                        GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true,
                                        GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true,
                                        GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        TORCH_CHECK(false, "Unsupported GPUKVFormat: ",
                    static_cast<int>(gpu_kv_format));
    }
  } else {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false,
                                        GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false,
                                        GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false,
                                        GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        TORCH_CHECK(false, "Unsupported GPUKVFormat: ",
                    static_cast<int>(gpu_kv_format));
    }
  }
}

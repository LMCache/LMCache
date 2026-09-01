// SPDX-License-Identifier: Apache-2.0
//
// Internals shared by the per-chunk and layer-wise multi-layer block KV
// transfer kernels.
//
// The per-chunk kernels live in mp_mem_kernels.cu and the layer-wise ones in
// mp_mem_kernels_layerwise.cu; neither translation unit includes the other.
// What they genuinely share lives here: the device-side offset and copy
// helpers, the launch geometry, and the host-side driver that walks a batch
// plan. Everything is a template or marked inline, so including it from
// several translation units is ODR-safe.
//
// Included only by those two .cu files -- nothing here is part of a public
// header or the pybind surface.

#pragma once

#include <c10/cuda/CUDAGuard.h>

#include "kv_transfer_plan_types.h"
#include "mem_kernels.cuh"
#include "transfer_plan_types.cuh"

#include <algorithm>

namespace lmcache_mp {

/**
 * Key logic in the kernel implementation:
 * 1. Each thread block is for (BS, NH, HS) part (i.e., a single block in the
 * paged buffer)
 * 2. The thread block is 3D: threadIdx.x strides over the transfer units
 * within a head (at most 32 threads), threadIdx.y selects the head (one head
 * per y index, and threadIdx.z partitions the BS dimension (i.e., number of
 * tokens in the block).
 * 3. Within a thread block, we do loop over the BS dimension with a stride of
 * blockDim.z.
 * 4. The grid will take over (2, NB, NL) dimensions. No matter what the actual
 * layout in memory is, we will calculate the global offset for the start of the
 * block
 * 5. For LMCache, the layout is 2LTD [2, L, T, D] by default; when
 * kv_interleaved is set it becomes L2TD [L, 2, T, D] (per-layer interleaved)
 */

/**
 * Calculate the offset for the current block in the paged buffer
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_engine_global_offset(
    const int k_or_v, const int engine_block_idx, const int layer_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_block = shape_desc.scalars_per_block<ScalarType>();
  if constexpr (format == EngineKVFormat::NB_NL_TWO_BS_NH_HS) {
    // Cross-layer: single tensor [NB, NL, 2, BS, NH, HS]
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  } else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    // Normal: L tensors [2, NB, BS, NH, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS) {
    // Normal HND: L tensors [2, NB, NH, BS, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    // Flash Infer: L tensors [NB, 2, BS, NH, HS]
    return engine_block_idx * shape_desc.kv_size * scalars_per_block +
           k_or_v * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS) {
    // Flash Infer HND: L tensors [NB, 2, NH, BS, HS]
    return engine_block_idx * shape_desc.kv_size * scalars_per_block +
           k_or_v * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_HS ||
                       format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    // MLA: L tensors [NB, BS, HS]; blocked-scale shares the block base,
    // only its within-block layout differs (handled in the transfer body).
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA (in-process): 2L tensors [NBBS, NH, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS) {
    // SGLang MHA (MP daemon): 2L tensors [NB, BS, NH, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NBBS_ONE_HS) {
    // SGLang MLA: L tensors [NBBS, 1, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS) {
    // Fused-K/V HND: L tensors [NB, NH, BS, 2*HS], handled like
    // NL_X_TWO_NB_NH_BS_HS but with an empty K/V axis: the desc carries
    // kv_size == 1 and hs == 2 * head_size, so k_or_v is always 0 and each
    // head copy moves the packed K+V pair.
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_NH_TWO_HS) {
    // Fused-K/V NHD: L tensors [NB, BS, NH, 2*HS]; same empty K/V axis as
    // NL_X_NB_NH_BS_TWO_HS (kv_size == 1, hs == 2 * head_size), tokens
    // before heads within a block.
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_NH_BS_CS) {
    // Content-size HND: L tensors [NB, NH, BS, CS]; same empty K/V axis as
    // NL_X_NB_NH_BS_TWO_HS (kv_size == 1, hs == content_size).
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NL_X_NB_BS_NH_CS) {
    // Content-size NHD: L tensors [NB, BS, NH, CS]; same empty K/V axis as
    // NL_X_NB_NH_BS_CS, tokens before heads within a block.
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == EngineKVFormat::NB_NL_TWO_NH_BS_HS) {
    // TRT-LLM cross-layer HND: single tensor [NB, NL, 2, NH, BS, HS]
    // same block-level strides as NB_NL_TWO_BS_NH_HS
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  }
}

/**
 * Calculate the offset for the current token against the start
 * of the block in the paged buffer.
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_engine_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  if constexpr (format == EngineKVFormat::NB_NL_TWO_NH_BS_HS ||
                format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||
                format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS ||
                format == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS ||
                format == EngineKVFormat::NL_X_NB_NH_BS_CS) {
    // HND: [NH, BS, HS] — heads are outermost within a block
    size_t scalars_per_head_block =
        shape_desc.bs * scalars_per_head;  // BS * HS
    return head_idx * scalars_per_head_block + token_offset * scalars_per_head;
  } else {
    // NHD: [BS, NH, HS] — tokens are outermost within a block
    return head_idx * scalars_per_head + token_offset * scalars_per_token;
  }
}

/**
 * Calculate the global offset for the current `block` in the LMCache object.
 * The `block` here is the memory region corresponding to a thread-block.
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_lmcache_global_offset(
    const int k_or_v,
    const int
        token_offset_in_lmcache_object,  // 0~255 if LMCache chunk size is 256
    const int layer_idx,
    const int lmcache_chunk_size,  // e.g., 256
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  if (shape_desc.kv_interleaved) {
    // L2TD layout: [L, 2, T, D] — per-layer interleaved [K0,V0,K1,V1,...]
    return token_offset_in_lmcache_object * scalars_per_token +
           k_or_v * lmcache_chunk_size * scalars_per_token +
           layer_idx * shape_desc.kv_size * lmcache_chunk_size *
               scalars_per_token;
  }
  // 2LTD layout: [2, L, T, D] — K-then-V across all layers
  return token_offset_in_lmcache_object * scalars_per_token +
         layer_idx * lmcache_chunk_size * scalars_per_token +
         k_or_v * shape_desc.nl * lmcache_chunk_size * scalars_per_token;
}

/**
 * Calculate the local offset for the current token against the start of the
 * block in the LMCache object.
 */
template <typename ScalarType, EngineKVFormat format>
__device__ inline size_t calculate_lmcache_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

__device__ inline uint4 ld_cs(const uint4* addr) {
#if defined(__CUDA_ARCH__) && !defined(LMCACHE_DISABLE_STREAMING_IO)
  // Cache-streaming load: some compilers (e.g. MetaX's) don't support this
  // raw PTX asm; define LMCACHE_DISABLE_STREAMING_IO to fall through to the
  // plain deref below.
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
#if defined(__CUDA_ARCH__) && !defined(LMCACHE_DISABLE_STREAMING_IO)
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

template <typename ScalarType, bool lmcache_to_engine, EngineKVFormat format>
__device__ void multi_layer_block_transfer_single_block(
    ScalarType* __restrict__ lmcache_object,
    ScalarType** __restrict__ paged_buffer_ptrs, const int engine_block_idx,
    const int offset_in_lmcache_block, const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size  // e.g., 256, used to calculate global offset
                                  // in LMCache object
) {
  const int head_idx = threadIdx.y;
  const int init_token_offset = threadIdx.z;
  const int token_stride = blockDim.z;
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
  if constexpr (format == EngineKVFormat::NB_NL_TWO_BS_NH_HS ||
                format == EngineKVFormat::NB_NL_TWO_NH_BS_HS) {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[0];
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA (in-process): ptrs[0..NL-1] = K per layer, ptrs[NL..2NL-1] = V
    // per layer
    paged_buffer_layer_ptr =
        (ScalarType*)paged_buffer_ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else if constexpr (format == EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS) {
    // SGLang MHA (MP daemon): ptrs[0..NL-1] = K per layer, ptrs[NL..2NL-1] = V
    // per layer
    paged_buffer_layer_ptr =
        (ScalarType*)paged_buffer_ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[layer_idx];
  }

  if constexpr (format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    // Blocked page [BSxvals][BSxscales] vs token-major chunk row: two
    // copies per token (host pins <=4B units, so scale is whole units).
    const size_t spt = shape_desc.scalars_per_token<ScalarType>();
    const size_t scale_units = 4 / sizeof(ScalarType);
    const size_t val_units = spt - scale_units;
    for (int t = init_token_offset; t < shape_desc.bs; t += token_stride) {
      ScalarType* eng_vals =
          paged_buffer_layer_ptr + engine_global_offset + t * val_units;
      ScalarType* eng_scale = paged_buffer_layer_ptr + engine_global_offset +
                              shape_desc.bs * val_units + t * scale_units;
      ScalarType* lmc_row = lmcache_object + lmcache_global_offset + t * spt;
      if constexpr (lmcache_to_engine) {
        warp_copy<ScalarType>(eng_vals, lmc_row, val_units);
        warp_copy<ScalarType>(eng_scale, lmc_row + val_units, scale_units);
      } else {
        warp_copy<ScalarType>(lmc_row, eng_vals, val_units);
        warp_copy<ScalarType>(lmc_row + val_units, eng_scale, scale_units);
      }
    }
    return;
  }

  for (int token_offset = init_token_offset; token_offset < shape_desc.bs;
       token_offset += token_stride) {
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

/**
 * Launch geometry for the multi-layer block transfer kernels.
 *
 * Shared by the per-chunk launcher in mp_mem_kernels.cu and the layer-wise
 * launcher in mp_mem_kernels_layerwise.cu so the two can never drift apart:
 * retuning the grid/block math here retunes both paths at once.
 */
struct MultiLayerLaunchConfig {
  int total_blocks;
  int num_blocks_per_object;
  dim3 block;
  dim3 grid;
};

template <typename ScalarType>
inline MultiLayerLaunchConfig make_multi_layer_launch_config(
    const torch::Tensor& block_ids, int num_objects,
    const PageBufferShapeDesc& shape_desc, int lmcache_chunk_size) {
  MultiLayerLaunchConfig cfg;
  cfg.total_blocks = static_cast<int>(block_ids.size(0));
  TORCH_CHECK(cfg.total_blocks % num_objects == 0, "block_ids length (",
              cfg.total_blocks, ") must be divisible by num_objects (",
              num_objects, ")");
  cfg.num_blocks_per_object = cfg.total_blocks / num_objects;

  TORCH_CHECK(cfg.num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              cfg.num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");

  int elements_per_head = shape_desc.hs * shape_desc.element_size /
                          static_cast<int>(sizeof(ScalarType));
  int thread_dim_x = std::min(elements_per_head, 32);
  int thread_dim_y = shape_desc.nh;
  TORCH_CHECK(thread_dim_y <= 32, "Number of heads (", thread_dim_y,
              ") exceeds max threads per block in y-dim (32). This"
              " should never happen in normal LLMs");
  int thread_dim_z =
      std::min(shape_desc.bs, 1024 / (thread_dim_x * thread_dim_y));
  thread_dim_z = std::min(thread_dim_z, 64);  // max threads per block in z-dim

  cfg.block = dim3(thread_dim_x, thread_dim_y, thread_dim_z);
  cfg.grid = dim3(shape_desc.kv_size, cfg.total_blocks, shape_desc.nl);
  return cfg;
}

}  // namespace lmcache_mp

#define LAUNCH_KERNEL(KERNEL, DIRECTION, FORMAT, FIRST_ARG)               \
  KERNEL<ScalarType, DIRECTION, FORMAT><<<grid, block, 0, stream>>>(      \
      FIRST_ARG, paged_buffer_ptrs, block_ids_ptr, num_blocks_per_object, \
      shape_desc, lmcache_chunk_size, skip_prefix_n_blocks);              \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define DISPATCH_FORMAT(KERNEL, DIRECTION, FIRST_ARG)                          \
  switch (engine_kv_format) {                                                  \
    case EngineKVFormat::NB_NL_TWO_BS_NH_HS:                                   \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NB_NL_TWO_BS_NH_HS,     \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:                                 \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_TWO_NB_BS_NH_HS,   \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:                                 \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_TWO_NB_NH_BS_HS,   \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:                                 \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_TWO_BS_NH_HS,   \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:                                 \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_TWO_NH_BS_HS,   \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_BS_HS:                                        \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_BS_HS,          \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS:                                \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS,  \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS:                               \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS, \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NBBS_ONE_HS:                                     \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NBBS_ONE_HS,       \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NB_NL_TWO_NH_BS_HS:                                   \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NB_NL_TWO_NH_BS_HS,     \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_NH_BS_TWO_HS:                                 \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_NH_BS_TWO_HS,   \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_BS_NH_TWO_HS:                                 \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_BS_NH_TWO_HS,   \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_NH_BS_CS:                                     \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_NH_BS_CS,       \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_BS_NH_CS:                                     \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_BS_NH_CS,       \
                    FIRST_ARG);                                                \
      break;                                                                   \
    case EngineKVFormat::NL_X_NB_BSV_BSS:                                      \
      LAUNCH_KERNEL(KERNEL, DIRECTION, EngineKVFormat::NL_X_NB_BSV_BSS,        \
                    FIRST_ARG);                                                \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported EngineKVFormat: ",                       \
                  static_cast<int>(engine_kv_format));                         \
  }

/**
 * Transfer-unit (vectorisation width) selection, shared by both kernel entry
 * points.
 *
 * Picks the widest scalar type a head row can be copied in and invokes ``fn``
 * with a ``TransferUnit<T>`` tag carrying that choice, so the caller recovers
 * the type via ``typename decltype(tag)::type``. A tag is used rather than an
 * explicit template argument because C++17 has no templated lambdas.
 *
 * Centralised deliberately: this ladder used to be duplicated in both entry
 * points, where adding an EngineKVFormat or changing the vectorisation rule in
 * only one place left the other silently copying at the old width -- a wrong
 * answer at runtime rather than a build failure.
 */
template <typename T>
struct TransferUnit {
  using type = T;
};

template <typename Fn>
void dispatch_by_transfer_unit(const PageBufferShapeDesc& shape_desc,
                               EngineKVFormat engine_kv_format, const Fn& fn) {
  const int head_bytes = shape_desc.hs * shape_desc.element_size;
  TORCH_CHECK(head_bytes % sizeof(uint16_t) == 0, "head_size * element_size (",
              head_bytes, ") must be divisible by 2 for vectorized access");

  if (engine_kv_format == EngineKVFormat::NL_X_NB_BSV_BSS) {
    // Blocked-scale indexer cache: the per-token fp32 scale must be a whole
    // number of transfer units, so pin 4-byte units regardless of row width.
    TORCH_CHECK(head_bytes % sizeof(uint32_t) == 0,
                "NL_X_NB_BSV_BSS row bytes (", head_bytes,
                ") must be divisible by 4");
    fn(TransferUnit<uint32_t>{});
    return;
  }

  if (head_bytes % sizeof(uint4) == 0) {
    fn(TransferUnit<uint4>{});  // 16 bytes per copy
  } else if (head_bytes % sizeof(uint32_t) == 0) {
    fn(TransferUnit<uint32_t>{});  // 4 bytes per copy
  } else {
    fn(TransferUnit<uint16_t>{});  // 2 bytes per copy (minimum granularity)
  }
}

/**
 * Shared driver behind both object-group transfer entry points.
 *
 * Walks the batch plan, validates every launch, and materialises the
 * non-owning tensor views each kernel needs. ``launch_fn`` performs the kernel
 * dispatch -- the only thing that differs between the per-chunk executor in
 * mp_mem_kernels.cu and the layer-wise one in mp_mem_kernels_layerwise.cu --
 * and is invoked once per launch as
 * ``launch_fn(group, launch, paged_buffer_ptrs_tensor, block_ids)``. It is
 * taken by const reference, not forwarded, because it is re-invoked in a loop.
 *
 * Only those two translation units include this header, so nothing here
 * reaches a public header or the pybind surface.
 */
template <typename LaunchFn>
void execute_object_group_transfer_common(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps, const LaunchFn& launch_fn) {
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
      // views so we can reuse the existing kernel entry points without touching
      // any of their code. The backing storage is owned by the caller's tensors
      // (kept alive for the duration of this call); these views only carry the
      // pointer/shape each launch needs. Downstream only reads
      // paged_buffer_ptrs_tensor.data_ptr() and block_ids.{data_ptr, size(0)}.
      const uintptr_t block_ids_addr =
          group.block_ids_base +
          static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t);
      const at::Tensor paged_buffer_ptrs_tensor = at::from_blob(
          reinterpret_cast<void*>(group.paged_buffer_ptrs), {1}, int64_opts);
      const at::Tensor block_ids = at::from_blob(
          reinterpret_cast<void*>(block_ids_addr),
          {static_cast<int64_t>(launch.total_blocks)}, int64_opts);

      launch_fn(group, launch, paged_buffer_ptrs_tensor, block_ids);
    }
    if (!is_h2d) {
      do_staging(step.staging);
    }
  }
}

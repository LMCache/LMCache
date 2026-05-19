// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_transfer_kernel.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <type_traits>

#include <cuda_runtime_api.h>

namespace lmcache::mp {
namespace {

enum class NativeGPUKVFormat : int {
  kNbNlTwoBsNhHs = 0,
  kNlXTwoNbBsNhHs = 1,
  kNlXNbTwoBsNhHs = 2,
  kNlXNbBsHs = 3,
  kTwoXNlXNbbsNhHs = 4,
  kNlXNbbsOneHs = 5,
  kNlXTwoNbNhBsHs = 6,
  kNlXNbTwoNhBsHs = 7,
  kNbNlTwoNhBsHs = 8,
};

constexpr int kMaxPagedBufferPtrs = 256;
constexpr int kMaxBlockIds = 1024;

template <typename ScalarType>
struct NativeMemoryObj4 {
  ScalarType* objects[4];
  int num_objects;
};

template <typename ScalarType>
struct NativePagedBufferPtrs {
  ScalarType* ptrs[kMaxPagedBufferPtrs];
  int num_ptrs;
};

struct NativeBlockIds {
  std::int64_t ids[kMaxBlockIds];
  int num_ids;
};

std::string CudaErrorString(cudaError_t err, const char* op) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  return out.str();
}

bool Check(cudaError_t err, const char* op, std::string* error) {
  if (err == cudaSuccess) {
    return true;
  }
  *error = CudaErrorString(err, op);
  return false;
}

template <typename ScalarType>
__host__ __device__ inline std::size_t ScalarsPerHead(
    const NativePageBufferShapeDesc& shape_desc) {
  return static_cast<std::size_t>(shape_desc.hs) * shape_desc.element_size /
         sizeof(ScalarType);
}

template <typename ScalarType>
__host__ __device__ inline std::size_t ScalarsPerToken(
    const NativePageBufferShapeDesc& shape_desc) {
  return static_cast<std::size_t>(shape_desc.nh) * shape_desc.hs *
         shape_desc.element_size / sizeof(ScalarType);
}

template <typename ScalarType>
__host__ __device__ inline std::size_t ScalarsPerBlock(
    const NativePageBufferShapeDesc& shape_desc) {
  const std::size_t elems =
      shape_desc.block_stride_elems > 0
          ? static_cast<std::size_t>(shape_desc.block_stride_elems)
          : static_cast<std::size_t>(shape_desc.bs) * shape_desc.nh *
                shape_desc.hs;
  return elems * shape_desc.element_size / sizeof(ScalarType);
}

template <typename ScalarType, NativeGPUKVFormat format>
__device__ inline std::size_t EngineGlobalOffset(
    int k_or_v, int engine_block_idx, int layer_idx,
    const NativePageBufferShapeDesc& shape_desc) {
  const std::size_t scalars_per_block = ScalarsPerBlock<ScalarType>(shape_desc);
  if constexpr (format == NativeGPUKVFormat::kNbNlTwoBsNhHs) {
    return static_cast<std::size_t>(k_or_v) * scalars_per_block +
           static_cast<std::size_t>(layer_idx) * shape_desc.kv_size *
               scalars_per_block +
           static_cast<std::size_t>(engine_block_idx) * shape_desc.kv_size *
               scalars_per_block * shape_desc.nl;
  } else if constexpr (format == NativeGPUKVFormat::kNlXTwoNbBsNhHs ||
                       format == NativeGPUKVFormat::kNlXTwoNbNhBsHs) {
    return static_cast<std::size_t>(engine_block_idx) * scalars_per_block +
           static_cast<std::size_t>(k_or_v) * shape_desc.nb *
               scalars_per_block;
  } else if constexpr (format == NativeGPUKVFormat::kNlXNbTwoBsNhHs ||
                       format == NativeGPUKVFormat::kNlXNbTwoNhBsHs) {
    return static_cast<std::size_t>(engine_block_idx) * shape_desc.kv_size *
               scalars_per_block +
           static_cast<std::size_t>(k_or_v) * scalars_per_block;
  } else if constexpr (format == NativeGPUKVFormat::kNlXNbBsHs) {
    return static_cast<std::size_t>(engine_block_idx) * scalars_per_block;
  } else if constexpr (format == NativeGPUKVFormat::kTwoXNlXNbbsNhHs) {
    return static_cast<std::size_t>(engine_block_idx) * scalars_per_block;
  } else if constexpr (format == NativeGPUKVFormat::kNlXNbbsOneHs) {
    return static_cast<std::size_t>(engine_block_idx) * scalars_per_block;
  } else if constexpr (format == NativeGPUKVFormat::kNbNlTwoNhBsHs) {
    return static_cast<std::size_t>(k_or_v) * scalars_per_block +
           static_cast<std::size_t>(layer_idx) * shape_desc.kv_size *
               scalars_per_block +
           static_cast<std::size_t>(engine_block_idx) * shape_desc.kv_size *
               scalars_per_block * shape_desc.nl;
  }
}

template <typename ScalarType, NativeGPUKVFormat format>
__device__ inline std::size_t EngineLocalOffset(
    int token_offset, int head_idx,
    const NativePageBufferShapeDesc& shape_desc) {
  const std::size_t scalars_per_head =
      ScalarsPerHead<ScalarType>(shape_desc);
  const std::size_t scalars_per_token =
      ScalarsPerToken<ScalarType>(shape_desc);
  if constexpr (format == NativeGPUKVFormat::kNbNlTwoNhBsHs ||
                format == NativeGPUKVFormat::kNlXTwoNbNhBsHs ||
                format == NativeGPUKVFormat::kNlXNbTwoNhBsHs) {
    const std::size_t scalars_per_head_block =
        static_cast<std::size_t>(shape_desc.bs) * scalars_per_head;
    return static_cast<std::size_t>(head_idx) * scalars_per_head_block +
           static_cast<std::size_t>(token_offset) * scalars_per_head;
  } else {
    return static_cast<std::size_t>(head_idx) * scalars_per_head +
           static_cast<std::size_t>(token_offset) * scalars_per_token;
  }
}

template <typename ScalarType>
__device__ inline std::size_t LmcacheGlobalOffset(
    int k_or_v, int token_offset_in_lmcache_object, int layer_idx,
    int lmcache_chunk_size, const NativePageBufferShapeDesc& shape_desc) {
  const std::size_t scalars_per_token =
      ScalarsPerToken<ScalarType>(shape_desc);
  return static_cast<std::size_t>(token_offset_in_lmcache_object) *
             scalars_per_token +
         static_cast<std::size_t>(layer_idx) * lmcache_chunk_size *
             scalars_per_token +
         static_cast<std::size_t>(k_or_v) * shape_desc.nl *
             lmcache_chunk_size * scalars_per_token;
}

template <typename ScalarType>
__device__ inline std::size_t LmcacheLocalOffset(
    int token_offset, int head_idx,
    const NativePageBufferShapeDesc& shape_desc) {
  const std::size_t scalars_per_head =
      ScalarsPerHead<ScalarType>(shape_desc);
  const std::size_t scalars_per_token =
      ScalarsPerToken<ScalarType>(shape_desc);
  return static_cast<std::size_t>(head_idx) * scalars_per_head +
         static_cast<std::size_t>(token_offset) * scalars_per_token;
}

__device__ inline uint4 LdCs(const uint4* addr) {
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

__device__ inline void StCs(uint4* addr, uint4 val) {
#ifdef __CUDA_ARCH__
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
  *addr = val;
#endif
}

template <typename ScalarType>
__device__ inline void WarpCopy(ScalarType* __restrict__ dst,
                                const ScalarType* __restrict__ src,
                                std::size_t num_elements) {
  const int idx = threadIdx.x;
  const int stride = blockDim.x;
  if constexpr (std::is_same_v<ScalarType, uint4>) {
    for (std::size_t i = idx; i < num_elements; i += stride) {
      StCs(dst + i, LdCs(src + i));
    }
  } else {
    for (std::size_t i = idx; i < num_elements; i += stride) {
      dst[i] = src[i];
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine,
          NativeGPUKVFormat format>
__device__ void TransferSingleBlock(
    ScalarType* __restrict__ lmcache_object,
    const NativePagedBufferPtrs<ScalarType>& paged_buffer_ptrs,
    int engine_block_idx, int offset_in_lmcache_block,
    const NativePageBufferShapeDesc& shape_desc, int lmcache_chunk_size) {
  const int head_idx = threadIdx.y;
  const int k_or_v = blockIdx.x;
  const int layer_idx = blockIdx.z;

  const std::size_t engine_global_offset =
      EngineGlobalOffset<ScalarType, format>(k_or_v, engine_block_idx,
                                             layer_idx, shape_desc);
  const std::size_t lmcache_global_offset =
      LmcacheGlobalOffset<ScalarType>(k_or_v, offset_in_lmcache_block,
                                      layer_idx, lmcache_chunk_size,
                                      shape_desc);
  ScalarType* paged_buffer_layer_ptr = nullptr;
  if constexpr (format == NativeGPUKVFormat::kNbNlTwoBsNhHs ||
                format == NativeGPUKVFormat::kNbNlTwoNhBsHs) {
    paged_buffer_layer_ptr = paged_buffer_ptrs.ptrs[0];
  } else if constexpr (format == NativeGPUKVFormat::kTwoXNlXNbbsNhHs) {
    paged_buffer_layer_ptr =
        paged_buffer_ptrs.ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else {
    paged_buffer_layer_ptr = paged_buffer_ptrs.ptrs[layer_idx];
  }

  for (int token_offset = 0; token_offset < shape_desc.bs; ++token_offset) {
    const std::size_t engine_local_offset =
        EngineLocalOffset<ScalarType, format>(token_offset, head_idx,
                                              shape_desc);
    const std::size_t lmcache_local_offset =
        LmcacheLocalOffset<ScalarType>(token_offset, head_idx, shape_desc);
    ScalarType* engine_ptr =
        paged_buffer_layer_ptr + engine_global_offset + engine_local_offset;
    ScalarType* lmcache_ptr =
        lmcache_object + lmcache_global_offset + lmcache_local_offset;
    if constexpr (lmcache_to_engine) {
      WarpCopy<ScalarType>(engine_ptr, lmcache_ptr,
                           ScalarsPerHead<ScalarType>(shape_desc));
    } else {
      WarpCopy<ScalarType>(lmcache_ptr, engine_ptr,
                           ScalarsPerHead<ScalarType>(shape_desc));
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine,
          NativeGPUKVFormat format>
__global__ void TransferKernel(NativeMemoryObj4<ScalarType> lmcache_objects,
                               NativePagedBufferPtrs<ScalarType>
                                   paged_buffer_ptrs,
                               NativeBlockIds engine_block_ids,
                               int num_blocks_per_object,
                               NativePageBufferShapeDesc shape_desc,
                               int lmcache_chunk_size,
                               int skip_prefix_n_blocks) {
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;
  const int engine_block_idx =
      static_cast<int>(engine_block_ids.ids[flat_block_idx]);
  TransferSingleBlock<ScalarType, lmcache_to_engine, format>(
      lmcache_objects.objects[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs, shape_desc, lmcache_chunk_size);
}

template <typename ScalarType, bool lmcache_to_engine,
          NativeGPUKVFormat format>
bool LaunchKernel(const std::vector<void*>& paged_buffer_ptrs,
                  const std::vector<void*>& lmcache_object_ptrs,
                  const std::vector<std::int64_t>& block_ids,
                  NativePageBufferShapeDesc shape_desc,
                  int lmcache_chunk_size, int skip_prefix_n_blocks,
                  cudaStream_t stream, std::string* error) {
  NativeMemoryObj4<ScalarType> objects{};
  objects.num_objects = static_cast<int>(lmcache_object_ptrs.size());
  for (int i = 0; i < 4; ++i) {
    objects.objects[i] =
        i < objects.num_objects
            ? reinterpret_cast<ScalarType*>(lmcache_object_ptrs[i])
            : nullptr;
  }

  if (paged_buffer_ptrs.size() > kMaxPagedBufferPtrs) {
    *error = "native CUDA kernel received too many paged-buffer pointers";
    return false;
  }
  if (block_ids.size() > kMaxBlockIds) {
    *error = "native CUDA kernel received too many block ids";
    return false;
  }

  NativePagedBufferPtrs<ScalarType> paged_ptrs{};
  paged_ptrs.num_ptrs = static_cast<int>(paged_buffer_ptrs.size());
  for (int i = 0; i < paged_ptrs.num_ptrs; ++i) {
    paged_ptrs.ptrs[i] = reinterpret_cast<ScalarType*>(paged_buffer_ptrs[i]);
  }

  NativeBlockIds block_id_list{};
  block_id_list.num_ids = static_cast<int>(block_ids.size());
  for (int i = 0; i < block_id_list.num_ids; ++i) {
    block_id_list.ids[i] = block_ids[i];
  }

  const int total_blocks = static_cast<int>(block_ids.size());
  const int num_blocks_per_object =
      total_blocks / static_cast<int>(lmcache_object_ptrs.size());
  const int elements_per_head =
      shape_desc.hs * shape_desc.element_size /
      static_cast<int>(sizeof(ScalarType));
  const int thread_dim_x = std::min(elements_per_head, 32);
  dim3 block(thread_dim_x, shape_desc.nh);
  dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

  TransferKernel<ScalarType, lmcache_to_engine, format>
      <<<grid, block, 0, stream>>>(objects, paged_ptrs,
                                   block_id_list, num_blocks_per_object,
                                   shape_desc, lmcache_chunk_size,
                                   skip_prefix_n_blocks);
  return true;
}

#define LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION, FORMAT)                      \
  do {                                                                        \
    if (use_uint4) {                                                          \
      if (!LaunchKernel<uint4, DIRECTION, FORMAT>(                            \
          paged_buffer_ptrs, lmcache_object_ptrs, block_ids, shape_desc,      \
          lmcache_chunk_size, skip_prefix_n_blocks, stream, error)) {         \
        return false;                                                         \
      }                                                                       \
    } else if (use_uint32) {                                                  \
      if (!LaunchKernel<std::uint32_t, DIRECTION, FORMAT>(                    \
          paged_buffer_ptrs, lmcache_object_ptrs, block_ids, shape_desc,      \
          lmcache_chunk_size, skip_prefix_n_blocks, stream, error)) {         \
        return false;                                                         \
      }                                                                       \
    } else {                                                                  \
      if (!LaunchKernel<std::uint16_t, DIRECTION, FORMAT>(                    \
          paged_buffer_ptrs, lmcache_object_ptrs, block_ids, shape_desc,      \
          lmcache_chunk_size, skip_prefix_n_blocks, stream, error)) {         \
        return false;                                                         \
      }                                                                       \
    }                                                                         \
  } while (0)

#define LMCACHE_NATIVE_DISPATCH_DIRECTION(DIRECTION)                          \
  switch (static_cast<NativeGPUKVFormat>(gpu_kv_format)) {                    \
    case NativeGPUKVFormat::kNbNlTwoBsNhHs:                                   \
      LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION,                                 \
                                   NativeGPUKVFormat::kNbNlTwoBsNhHs);        \
      break;                                                                  \
    case NativeGPUKVFormat::kNlXTwoNbBsNhHs:                                  \
      LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION,                                 \
                                   NativeGPUKVFormat::kNlXTwoNbBsNhHs);       \
      break;                                                                  \
    case NativeGPUKVFormat::kNlXNbTwoBsNhHs:                                  \
      LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION,                                 \
                                   NativeGPUKVFormat::kNlXNbTwoBsNhHs);       \
      break;                                                                  \
    case NativeGPUKVFormat::kNlXNbBsHs:                                       \
      LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION, NativeGPUKVFormat::kNlXNbBsHs); \
      break;                                                                  \
    case NativeGPUKVFormat::kNlXTwoNbNhBsHs:                                  \
      LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION,                                 \
                                   NativeGPUKVFormat::kNlXTwoNbNhBsHs);       \
      break;                                                                  \
    case NativeGPUKVFormat::kNlXNbTwoNhBsHs:                                  \
      LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION,                                 \
                                   NativeGPUKVFormat::kNlXNbTwoNhBsHs);       \
      break;                                                                  \
    case NativeGPUKVFormat::kNbNlTwoNhBsHs:                                   \
      LMCACHE_NATIVE_LAUNCH_FORMAT(DIRECTION,                                 \
                                   NativeGPUKVFormat::kNbNlTwoNhBsHs);        \
      break;                                                                  \
    default:                                                                  \
      *error = "native CUDA kernel does not support GPU KV format " +         \
               std::to_string(gpu_kv_format);                                 \
      return false;                                                           \
  }

}  // namespace

bool NativeCudaBlockTransfer(const std::vector<void*>& paged_buffer_ptrs,
                             const std::vector<void*>& lmcache_object_ptrs,
                             const std::vector<std::int64_t>& block_ids,
                             bool lmcache_to_engine,
                             NativePageBufferShapeDesc shape_desc,
                             int lmcache_chunk_size, int gpu_kv_format,
                             int skip_prefix_n_blocks, std::string* error) {
  return NativeCudaBlockTransferWithStream(
      paged_buffer_ptrs, lmcache_object_ptrs, block_ids, lmcache_to_engine,
      shape_desc, lmcache_chunk_size, gpu_kv_format, skip_prefix_n_blocks,
      nullptr, true, error);
}

bool NativeCudaBlockTransferWithStream(
    const std::vector<void*>& paged_buffer_ptrs,
    const std::vector<void*>& lmcache_object_ptrs,
    const std::vector<std::int64_t>& block_ids, bool lmcache_to_engine,
    NativePageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    int gpu_kv_format, int skip_prefix_n_blocks, void* cuda_stream,
    bool synchronize, std::string* error) {
  if (paged_buffer_ptrs.empty() || lmcache_object_ptrs.empty() ||
      lmcache_object_ptrs.size() > 4 || block_ids.empty()) {
    *error = "native CUDA kernel received an invalid empty transfer";
    return false;
  }
  if (block_ids.size() % lmcache_object_ptrs.size() != 0) {
    *error = "native CUDA kernel block ids are not divisible by object count";
    return false;
  }
  if (shape_desc.kv_size <= 0 || shape_desc.nl <= 0 || shape_desc.nb <= 0 ||
      shape_desc.bs <= 0 || shape_desc.nh <= 0 || shape_desc.hs <= 0 ||
      shape_desc.element_size <= 0 || lmcache_chunk_size <= 0) {
    *error = "native CUDA kernel received invalid shape metadata";
    return false;
  }
  if (shape_desc.nh > 32) {
    *error = "native CUDA kernel supports at most 32 KV heads per block";
    return false;
  }
  const int head_bytes = shape_desc.hs * shape_desc.element_size;
  if (head_bytes % static_cast<int>(sizeof(std::uint16_t)) != 0) {
    *error =
        "native CUDA kernel requires head_size * element_size divisible by 2";
    return false;
  }

  const bool use_uint4 = head_bytes % static_cast<int>(sizeof(uint4)) == 0;
  const bool use_uint32 =
      !use_uint4 && head_bytes % static_cast<int>(sizeof(std::uint32_t)) == 0;
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (lmcache_to_engine) {
    LMCACHE_NATIVE_DISPATCH_DIRECTION(true);
  } else {
    LMCACHE_NATIVE_DISPATCH_DIRECTION(false);
  }

  if (!Check(cudaGetLastError(), "native CUDA block transfer launch", error)) {
    return false;
  }
  if (!synchronize) {
    return true;
  }
  return Check(cudaStreamSynchronize(stream), "native CUDA block transfer",
               error);
}

#undef LMCACHE_NATIVE_DISPATCH_DIRECTION
#undef LMCACHE_NATIVE_LAUNCH_FORMAT

}  // namespace lmcache::mp

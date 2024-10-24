#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "mem_kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

namespace lmc {
enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

// TODO: Might need to specia adaptation fp8 (convert back to fp8)
// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/
// group__CUDA__MATH__FP8__MISC.html
// https://pytorch.org/docs/stable/tensors.html
// see torch.float8_e4m3fn, torch.float8_e5m2
// both data types only have limited support in pytorch
template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
  return x;
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void load_and_reshape_flash_kernel(
    scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    const cache_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    const cache_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride, const int key_stride, const int value_stride,
    const int num_heads, const int head_size, const int block_size,
    const float k_scale, const float v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx = token_idx * key_stride + i;
    const int64_t tgt_value_idx = token_idx * value_stride + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int64_t src_key_value_idx = block_idx * block_stride +
                                      block_offset * num_heads * head_size +
                                      head_idx * head_size + head_offset;
    
    cache_t tgt_key = key_cache[src_key_value_idx];
    cache_t tgt_value = value_cache[src_key_value_idx];

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key[tgt_key_idx] = tgt_key;
      value[tgt_value_idx] = tgt_value;
    } else {
      // TODO: Need to convert data type back to fp8
      assert(false); 
      //key[tgt_key_idx] =
      //    scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, k_scale);
      //value[tgt_value_idx] =
      //    scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, v_scale);
    }
  }
}

} // namespace lmc

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)         \
  lmc::load_and_reshape_flash_kernel<KV_T, CACHE_T, KV_DTYPE>       \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), block_stride, key_stride, \
          value_stride, num_heads, head_size, block_size, k_scale, v_scale);

std::tuple<torch::Tensor, torch::Tensor> load_and_reshape_flash(
    //torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    //torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, const double k_scale,
    const double v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads= key_cache.size(2);
  int head_size = key_cache.size(3);
  //int num_tokens = key.size(0);
  //int num_heads = key.size(1);
  //int head_size = key.size(2);
  int block_size = key_cache.size(1);
  
  torch::Tensor key = torch::empty({num_tokens, num_heads, head_size}, \
    torch::TensorOptions().dtype(key_cache.dtype()).device(torch::kCUDA));
  torch::Tensor value = torch::empty({num_tokens, num_heads, head_size}, \
    torch::TensorOptions().dtype(value_cache.dtype()).device(torch::kCUDA));

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int block_stride = key_cache.stride(0);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
  return std::make_tuple(key, value);
}

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "mem_kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>

namespace lmc {

template <typename scalar_t>
__global__ void load_and_reshape_flash_kernel(
    scalar_t* __restrict__ key_value,  // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    const scalar_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx = key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx = value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx = block_idx * block_stride_in_64bit +
                                      block_offset * num_heads * head_size_in_64bit +
                                      head_idx * head_size_in_64bit + head_offset;
    
    scalar_t tgt_key = key_cache[src_key_value_idx];
    scalar_t tgt_value = value_cache[src_key_value_idx];

    key_value[tgt_key_idx] = tgt_key;
    key_value[tgt_value_idx] = tgt_value;
  }
}

template <typename scalar_t>
__global__ void reshape_and_cache_back_flash_kernel(
    const scalar_t* __restrict__ key_value,    // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    scalar_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx = key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx = value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx = block_idx * block_stride_in_64bit +
                                      block_offset * num_heads * head_size_in_64bit +
                                      head_idx * head_size_in_64bit + head_offset;
    
    scalar_t tgt_key = key_value[tgt_key_idx];
    scalar_t tgt_value = key_value[tgt_value_idx];

    key_cache[src_key_value_idx] = tgt_key;
    value_cache[src_key_value_idx] = tgt_value;
  }
}

} // namespace lmc



template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu() && tensor.is_pinned()) {
    T* ptr;
    cudaHostGetDevicePointer((void**)&ptr,
                             static_cast<void*>(tensor.data_ptr()), 0);
    return ptr;
  } else if (device.is_cpu()) {
    //return NULL;
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}


void load_and_reshape_flash(
    torch::Tensor& key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                               // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens],
    const int layer_idx
    ) {
  

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
  
  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads= key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;


  int block_size = key_cache.size(1);
  
  
  int key_value_stride = key_value.stride(2) / elements_per_entry;
  
  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset = (layer_idx+num_layers) * key_value.stride(1)
            / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::load_and_reshape_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit, block_size,
      key_layer_offset, value_layer_offset);
}


void reshape_and_cache_back_flash(
    torch::Tensor& key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                               // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const int layer_idx
    ) 
{  

  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
  
  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads= key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;


  int block_size = key_cache.size(1);
  
  int key_value_stride = key_value.stride(2) / elements_per_entry;
  
  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset = (layer_idx+num_layers) * key_value.stride(1)
            / elements_per_entry;


  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::reshape_and_cache_back_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit, block_size,
      key_layer_offset, value_layer_offset);
}
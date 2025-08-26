// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include "mem_kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#ifdef USE_ROCM
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

namespace lmc {

template <typename scalar_t>
__global__ void load_and_reshape_flash_kernel(
    scalar_t* __restrict__ key_value,  // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ key_cache,    // [num_blocks, block_size,
                                               // num_heads, head_size]
    const scalar_t* __restrict__ value_cache,  // [num_blocks, block_size,
                                               // num_heads, head_size]
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
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
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
    const scalar_t* __restrict__ key_value,  // [num_tokens, num_heads,
                                             // head_size]
    scalar_t* __restrict__ key_cache,    // [num_blocks, block_size, num_heads,
                                         // head_size]
    scalar_t* __restrict__ value_cache,  // [num_blocks, block_size, num_heads,
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
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    scalar_t tgt_key = key_value[tgt_key_idx];
    scalar_t tgt_value = key_value[tgt_value_idx];

    key_cache[src_key_value_idx] = tgt_key;
    value_cache[src_key_value_idx] = tgt_value;
  }
}

template <typename scalar_t>
__global__ void single_layer_kv_transfer_kernel(
    // scalar_t* __restrict__ lmc_key_cache,    // [num_tokens,
    // num_heads*head_size] scalar_t* __restrict__ lmc_value_cache,  //
    // [num_tokens, num_heads*head_size]
    scalar_t* __restrict__ lmc_key_value_cache,   // [num_tokens, 2,
                                                  // num_heads*head_size]
                                                  // or
                                                  // [2, num_tokens,
                                                  // num_heads*head_size]
    scalar_t* __restrict__ vllm_key_value_cache,  // [2, num_blocks, block_size,
                                                  // num_heads, head_size] or
                                                  // [num_blocks, 2, block_size,
                                                  // num_heads, head_size]

    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int vllm_block_key_stride_in_64bit, const int vllm_value_offset,
    const int lmc_stride, const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size, const bool direction) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;
    const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                                 block_offset * num_heads * head_size_in_64bit +
                                 head_idx * head_size_in_64bit + head_offset;
    const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
    if (direction) {
      lmc_key_value_cache[lmc_key_idx] = vllm_key_value_cache[vllm_key_idx];
      lmc_key_value_cache[lmc_value_idx] = vllm_key_value_cache[vllm_value_idx];
    } else {
      vllm_key_value_cache[vllm_key_idx] = lmc_key_value_cache[lmc_key_idx];
      vllm_key_value_cache[vllm_value_idx] = lmc_key_value_cache[lmc_value_idx];
    }
  }
}

__device__ __forceinline__ int64_t page_buffer_offset(
    const int k_or_v, const int token_idx, const int scalar_offset,
    const int scalars_per_token, const int page_buffer_size) {
  return k_or_v * page_buffer_size * scalars_per_token +
         token_idx * scalars_per_token + scalar_offset;
}

__device__ __forceinline__ int64_t page_buffer_offset_unilateral(
    const int token_idx, const int scalar_offset, const int scalars_per_token) {
  return token_idx * scalars_per_token + scalar_offset;
}

__device__ __forceinline__ int64_t
key_value_offset(const int k_or_v, const int layer_idx, const int token_idx,
                 const int scalar_offset, const int scalars_per_token,
                 const int num_tokens, const int num_layers) {
  return k_or_v * num_layers * num_tokens * scalars_per_token +
         layer_idx * num_tokens * scalars_per_token +
         token_idx * scalars_per_token + scalar_offset;
}

/**
 * Quickly load KV cache between vLLM paged memory and offloading buffer
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] <=> ptrs[block.y][block.z,
 * slot_id, thread.x]
 */
template <typename scalar_t, bool DIRECTION>
__global__ void load_and_reshape_multi_layer_kernel(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers] * [2,
                                                // PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t slot_idx = slot_mapping[token_id];
  int64_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t vllm_offset = page_buffer_offset(
        k_or_v, slot_idx, i, scalars_per_token, page_buffer_size);

    if (DIRECTION)  // 1 is paged buffer to LMCache
      key_value[lmcache_offset] = paged_buffer_ptr[vllm_offset];
    else  // 0 is LMCache to paged buffer
      paged_buffer_ptr[vllm_offset] = key_value[lmcache_offset];
  }
}

/*
 * handle sglang MHA offload between CPU and GPU
 */
template <typename scalar_t, bool DIRECTION>
__global__ void load_and_reshape_multi_layer_kernel_unilateral(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers *2] *
                                                // [PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t slot_idx = slot_mapping[token_id];
  int64_t* key_ptr = paged_buffer_ptrs[layer_id];
  int64_t* value_ptr = paged_buffer_ptrs[layer_id + num_layers];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t sgl_offset =
        page_buffer_offset_unilateral(slot_idx, i, scalars_per_token);

    if (k_or_v == 0) {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = key_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        key_ptr[sgl_offset] = key_value[lmcache_offset];
    } else {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = value_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        value_ptr[sgl_offset] = key_value[lmcache_offset];
    }
  }
}

}  // namespace lmc

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: false  means LMCache to PagedBuffer, true  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer(
    torch::Tensor&
        key_value,  // key/value must be on gpu/pinned cpu.
                    // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn.
                    // [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA.

    const torch::Tensor& key_value_ptrs,  // [num_layers]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla) {
  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  int64_t** page_buffer_ptrs =
      get_kernel_ptr<int64_t*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int num_origin_elements = key_value.size(3);
  int elements_per_qword = 8 / key_value.element_size();
  int num_qwords = num_origin_elements / elements_per_qword;

  int k_or_v_size = 2;
  if (use_mla) {
    k_or_v_size = 1;
  }

  dim3 grid(key_value.size(2), num_layers, k_or_v_size);
  dim3 block(std::min(num_qwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (not direction) {
    lmc::load_and_reshape_multi_layer_kernel<int64_t, false>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    lmc::load_and_reshape_multi_layer_kernel<int64_t, true>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

/**
 * Quickly offload KV cache from SGLang paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in SGLang's K/V buffer has a shape of
 * [PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: false  means LMCache to PagedBuffer, true  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer_unilateral(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA key/value must be on gpu/pinned cpu

    const torch::Tensor& key_value_ptrs,  // [num_layers*2]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla) {
  if (use_mla) {
    return multi_layer_kv_transfer(key_value, key_value_ptrs, slot_mapping,
                                   paged_memory_device, page_buffer_size,
                                   direction, use_mla);
  }

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  int64_t** page_buffer_ptrs =
      get_kernel_ptr<int64_t*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int num_origin_elements = key_value.size(3);
  int elements_per_qword = 8 / key_value.element_size();
  int num_qwords = num_origin_elements / elements_per_qword;

  int k_or_v_size = 2;

  dim3 grid(key_value.size(2), key_value.size(1), k_or_v_size);
  dim3 block(std::min(num_qwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (not direction) {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, false>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, true>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void single_layer_kv_transfer(
    // torch::Tensor& lmc_key_cache,  // [num_tokens, num_heads*head_size]
    //  key/value must be on gpu/pinned cpu
    // torch::Tensor& lmc_value_cache,  // [num_tokens, num_heads*head_size]

    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]

    // torch::Tensor&
    //     vllm_key_cache,  // [num_blocks, block_size, num_heads, head_size]
    // torch::Tensor&
    //     vllm_value_cache,  // [num_blocks, block_size, num_heads, head_size]
    //  key_cache/value_cache must be on gpu
    torch::Tensor&
        vllm_key_value_cache,  // [2, num_blocks, block_size, num_heads,
                               // head_size] for flash attention
    // [num_blocks, 2, block_size, num_heads, head_size] for flash infer

    torch::Tensor& slot_mapping,  // [num_tokens]
    const bool direction,    // false: LMCache to PagedBuffer, true: PagedBuffer
                             // to LMCache
    const bool token_major,  // true: lmc_key_value_cache is
                             // [num_tokens, 2, num_heads*head_size]
                             // false: lmc_key_value_cache is
                             // [2, num_tokens, num_heads*head_size]
    const bool vllm_two_major  // true: vllm_key_value_cache is
                               // [2, num_blocks, block_size, num_heads,
                               // head_size]
                               // false: vllm_key_value_cache is
                               // [num_blocks, 2, block_size, num_heads,
                               // head_size]
) {
  // int64_t* lmc_key_cache_ptr = get_kernel_ptr<int64_t,
  // torch::Tensor>(lmc_key_cache); int64_t* lmc_value_cache_ptr =
  // get_kernel_ptr<int64_t, torch::Tensor>(lmc_value_cache);
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);
  // int64_t* vllm_value_cache_ptr =
  //     get_kernel_ptr<int64_t, torch::Tensor>(vllm_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = vllm_key_value_cache.size(3);
  int head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;

  int block_size = vllm_key_value_cache.size(2);

  int lmc_stride;
  int lmc_value_offset;
  if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (vllm_two_major) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  }

  // int block_stride_in_64bit = vllm_key_cache.stride(0) / elements_per_entry;
  // TORCH_CHECK(vllm_key_cache.stride(0) == vllm_value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(vllm_key_value_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::single_layer_kv_transfer_kernel<int64_t><<<grid, block, 0, stream>>>(
      lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr,
      vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,
      lmc_value_offset, num_heads, head_size_in_64bit, block_size, direction);
}

void load_and_reshape_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens],
    const int layer_idx) {
  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::load_and_reshape_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

void reshape_and_cache_back_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const int layer_idx) {
  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::reshape_and_cache_back_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

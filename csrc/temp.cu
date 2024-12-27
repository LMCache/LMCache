#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>
#include <cstdio>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

namespace vllm {

template <typename T>
__global__ void paged_copy(T* __restrict__ dst, const T* __restrict__ src,
                           const int64_t* src_to_dst, const int num_pages,
                           const int num_elements_per_page) {
  const int64_t srcPageIdx = src_to_dst[blockIdx.x << 1];
  const int64_t dstPageIdx = src_to_dst[(blockIdx.x << 1) | 1];

  const int64_t srcPageOffset = srcPageIdx * num_elements_per_page;
  const int64_t dstPageOffset = dstPageIdx * num_elements_per_page;

  for (int i = threadIdx.x; i < num_elements_per_page; i += blockDim.x) {
    dst[dstPageOffset + i] = src[srcPageOffset + i];
  }
}

}  // namespace vllm

template <int DTYPE_LEN, typename DTYPE>
void launch_swap_block_kernel(DTYPE* dst, const DTYPE* src,
                              const int64_t* block_mapping_ptr,
                              const int num_blocks,
                              const int block_size_in_bytes,
                              const torch::Device& device) {
  c10::cuda::CUDAGuard device_guard(device);

  int num_threads = 1024;
  int grid_size = num_blocks;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::paged_copy<<<grid_size, num_threads, 0, stream>>>(
      dst, src, block_mapping_ptr, num_blocks, block_size_in_bytes / DTYPE_LEN);
}

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
    return NULL;
  } else {
    TORCH_CHECK(false, "Invalid device");
  }
}

void swap_blocks_slow(torch::Tensor& src, torch::Tensor& dst,
                      const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
}

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  int64_t* src_ptr = get_kernel_ptr<int64_t, torch::Tensor>(src);
  int64_t* dst_ptr = get_kernel_ptr<int64_t, torch::Tensor>(dst);
  const int64_t* block_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(block_mapping);

  if (src_ptr == NULL || dst_ptr == NULL || block_mapping_ptr == NULL) {
    // fall back to the slow implementation
    swap_blocks_slow(src, dst, block_mapping);
  } else {
    // Check the device
    torch::Device src_device = src.device();
    torch::Device dst_device = dst.device();
    torch::Device block_mapping_device = block_mapping.device();
    if (src_device.is_cuda() && dst_device.is_cuda()) {
      TORCH_CHECK(src_device.index() == dst_device.index(),
                  "src and dst must be on the same GPU");
    }
    torch::Device cuda_device = src_device.is_cuda() ? src_device : dst_device;

    const int64_t num_blocks = block_mapping.size(0);
    const int64_t block_size_in_bytes = src.element_size() * src[0].numel();

    launch_swap_block_kernel<8, int64_t>(dst_ptr, (const int64_t*)src_ptr,
                                         block_mapping_ptr, num_blocks,
                                         block_size_in_bytes, cuda_device);
  }
}
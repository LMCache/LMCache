// SPDX-License-Identifier: Apache-2.0

#include "sdk_kernels.cuh"
#include "mp_mem_kernels.cuh"  // batch_memcpy_supported

#if defined(USE_ROCM)
  #include <hip/hip_version.h>
#endif

// cudaMemcpyBatchAsync exists from CUDA 12.8 and HIP 7.15 (same guard as
// mp_mem_kernels.cu).
#if !defined(USE_ROCM) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  #define LMC_HAS_BATCH_MEMCPY 1
#elif defined(USE_ROCM) && defined(HIP_VERSION) && HIP_VERSION >= 71500000
  #define LMC_HAS_BATCH_MEMCPY 1
#else
  #define LMC_HAS_BATCH_MEMCPY 0
#endif

// This function is for transfer between QRingBuffer and SDK's contiguous CPU
// buffer in lmcache/sdk/wrapper/ring_transfer.py.
void batched_memcpy_async(const std::vector<uintptr_t>& src_ptrs,
                          const std::vector<uintptr_t>& dst_ptrs,
                          const std::vector<size_t>& sizes,
                          const torch::Device& device) {
  TORCH_CHECK(batch_memcpy_supported(),
              "cudaMemcpyBatchAsync is not available: the extension must be "
              "built against CUDA >= 12.8 and run on a >= 12.8 runtime/driver");
  TORCH_CHECK(
      src_ptrs.size() == dst_ptrs.size() && src_ptrs.size() == sizes.size(),
      "batched_memcpy_async expects equally sized src_ptrs, dst_ptrs, "
      "and sizes");
#if LMC_HAS_BATCH_MEMCPY
  if (sizes.empty()) return;
  const at::cuda::OptionalCUDAGuard device_guard(device);
  // cudaMemcpyBatchAsync rejects the legacy NULL stream (see above).
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (stream == nullptr) {
    stream = cudaStreamPerThread;
  }
  std::vector<void*> srcs(src_ptrs.size());
  std::vector<void*> dsts(dst_ptrs.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    srcs[i] = reinterpret_cast<void*>(src_ptrs[i]);
    dsts[i] = reinterpret_cast<void*>(dst_ptrs[i]);
  }
  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attrs_idx = 0;
  #if CUDART_VERSION >= 13000
  const cudaError_t err =
      cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), sizes.size(),
                           &attrs, &attrs_idx, 1, stream);
  #else
  size_t fail_idx = 0;
  const cudaError_t err =
      cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), sizes.size(),
                           &attrs, &attrs_idx, 1, &fail_idx, stream);
  #endif
  TORCH_CHECK(err == cudaSuccess, "cudaMemcpyBatchAsync failed with ",
              cudaGetErrorString(err), " (", sizes.size(), " entries)");
#endif
}

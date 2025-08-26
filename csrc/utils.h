#include <string>

#ifdef USE_ROCM
  #include <c10/hip/HIPException.h>
#else
  #include <c10/cuda/CUDAException.h>

#endif

// kernel_launch_wrapper.h
#pragma once

std::string get_gpu_pci_bus_id(int device);

// CUDA version
#if !defined(USE_ROCM)
  #define LAUNCH_KERNEL_WITH_CHECK(kernel, grid, block, shmem, stream, \
                                   /*args*/...)                        \
    do {                                                               \
      kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__);     \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                  \
    } while (0);

  // Convenience: default shmem=0, stream=0
  #define LAUNCH_KERNEL_WITH_CHECK_DEFAULT(kernel, grid, block, /*args*/...) \
    LAUNCH_KERNEL_WITH_CHECK(kernel, grid, block, 0 /*shmem*/, 0 /*stream*/, \
                             __VA_ARGS__)

// ROCm version
#else
  #include <c10/hip/HIPException.h>
  #define LAUNCH_KERNEL_WITH_CHECK(kernel, grid, block, shmem, stream, \
                                   /*args*/...)                        \
    do {                                                               \
      kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__);     \
      C10_HIP_KERNEL_LAUNCH_CHECK();                                   \
    } while (0)

  #define LAUNCH_KERNEL_WITH_CHECK_DEFAULT(kernel, grid, block, /*args*/...) \
    LAUNCH_KERNEL_WITH_CHECK(kernel, grid, block, 0 /*shmem*/, 0 /*stream*/, \
                             __VA_ARGS__)
#endif

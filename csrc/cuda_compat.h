// SPDX-License-Identifier: Apache-2.0

/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/main/csrc/cuda_compat.h
 */

#pragma once
#if defined(USE_ROCM) || defined(USE_XPU)
  #define LMCACHE_LDG(arg) *(arg)
#else
  #define LMCACHE_LDG(arg) __ldg(arg)
#endif

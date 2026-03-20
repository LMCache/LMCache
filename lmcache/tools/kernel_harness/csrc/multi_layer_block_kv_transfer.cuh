// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

enum class GPUKVFormat : int {
  NB_NL_TWO_BS_NH_HS =
      0,  // vLLM cross-layer: single tensor [NB, NL, 2, BS, NH, HS]
  NL_X_TWO_NB_BS_NH_HS =
      1,  // vLLM flash attention: L tensors [2, NB, BS, NH, HS]
  NL_X_NB_TWO_BS_NH_HS = 2,   // vLLM flash infer: L tensors [NB, 2, BS, NH, HS]
  NL_X_NB_BS_HS = 3,          // vLLM MLA: L tensors [NB, BS, HS]
  TWO_X_NL_X_NBBS_NH_HS = 4,  // SGLang MHA: 2L tensors [NBBS, NH, HS]
  NL_X_NBBS_ONE_HS = 5,       // SGLang MLA: L tensors [NBBS, 1, HS]
};

struct PageBufferShapeDesc {
  int kv_size;       // 1 or 2
  int nl;            // num layers
  int nb;            // num blocks
  int bs;            // block size
  int nh;            // num heads
  int hs;            // head size
  int element_size;  // bytes (1 or 2)

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_block() const {
    return bs * nh * hs * element_size / sizeof(ScalarType);
  }
};

template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;  // 0 - 4
};

/**
 * Block-level multi-layer KV transfer between vLLM paged buffers and
 * LMCache contiguous memory objects.
 *
 * @param paged_buffer_ptrs_tensor  GPU int64 tensor of data pointers into
 *                                  vLLM paged buffers (one per tensor)
 * @param lmcache_objects_ptrs      Raw pointers to LMCache memory objects
 * @param block_ids                 Block indices in vLLM paged buffer
 * @param device                    CUDA device of vLLM tensors
 * @param direction                 H2D (LMCache->vLLM) or D2H (vLLM->LMCache)
 * @param shape_desc                Shape descriptor for the paged buffer
 * @param lmcache_chunk_size        Tokens per LMCache memory object
 * @param gpu_kv_format             GPUKVFormat identifier
 * @param skip_prefix_n_blocks      Number of blocks to skip at the beginning
 */
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format, int skip_prefix_n_blocks);

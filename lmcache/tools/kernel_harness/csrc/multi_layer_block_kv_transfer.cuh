// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
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
 * @param key_value_tensors  vLLM paged buffer tensors (per-layer or single
 * cross-layer)
 * @param memory_objects     LMCache memory objects (pinned CPU tensors)
 * @param block_ids          Block indices in vLLM paged buffer (pinned CPU
 * int64)
 * @param device             CUDA device of vLLM tensors
 * @param direction          H2D (LMCache->vLLM) or D2H (vLLM->LMCache)
 * @param gpu_kv_format      GPUKVFormat identifier
 * @param block_size         Block size (BS) for vLLM paged buffers
 * @param num_blocks         Number of blocks (NB) for vLLM paged buffers
 * @param skip_prefix_n_blocks  Number of blocks to skip at the beginning
 */
void multi_layer_block_kv_transfer(
    const std::vector<torch::Tensor>& key_value_tensors,
    std::vector<torch::Tensor>& memory_objects, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    GPUKVFormat gpu_kv_format, int block_size, int num_blocks,
    int skip_prefix_n_blocks);

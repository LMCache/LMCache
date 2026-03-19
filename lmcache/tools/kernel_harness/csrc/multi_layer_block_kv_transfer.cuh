// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <vector>

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

enum class GPUKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,    // vLLM cross-layer: single tensor
  NL_X_TWO_NB_BS_NH_HS = 1,  // vLLM non-MLA flash attention: L separate tensors
  NL_X_NB_BS_HS = 3,         // vLLM MLA: L separate tensors
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

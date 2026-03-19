// SPDX-License-Identifier: Apache-2.0

#include "multi_layer_block_kv_transfer.cuh"

/**
 * Stub implementation of block-level multi-layer KV transfer.
 *
 * TODO: Replace this stub with the actual CUDA kernel implementation.
 *
 * The kernel should:
 * 1. For each block in block_ids (after skipping skip_prefix_n_blocks):
 *    - Determine which memory_object and local offset this block maps to
 *    - For each layer: copy BS tokens between the vLLM paged buffer and
 *      the LMCache memory object
 * 2. Handle all GPUKVFormat layouts:
 *    - NL_X_TWO_NB_BS_NH_HS: L separate tensors [2, NB, BS, NH, HS]
 *    - NB_NL_TWO_BS_NH_HS: single tensor [NB, NL, 2, BS, NH, HS]
 *    - NL_X_NB_BS_HS: L separate tensors [NB, BS, HS] (MLA)
 * 3. Support both bfloat16 and float8_e4m3fn
 * 4. Use async CUDA memcpy for efficient H2D/D2H transfers
 */
void multi_layer_block_kv_transfer(
    const std::vector<torch::Tensor>& key_value_tensors,
    std::vector<torch::Tensor>& memory_objects, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    GPUKVFormat gpu_kv_format, int block_size, int num_blocks,
    int skip_prefix_n_blocks) {
  TORCH_CHECK(false,
              "multi_layer_block_kv_transfer CUDA kernel not yet implemented. "
              "Use --use-reference flag to run with the Python reference "
              "implementation.");
}

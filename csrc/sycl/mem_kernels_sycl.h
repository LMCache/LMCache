// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include "../kv_transfer_types.h"
#include <vector>

<<<<<<< HEAD
=======
enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
TWO: 2
ONE: 1

_ means a dimension within the same tensor
_X_ means a dimension across a list

A_X_B_X_C_D_E means:
kv_cache: List[List[torch.Tensor]]
len(kv_cache) = A
len(kv_cache[0]) = B
kv_cache[0][0].shape = (C, D, E)

The logic for identifying the format currently lives in
`lmcache/v1/gpu_connector/utils.py`
*/
enum class EngineKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,
  /*
  used by:
  - vLLM CROSS_LAYER mode
  */

  NL_X_TWO_NB_BS_NH_HS = 1,
  /*
  used by:
  - vLLM non-MLA flash attention
  */

  NL_X_NB_TWO_BS_NH_HS = 2,
  /*
  used by:
  - vLLM non-MLA flash infer
  */

  NL_X_NB_BS_HS = 3,
  /*
  used by:
  - vLLM MLA
  */

  TWO_X_NL_X_NBBS_NH_HS = 4,
  /*
  used by:
  - SGLang MHA (flash attention and flash infer)
  */

  NL_X_NBBS_ONE_HS = 5,
  /*
  used by:
  - SGLang MLA
  */

  NL_X_TWO_NB_NH_BS_HS = 6,
  /*
  used by:
  - vLLM non-MLA flash attention (HND layout)
  physical shape per layer: [2, num_blocks, num_heads, block_size, head_size]
  */

  NL_X_NB_TWO_NH_BS_HS = 7,
  /*
  used by:
  - vLLM non-MLA flash infer (HND layout)
  physical shape per layer: [num_blocks, 2, num_heads, block_size, head_size]
  */

  NB_NL_TWO_NH_BS_HS = 8,
  /*
  used by:
  - TRT-LLM cross-layer (HND layout)
  physical shape: [num_blocks, num_layers, 2, num_heads, block_size, head_size]
  */

  TWO_X_NL_X_NB_BS_NH_HS = 9,
  /*
  used by:
  - SGLang MHA via the MP daemon path
  physical shape per layer: [num_blocks, block_size, num_heads, head_size]
  */

  NL_X_NB_NH_BS_TWO_HS = 10,
  /*
  used by:
  - vLLM non-MLA flash attention with K/V interleaved last
  physical shape per layer: [num_blocks, num_heads, block_size, 2, head_size]
  */
};

>>>>>>> 28d20cb4 (Rebase)
struct PageBufferShapeDesc {
  int kv_size = 0;
  int nl = 0;
  int nb = 0;
  int bs = 0;
  int nh = 0;
  int hs = 0;
  int element_size = 0;
  int block_stride_elems = 0;

  template <typename ScalarType>
  inline size_t scalars_per_head() const {
    return static_cast<size_t>(hs) * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  inline size_t scalars_per_token() const {
    return static_cast<size_t>(nh) * hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size = 0,
    const int head_size = 0, const int skip_prefix_n_tokens = 0);

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const TransferDirection direction,
                              const EngineKVFormat engine_kv_format,
                              const bool token_major = false);

// Asynchronous memory copy between host and device buffers.
// The `direction` parameter is retained for API compatibility but is unused:
// SYCL USM memcpy infers direction from pointer allocation types.
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

// deprecated / unused except in unit tests
void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

// deprecated / unused except in unit tests
void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);

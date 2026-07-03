// SPDX-License-Identifier: Apache-2.0

#pragma once

// Physical KV-cache memory layout an engine hands to LMCache, plus the
// classification predicates over it. Vendor-header-free, so every backend and
// the Python facade (lmc_ops) share one definition. Detection (raw layout ->
// format) lives in lmcache/v1/gpu_connector/kv_format.

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
  - vLLM non-MLA blocks-first attention with K/V fused into the trailing dim
  physical shape per layer: [num_blocks, num_heads, block_size, 2, head_size]
  (recovered by splitting the fused trailing [block_size, 2 * head_size]).
  Currently only reached via the host gather/scatter path, not the device
  transfer kernels.
  */
};

// __host__ __device__ under CUDA/HIP so the kernels can call these; the guard
// keeps the header vendor-runtime-free.
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define LMC_KV_FORMAT_HD __host__ __device__
#else
  #define LMC_KV_FORMAT_HD
#endif

// Structural shape of the normalized kv_caches: exactly one is true per format.

// All layers in one fused tensor.
LMC_KV_FORMAT_HD constexpr bool is_cross_layer(EngineKVFormat f) {
  return f == EngineKVFormat::NB_NL_TWO_BS_NH_HS ||
         f == EngineKVFormat::NB_NL_TWO_NH_BS_HS;
}

// Keys and values in two separate top-level lists: [key_layers, value_layers].
LMC_KV_FORMAT_HD constexpr bool is_kv_list(EngineKVFormat f) {
  return f == EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS ||
         f == EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS;
}

// One list entry per layer: kv_caches[layer_idx] is that layer's tensor.
LMC_KV_FORMAT_HD constexpr bool is_layer_list(EngineKVFormat f) {
  return f == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS ||
         f == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS ||
         f == EngineKVFormat::NL_X_NB_BS_HS ||
         f == EngineKVFormat::NL_X_NBBS_ONE_HS ||
         f == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||
         f == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS ||
         f == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS;
}

// Multi-head Latent Attention: a single latent KV head (no separate K/V split).
LMC_KV_FORMAT_HD constexpr bool is_mla(EngineKVFormat f) {
  return f == EngineKVFormat::NL_X_NB_BS_HS ||   // vLLM MLA
         f == EngineKVFormat::NL_X_NBBS_ONE_HS;  // SGLang MLA
}

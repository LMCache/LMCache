// SPDX-License-Identifier: Apache-2.0

#pragma once

// Physical KV-cache memory layout an engine hands to LMCache, plus the
// classification predicates over it. Vendor-header-free, so every backend and
// the Python facade (device_ops) share one definition. Detection (raw layout ->
// format) lives in lmcache/v1/gpu_connector/kv_format.

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
CS: content size (per-head content width; 2 * head size when K/V are fused)
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
  DEPRECATED: superseded by NL_X_NB_NH_BS_CS; no longer produced by detection.
  used by:
  - vLLM non-MLA blocks-first attention with K/V fused into the trailing dim
  physical shape per layer: [num_blocks, num_heads, block_size, 2, head_size]
  (recovered by splitting the fused trailing [block_size, 2 * head_size]).
  The device transfer kernels treat it as HND with kv_size == 1 and
  hs == 2 * head_size (the K/V axis stays packed inside each head copy).
  */

  NL_X_NB_BS_NH_TWO_HS = 11,
  /*
  DEPRECATED: superseded by NL_X_NB_BS_NH_CS; no longer produced by detection.
  used by:
  - vLLM non-MLA blocks-first attention (NHD layout) with K/V fused into the
    trailing dim
  physical shape per layer: [num_blocks, block_size, num_heads, 2, head_size]
  (recovered by splitting the fused trailing [num_heads, 2 * head_size]).
  Like NL_X_NB_NH_BS_TWO_HS but tokens before heads; the device transfer
  kernels treat it as NHD with kv_size == 1 and hs == 2 * head_size.
  */

  NL_X_NB_NH_BS_CS = 12,
  /*
  used by:
  - vLLM non-MLA blocks-first attention (HND layout) with K/V fused into the
    trailing content dim (unified KV cache)
  physical shape per layer: [num_blocks, num_heads, block_size, content_size]
  The device transfer kernels treat it as HND with kv_size == 1 and
  hs == content_size.
  */

  NL_X_NB_BS_NH_CS = 13,
  /*
  used by:
  - vLLM non-MLA blocks-first attention (NHD layout) with K/V fused into the
    trailing content dim (unified KV cache)
  physical shape per layer: [num_blocks, block_size, num_heads, content_size]
  Like NL_X_NB_NH_BS_CS but tokens before heads; the device transfer kernels
  treat it as NHD with kv_size == 1 and hs == content_size.
  */

  // vLLM DSA indexer k-cache [NB,BS,132] u8, paged [BSxvals][BSxscales]; kv 1
  NL_X_NB_BSV_BSS = 14,
};

// __host__ __device__ under CUDA/HIP so the kernels can call these; the guard
// keeps the header vendor-runtime-free.
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define LMC_KV_FORMAT_HD __host__ __device__
#else
  #define LMC_KV_FORMAT_HD
#endif

// Static layout facts for each format, declared once per format (indexed by the
// enum value). This mirrors the facts declared on each Python KVFormatSpec so
// the two sides can never drift; the predicates below are one-line lookups.
// Exactly one structural shape (cross_layer / kv_list / layer_list) is true per
// format. The remaining flags are modifiers layered on top of it.
// Every field defaults to false; the table below only sets the true ones.
struct FormatFacts {
  bool is_cross_layer = false;   // all layers in one fused tensor
  bool is_kv_list = false;       // keys and values in two top-level lists
  bool is_layer_list = false;    // one list entry per layer
  bool is_mla = false;           // MLA: single latent KV head (no separate K/V)
  bool is_hnd = false;           // heads before block tokens (HND layout)
  bool is_fused_packed = false;  // K/V packed in trailing dim (kv_size == 1)
  bool is_two_major = false;     // size-2 K/V axis precedes the block axis
  bool is_pbs_fused = false;     // paged buffer size fused into one axis
};

// clang-format off
LMC_KV_FORMAT_HD constexpr FormatFacts FORMAT_FACTS[] = {
    /*  0 NB_NL_TWO_BS_NH_HS      */ {.is_cross_layer = true},
    /*  1 NL_X_TWO_NB_BS_NH_HS    */ {.is_layer_list = true,  .is_two_major = true},
    /*  2 NL_X_NB_TWO_BS_NH_HS    */ {.is_layer_list = true},
    /*  3 NL_X_NB_BS_HS           */ {.is_layer_list = true,  .is_mla = true},
    /*  4 TWO_X_NL_X_NBBS_NH_HS   */ {.is_kv_list = true},
    /*  5 NL_X_NBBS_ONE_HS        */ {.is_layer_list = true,  .is_mla = true,
                                      .is_pbs_fused = true},
    /*  6 NL_X_TWO_NB_NH_BS_HS    */ {.is_layer_list = true,  .is_hnd = true,
                                      .is_two_major = true},
    /*  7 NL_X_NB_TWO_NH_BS_HS    */ {.is_layer_list = true,  .is_hnd = true},
    /*  8 NB_NL_TWO_NH_BS_HS      */ {.is_cross_layer = true, .is_hnd = true},
    /*  9 TWO_X_NL_X_NB_BS_NH_HS  */ {.is_kv_list = true},
    /* 10 NL_X_NB_NH_BS_TWO_HS    */ {.is_layer_list = true,  .is_hnd = true,
                                      .is_fused_packed = true},
    /* 11 NL_X_NB_BS_NH_TWO_HS    */ {.is_layer_list = true,  .is_fused_packed = true},
    /* 12 NL_X_NB_NH_BS_CS        */ {.is_layer_list = true,  .is_hnd = true,
                                      .is_fused_packed = true},
    /* 13 NL_X_NB_BS_NH_CS        */ {.is_layer_list = true,  .is_fused_packed = true},
    /* 14 NL_X_NB_BSV_BSS         */ {.is_layer_list = true,  .is_mla = true},
};
// clang-format on

LMC_KV_FORMAT_HD constexpr const FormatFacts& format_facts(EngineKVFormat f) {
  return FORMAT_FACTS[static_cast<int>(f)];
}

// All layers in one fused tensor.
LMC_KV_FORMAT_HD constexpr bool is_cross_layer(EngineKVFormat f) {
  return format_facts(f).is_cross_layer;
}

// Keys and values in two separate top-level lists: [key_layers, value_layers].
LMC_KV_FORMAT_HD constexpr bool is_kv_list(EngineKVFormat f) {
  return format_facts(f).is_kv_list;
}

// One list entry per layer: kv_caches[layer_idx] is that layer's tensor.
LMC_KV_FORMAT_HD constexpr bool is_layer_list(EngineKVFormat f) {
  return format_facts(f).is_layer_list;
}

// Multi-head Latent Attention: a single latent KV head (no separate K/V split).
// The blocked-scale indexer cache transfers like MLA (single plane,
// kv_size == 1); only its paged addressing differs.
LMC_KV_FORMAT_HD constexpr bool is_mla(EngineKVFormat f) {
  return format_facts(f).is_mla;
}

// vLLM fused K/V: K and V packed in the trailing dim (2 * head_size), no
// separate K/V axis — transferred as one k_or_v == 0 pass (like MLA).
LMC_KV_FORMAT_HD constexpr bool is_fused_packed(EngineKVFormat f) {
  return format_facts(f).is_fused_packed;
}

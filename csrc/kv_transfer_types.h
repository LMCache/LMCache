// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-agnostic descriptors for KV-cache transfers.
//
// These enums describe *what* is being transferred (the direction of a
// host/device copy and the physical KV-cache memory layout) and carry no
// accelerator-specific dependency. They are intentionally free of any
// <torch/...>, <ATen/...> or vendor runtime headers so that every backend
// (CUDA, ROCm, MUSA, SYCL/XPU, ...) can share a single definition instead of
// redeclaring them in each accelerator-coupled kernel header.

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
  - vLLM non-MLA blocks-first attention with K/V fused into the trailing dim
  physical shape per layer: [num_blocks, num_heads, block_size, 2, head_size]
  (recovered by splitting the fused trailing [block_size, 2 * head_size]).
  Currently only reached via the host gather/scatter path, not the device
  transfer kernels.
  */
};

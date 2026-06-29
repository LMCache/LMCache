# SPDX-License-Identifier: Apache-2.0
#
# AUTO-GENERATED FROM csrc/engine_kv_format.def -- DO NOT EDIT.
# Regenerate with: python tools/gen_engine_kv_format.py
# (or `pre-commit run gen-engine-kv-format --all-files`).
#
# This is the static, typed Python mirror of the C++ EngineKVFormat enum. The
# members are single-sourced in csrc/engine_kv_format.def; edit that file (one
# X(...) line) and regenerate, never edit this file by hand.

# Standard
from enum import IntEnum


class EngineKVFormat(IntEnum):
    """Enumeration of different engine KV cache memory layouts."""

    # used by: vLLM CROSS_LAYER mode
    NB_NL_TWO_BS_NH_HS = 0

    # used by: vLLM non-MLA flash attention
    NL_X_TWO_NB_BS_NH_HS = 1

    # used by: vLLM non-MLA flash infer
    NL_X_NB_TWO_BS_NH_HS = 2

    # used by: vLLM MLA
    NL_X_NB_BS_HS = 3

    # used by: SGLang MHA (flash attention and flash infer)
    TWO_X_NL_X_NBBS_NH_HS = 4

    # used by: SGLang MLA
    NL_X_NBBS_ONE_HS = 5

    # used by: vLLM non-MLA flash attention (HND layout)
    # physical shape per layer: [2, num_blocks, num_heads, block_size, head_size]
    NL_X_TWO_NB_NH_BS_HS = 6

    # used by: vLLM non-MLA flash infer (HND layout)
    # physical shape per layer: [num_blocks, 2, num_heads, block_size, head_size]
    NL_X_NB_TWO_NH_BS_HS = 7

    # used by: TRT-LLM cross-layer (HND layout)
    # physical shape: [num_blocks, num_layers, 2, num_heads, block_size, head_size]
    NB_NL_TWO_NH_BS_HS = 8

    # used by: SGLang MHA via the MP daemon path
    # physical shape per layer: [num_blocks, block_size, num_heads, head_size]
    TWO_X_NL_X_NB_BS_NH_HS = 9

    # used by: vLLM non-MLA blocks-first attention with K/V fused into the trailing
    # dim. physical shape per layer: [num_blocks, num_heads, block_size, 2,
    # head_size] (recovered by splitting the fused trailing [block_size,
    # 2 * head_size]). Currently only reached via the host gather/scatter path,
    # not the device transfer kernels.
    NL_X_NB_NH_BS_TWO_HS = 10


# Backward-compat alias for the pre-#3673 GPUKVFormat name.
GPUKVFormat = EngineKVFormat

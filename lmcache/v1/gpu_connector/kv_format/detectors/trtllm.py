# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM KV cache detector."""

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detection_base import (
    EngineDetector,
    list_depth_tensor_dim,
)
import lmcache.c_ops as lmc_ops


class TRTLLMDetector(EngineDetector):
    abstract: ClassVar[bool] = False
    engine: ClassVar = EngineType.TRTLLM

    def normalize(self, kv_caches, layout_hints):
        # TRT-LLM hands us a 4-D pool tensor (possibly wrapped in a
        # 1-element list for adapter-side ergonomics). Reshape to the
        # canonical 6-D cross-layer form here.
        if isinstance(kv_caches, list) and len(kv_caches) == 1:
            kv_caches = kv_caches[0]
        if isinstance(kv_caches, torch.Tensor) and kv_caches.dim() == 4:
            num_kv_heads = layout_hints.get("num_kv_heads")
            tokens_per_block = layout_hints.get("tokens_per_block")
            head_dim = layout_hints.get("head_dim")
            if num_kv_heads is None or tokens_per_block is None or head_dim is None:
                raise ValueError(
                    "TRT-LLM normalize requires layout_hints with "
                    "num_kv_heads, tokens_per_block, head_dim"
                )
            nb, nl, kv, flat = kv_caches.shape
            if flat != num_kv_heads * tokens_per_block * head_dim:
                raise ValueError(
                    f"TRT-LLM 4D tensor flat dim {flat} does not match "
                    f"num_kv_heads ({num_kv_heads}) * tokens_per_block "
                    f"({tokens_per_block}) * head_dim ({head_dim})"
                )
            kv_caches = kv_caches.view(
                nb, nl, kv, num_kv_heads, tokens_per_block, head_dim
            )
        return kv_caches

    def detect(self, kv_caches, layout_hints):
        list_depth, tensor_dim = list_depth_tensor_dim(kv_caches)
        if list_depth == 0 and tensor_dim == 6:
            return lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS
        return None

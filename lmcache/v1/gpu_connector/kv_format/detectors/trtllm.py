# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM KV cache detector."""

# Standard
from typing import ClassVar, Optional

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detection_base import (
    EngineDetector,
    list_depth_tensor_dim,
)
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
)
import lmcache.c_ops as lmc_ops


class TRTLLMDetector(EngineDetector):
    """Detector for TRT-LLM serving engine KV cache layouts."""

    abstract: ClassVar[bool] = False
    engine: ClassVar = EngineType.TRTLLM

    def normalize(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> DiscoverableKVCache:
        """Reshape TRT-LLM's 4-D pool tensor to the canonical 6-D form.

        TRT-LLM hands us a 4-D pool tensor (possibly wrapped in a
        1-element list for adapter-side ergonomics). The block dim is
        flattened across ``num_kv_heads``, ``tokens_per_block`` and
        ``head_dim``, so we ``view`` it back to the standard 6-D
        cross-layer shape ``(NB, NL, 2, NH, BS, HS)`` here.

        Args:
            kv_caches: Either a 4-D :class:`torch.Tensor` or a
                1-element list wrapping one. Already-6-D inputs pass
                through untouched.
            layout_hints: Must contain ``num_kv_heads``,
                ``tokens_per_block`` and ``head_dim`` when the input is
                4-D.

        Returns:
            The 6-D canonical form (or the unmodified input when no
            reshape was needed).

        Raises:
            ValueError: If required hints are missing or the flat
                dimension does not match ``NH * BS * HS``.
        """
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

    def detect(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> Optional["lmc_ops.GPUKVFormat"]:
        """Identify the GPU KV format for TRT-LLM.

        Args:
            kv_caches: KV cache structure already passed through
                :meth:`normalize`.
            layout_hints: Engine-supplied layout hints (unused here).

        Returns:
            The matching ``GPUKVFormat`` enum value, or ``None`` if the
            structure does not match any known TRT-LLM layout.
        """
        list_depth, tensor_dim = list_depth_tensor_dim(kv_caches)
        if list_depth == 0 and tensor_dim == 6:
            return lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS
        return None

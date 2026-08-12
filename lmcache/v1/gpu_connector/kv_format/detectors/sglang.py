# SPDX-License-Identifier: Apache-2.0
"""SGLang KV cache discovery."""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops


def _two_level_nested_format(
    kv_caches: DiscoverableKVCache,
) -> Optional[lmc_ops.EngineKVFormat]:
    list_depth, tensor_ndim, _ = measure_list_depth_until_tensor(kv_caches)
    if list_depth != 2:
        return None
    return (
        lmc_ops.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
        if tensor_ndim == 4
        else lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS
    )


def _reshape_and_regroup_mha_gqa(
    kv_caches: list[torch.Tensor], block_size: Optional[int]
) -> Optional[list[list[torch.Tensor]]]:
    if block_size is None or len(kv_caches) % 2 != 0:
        return None

    half = len(kv_caches) // 2
    regrouped: list[list[torch.Tensor]] = []
    for layers in (kv_caches[:half], kv_caches[half:]):
        reshaped: list[torch.Tensor] = []
        for layer in layers:
            if layer.shape[0] % block_size != 0:
                raise ValueError(
                    f"SGLang page_buffer_size {layer.shape[0]} not "
                    f"divisible by tokens_per_block {block_size}"
                )
            num_blocks = layer.shape[0] // block_size
            reshaped.append(layer.view(num_blocks, block_size, *layer.shape[1:]))
        regrouped.append(reshaped)

    return regrouped


class SGLANG_Detector(EngineDetector):
    engine_type = EngineType.SGLANG

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmc_ops.EngineKVFormat], DiscoverableKVCache]":
        # -- Non-list/empty-list → try nested 2-level [[K_layers], [V_layers]] --
        if not (isinstance(kv_caches, list) and len(kv_caches) > 0):
            nested_format = _two_level_nested_format(kv_caches)
            if nested_format is not None:
                return nested_format, kv_caches
            return None, kv_caches

        first = kv_caches[0]

        # -- 2-level nested [[K_layers], [V_layers]] (non-MP MHA/GQA) --
        if not isinstance(first, torch.Tensor):
            nested_format = _two_level_nested_format(kv_caches)
            if nested_format is not None:
                return nested_format, kv_caches
            return None, kv_caches

        block_size = layout_hints.get("tokens_per_block")

        # -- shape[1] > 1 (MHA/GQA): flat list[2*NL] of 3-D (PBS, NH, HS) --
        # Regroup [K_layers, V_layers] and reshape (PBS, NH, HS) → (NB, BS, NH, HS).
        if first.dim() == 3 and first.shape[1] > 1:
            regrouped = _reshape_and_regroup_mha_gqa(kv_caches, block_size)
            if regrouped is None:
                return None, kv_caches

            # Regrouped into 2-level nested → detect final format
            nested_format = _two_level_nested_format(regrouped)
            if nested_format is not None:
                return nested_format, regrouped
            return None, regrouped

        # -- shape[1] == 1 (MLA / DSA-mixed): list of 3-D (PBS, 1, HS) --
        # Pure MLA: all layers uniform → reshape with tokens_per_block.
        # DSA-mixed: trailing 2-D index buffers → pass through, per-shape retry
        # in normalize_and_discover_per_layer_formats handles each group.
        elif first.dim() == 3 and first.shape[1] == 1:
            if block_size is not None:
                reshaped: list[DiscoverableKVCache] = []
                for layer in kv_caches:
                    if (
                        not isinstance(layer, torch.Tensor)
                        or layer.dim() != 3
                        or layer.shape[1] != 1
                        or layer.shape[0] % block_size != 0
                    ):
                        # Mixed shape (DSA) — let per-shape retry classify each group
                        return (
                            lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS,
                            kv_caches,
                        )
                    num_blocks = layer.shape[0] // block_size
                    reshaped.append(layer.view(num_blocks, block_size, layer.shape[2]))
                return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS, reshaped
            # No block_size → generic pass-through
            return lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS, kv_caches

        # -- 2-D uint8 (DSA index buffer): list[NL] of (num_pages, page_bytes) --
        # Treat each page as an opaque slot (block_size=1).
        elif first.dim() == 2 and first.dtype == torch.uint8:
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS, [
                t.unsqueeze(1) for t in kv_caches
            ]

        return None, kv_caches

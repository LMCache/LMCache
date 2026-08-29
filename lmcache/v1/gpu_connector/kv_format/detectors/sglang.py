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
import lmcache.lmcache_native as lmcache_native


class SGLANG_Detector(EngineDetector):
    engine_type = EngineType.SGLANG

    def discover(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        # MP path: a flat list[2*NL] of 3-D tensors (K layers then V layers)
        # plus explicit K/V-list-layout and tokens-per-block hints. Regroup
        # into [K_layers, V_layers] and reshape each
        # (PBS, NH, HS) -> (NB, BS, NH, HS). The explicit layout hint is
        # required because NH can be 1 after tensor parallelism, which is
        # otherwise indistinguishable by shape from SGLang MLA.
        if (
            isinstance(kv_caches, list)
            and len(kv_caches) > 0
            and len(kv_caches) % 2 == 0
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 3
            and layout_hints.get("kv_list_layout") == "k_v"
            and "tokens_per_block" in layout_hints
        ):
            block_size = layout_hints["tokens_per_block"]
            half = len(kv_caches) // 2
            regrouped: list[DiscoverableKVCache] = []
            for layers in (kv_caches[:half], kv_caches[half:]):
                reshaped: list[DiscoverableKVCache] = []
                for layer in layers:
                    page_buffer_size = layer.shape[0]
                    if page_buffer_size % block_size != 0:
                        raise ValueError(
                            f"SGLang page_buffer_size {page_buffer_size} not "
                            f"divisible by tokens_per_block {block_size}"
                        )
                    num_blocks = page_buffer_size // block_size
                    reshaped.append(
                        layer.view(num_blocks, block_size, *layer.shape[1:])
                    )
                regrouped.append(reshaped)
            kv_caches = regrouped

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )
        if list_depth == 1 and first_tensor.shape[1] == 1:  # MLA, fused PBS
            mla_block_size = layout_hints.get("tokens_per_block")
            if mla_block_size:
                # Un-fuse the folded dim-0 (num_blocks*block_size) into a real
                # block axis and drop the singleton head, so the tensor carries
                # its own block size (NL_X_NB_BS_HS, same as vLLM MLA) instead of
                # the block-less fused format whose block_size() is undefined.
                mla_reshaped: list[DiscoverableKVCache] = []
                for layer in kv_caches:
                    page_buffer_size = layer.shape[0]
                    if page_buffer_size % mla_block_size != 0:
                        raise ValueError(
                            f"SGLang MLA page_buffer_size {page_buffer_size} not "
                            f"divisible by tokens_per_block {mla_block_size}"
                        )
                    mla_reshaped.append(
                        layer.view(
                            page_buffer_size // mla_block_size,
                            mla_block_size,
                            layer.shape[2],
                        )
                    )
                return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, mla_reshaped
            return lmcache_native.EngineKVFormat.NL_X_NBBS_ONE_HS, kv_caches
        if list_depth == 2:
            if tensor_ndim == 4:  # MP daemon: NB/BS split into separate axes
                return lmcache_native.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS, kv_caches
            return lmcache_native.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS, kv_caches
        return None, kv_caches

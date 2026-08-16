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
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        # MP path: a flat list[2*NL] of 3-D tensors (K layers then V layers)
        # plus a tokens_per_block hint. Regroup into [K_layers, V_layers] and
        # reshape each (PBS, NH, HS) -> (NB, BS, NH, HS).
        if (
            isinstance(kv_caches, list)
            and len(kv_caches) > 0
            and len(kv_caches) % 2 == 0
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 3
            and kv_caches[0].shape[1] > 1
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
        # DSA indexer cache: list[NL] of 2-D uint8 tensors. Each tensor is
        # (num_pages, page_size * 132) where 132 = 128B fp8 key + 4B fp32 scale
        # per token. SGLang packs the page dimension into the last axis (unlike
        # vLLM which keeps a 3-D (NB, BS, 132) layout), so we check divisibility
        # by 132 rather than equality. Detected before MLA since the indexer is
        # 2-D while MLA latent is 3-D with a singleton head axis.
        #
        # Reshape (num_pages, page_size * 132) -> (num_pages, page_size, 132)
        # so the tensor matches the 3-D (NB, BS, HS) layout that the
        # NL_X_NB_BSV_BSS spec (inheriting NL_X_NB_BS_HS_Spec) expects. The
        # spec's geometry accessors only read shape/dtype/data_ptr, so the view
        # is safe even though the physical per-block layout interleaves values
        # and scales differently ([BS x 128 vals][BS x 4B scales]); the
        # transfer kernels address values and scales separately via raw ptrs.
        if (
            list_depth == 1
            and tensor_ndim == 2
            and first_tensor.dtype == torch.uint8
            and int(first_tensor.shape[-1]) % 132 == 0
        ):
            bs = int(first_tensor.shape[-1]) // 132
            kv_caches = [t.view(t.shape[0], bs, 132) for t in kv_caches]
            return lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS, kv_caches
        if list_depth == 1 and first_tensor.shape[1] == 1:  # MLA, fused PBS
            return lmcache_native.EngineKVFormat.NL_X_NBBS_ONE_HS, kv_caches
        if list_depth == 2:
            if tensor_ndim == 4:  # MP daemon: NB/BS split into separate axes
                return lmcache_native.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS, kv_caches
            return lmcache_native.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS, kv_caches
        return None, kv_caches

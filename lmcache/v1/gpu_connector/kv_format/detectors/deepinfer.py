# SPDX-License-Identifier: Apache-2.0
"""DeepInfer KV-cache format discovery."""

# Standard
from typing import Optional

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


class DeepInferDetector(EngineDetector):
    """Detect DeepInfer's per-component paged cache views."""

    engine_type = EngineType.DEEPINFER

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        """Recognize DeepInfer's ``[num_blocks, slots, width]`` tensors.

        Args:
            kv_caches: Per-component cache tensors registered by DeepInfer.
            layout_hints: Reserved for future DeepInfer layouts.

        Returns:
            The paged single-vector format and original tensor views, or
            ``None`` when the structure is not a DeepInfer paged cache.
        """
        del layout_hints
        list_depth, tensor_ndim, _first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )
        if list_depth == 1 and tensor_ndim == 3:
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        return None, kv_caches

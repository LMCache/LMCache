# SPDX-License-Identifier: Apache-2.0
"""ATOM KV-cache format discovery."""

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


class AtomDetector(EngineDetector):
    """Detect ATOM's per-layer paged latent and index cache views."""

    engine_type = EngineType.ATOM

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        """Recognize ATOM's ``[num_blocks, block_size, width]`` tensors.

        Args:
            kv_caches: Per-layer latent or index cache tensors.
            layout_hints: Reserved for future ATOM layouts.

        Returns:
            The paged single-vector format and original tensor views, or
            ``None`` when the structure is not an ATOM paged cache.
        """
        del layout_hints
        list_depth, tensor_ndim, _first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )
        if list_depth == 1 and tensor_ndim == 3:
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        return None, kv_caches

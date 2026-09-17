# SPDX-License-Identifier: Apache-2.0
"""Basic benchmark views, independent of serving-engine layout conventions."""

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import EngineDetector
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


class MockDetector(EngineDetector):
    engine_type = EngineType.MOCK

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> tuple[lmcache_native.EngineKVFormat | None, DiscoverableKVCache]:
        """Match a nonempty list of [blocks, slots, content] views with NHD hints.

        Returns the basic format and unchanged input, or None for unsupported
        structures/layout hints. Contiguity is checked by the common pipeline.
        """
        if (
            layout_hints.get("kv_layout", "NHD") == "NHD"
            and isinstance(kv_caches, list)
            and kv_caches
            and all(isinstance(t, torch.Tensor) and t.ndim == 3 for t in kv_caches)
        ):
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        return None, kv_caches

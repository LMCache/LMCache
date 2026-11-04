# SPDX-License-Identifier: Apache-2.0
"""SGLang KV cache discovery."""

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


class SGLANG_Detector(EngineDetector):
    engine_type = EngineType.SGLANG

    def discover(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        del layout_hints
        list_depth, tensor_ndim, _ = measure_list_depth_until_tensor(kv_caches)
        if list_depth != 1:
            return None, kv_caches
        if tensor_ndim == 3:
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        if tensor_ndim == 4:
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_HS, kv_caches
        return None, kv_caches

# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery."""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


def resolve_vllm_kv_layout(
    layout_hints: LayoutHints, cpu_attention_backend: bool
) -> str:
    """Resolve vLLM's KV layout from engine hints and backend behavior."""
    if cpu_attention_backend:
        return "HND"
    return layout_hints.get("kv_layout", "NHD")


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        # vLLM's CPU attention backend stores KV in HND but misreports it, so
        # force HND there; otherwise honor the hint, defaulting to NHD.
        kv_layout = resolve_vllm_kv_layout(
            layout_hints, cpu_attention_backend=torch_device_type == "cpu"
        )
        is_hnd = kv_layout == "HND"

        # Blocks-first fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (a 5-D split would collide with
        # flash-infer when num_heads == 2). The two middle axes are NH/BS
        # (HND) or BS/NH (NHD) -- indistinguishable from the shape alone, so the
        # resolved kv_layout decides. The tensor is kept raw: the trailing axis
        # is the per-head content size (2 * head_size, K/V packed).
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            if is_hnd:
                return lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS, kv_caches
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS, kv_caches

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )

        if list_depth == 0:
            return lmcache_native.EngineKVFormat.NB_NL_TWO_BS_NH_HS, kv_caches
        # vLLM-RBLN: HND with a singleton between heads and block tokens that
        # its attention backend requires. Always HND, so the hint is not read.
        if (
            list_depth == 1
            and tensor_ndim == 6
            and first_tensor.shape[0] == 2
            and first_tensor.shape[3] == 1
        ):
            return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 5:
            if first_tensor.shape[0] == 2:  # K/V axis first
                if is_hnd:
                    return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_caches
                return lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_caches
            if first_tensor.shape[1] == 2:  # num_blocks first
                if is_hnd:
                    return lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_caches
                return lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 3:  # MLA (or DSA indexer cache)
            if first_tensor.dtype == torch.uint8 and int(first_tensor.shape[-1]) == 132:
                return lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS, kv_caches
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        if (
            list_depth == 2
            and tensor_ndim == 4
            and isinstance(kv_caches[0], (list, tuple))
            and len(kv_caches[0]) == 2
        ):
            return lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS, kv_caches
        return None, kv_caches

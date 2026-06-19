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
import lmcache.c_ops as lmc_ops


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmc_ops.EngineKVFormat], DiscoverableKVCache]":
        # Blocks-first fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (the post-split 5-D shape would collide
        # with flash-infer when num_heads == 2). Split [NB, NH, BS, 2*HS] into
        # [NB, NH, BS, 2, HS].
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            fused_dim = kv_caches[0].shape[3]
            if fused_dim % 2 != 0:
                raise ValueError(
                    f"blocks-first fused trailing dim {fused_dim} is not 2 * head_size"
                )
            split = [t.reshape(*t.shape[:3], 2, fused_dim // 2) for t in kv_caches]
            return lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS, split

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )

        # vLLM's CPU attention backend stores KV in HND but misreports it, so
        # force HND there; otherwise honor the hint, defaulting to NHD.
        kv_layout = layout_hints.get("kv_layout")
        if torch_device_type == "cpu":
            kv_layout = "HND"
        elif kv_layout is None:
            kv_layout = "NHD"
        is_hnd = kv_layout == "HND"

        if list_depth == 0:
            return lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 5:
            if first_tensor.shape[0] == 2:  # K/V axis first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_caches
            if first_tensor.shape[1] == 2:  # num_blocks first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 3:  # MLA
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        return None, kv_caches

# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache detector."""

# Standard
from typing import ClassVar

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detection_base import (
    EngineDetector,
    descend_to_tensor,
    list_depth_tensor_dim,
)
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class VLLMDetector(EngineDetector):
    abstract: ClassVar[bool] = False
    engine: ClassVar = EngineType.VLLM

    def detect(self, kv_caches, layout_hints):
        kv_layout = layout_hints.get("kv_layout")
        # NOTE: vLLM's CPU attention backend stores KV cache in HND layout.
        # However, ``get_kv_cache_layout`` from
        # ``vllm.v1.attention.backends.utils`` does not return the right
        # layout for CPU attention. The proper fix should come from the
        # vLLM side, but hardcode here as a safeguard.
        if torch_device_type == "cpu":
            kv_layout = "HND"
            logger.info("CPU backend detected, using HND KV cache layout")
        elif kv_layout is None:
            logger.warning(
                "No KV Cache Layout hint provided when using vLLM, defaulting to NHD"
            )
            kv_layout = "NHD"
        logger.info("vLLM KV cache layout: %s", kv_layout)
        is_hnd = kv_layout == "HND"

        list_depth, tensor_dim = list_depth_tensor_dim(kv_caches)
        if list_depth == 0:
            return lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS
        if list_depth == 1:
            probe = descend_to_tensor(kv_caches, 1)
            if tensor_dim == 5:
                if probe.shape[0] == 2:
                    return (
                        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS
                        if is_hnd
                        else lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
                    )
                if probe.shape[1] == 2:
                    return (
                        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS
                        if is_hnd
                        else lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS
                    )
            elif tensor_dim == 3:
                return lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
        return None

# SPDX-License-Identifier: Apache-2.0
"""vLLM cross-layer KV layout (NHD)."""

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.gpu_connector.kv_format.base import AxisLayout
from lmcache.v1.gpu_connector.kv_format.kv_format_spec_families import CrossLayer6DSpec
import lmcache.c_ops as lmc_ops


class VLLMCrossLayerNHDSpec(CrossLayer6DSpec):
    """vLLM CROSS_LAYER, NHD: ``[NB, NL, 2, BS, NH, HS]``."""

    abstract: ClassVar[bool] = False
    engine: ClassVar[str] = "vllm"
    layout: ClassVar[AxisLayout] = AxisLayout.NHD
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS
    shape_desc: ClassVar = "[NB, NL, 2, BS, NH, HS]"
    backend_label: ClassVar = "vLLM CROSS_LAYER"

    _AX_NB: ClassVar[int] = 0
    _AX_NL: ClassVar[int] = 1
    _AX_TWO: ClassVar[int] = 2
    _AX_BS: ClassVar[int] = 3
    _AX_NH: ClassVar[int] = 4
    _AX_HS: ClassVar[int] = 5

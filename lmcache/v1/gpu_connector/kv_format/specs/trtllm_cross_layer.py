# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM cross-layer KV layout (HND)."""

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.gpu_connector.kv_format.base import AxisLayout
from lmcache.v1.gpu_connector.kv_format.families import CrossLayer6DSpec
import lmcache.c_ops as lmc_ops


class TRTLLMCrossLayerHNDSpec(CrossLayer6DSpec):
    """TRT-LLM cross-layer, HND: ``[NB, NL, 2, NH, BS, HS]``."""

    abstract: ClassVar[bool] = False
    engine: ClassVar[str] = "trtllm"
    layout: ClassVar[AxisLayout] = AxisLayout.HND
    is_hnd: ClassVar[bool] = True
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.NB_NL_TWO_NH_BS_HS
    shape_desc: ClassVar = "[NB, NL, 2, NH, BS, HS]"
    backend_label: ClassVar = "TRT-LLM cross-layer (HND layout)"

    _AX_NB: ClassVar[int] = 0
    _AX_NL: ClassVar[int] = 1
    _AX_TWO: ClassVar[int] = 2
    _AX_NH: ClassVar[int] = 3
    _AX_BS: ClassVar[int] = 4
    _AX_HS: ClassVar[int] = 5

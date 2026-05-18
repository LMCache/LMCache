# SPDX-License-Identifier: Apache-2.0
"""vLLM non-MLA flash-infer KV layouts (NHD and HND)."""

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.gpu_connector.kv_format.base import AxisLayout
from lmcache.v1.gpu_connector.kv_format.families import PerLayer5DSpec
import lmcache.c_ops as lmc_ops


class VLLMFlashInferNHDSpec(PerLayer5DSpec):
    """vLLM non-MLA flash infer: ``NL x [NB, 2, BS, NH, HS]``."""

    abstract: ClassVar[bool] = False
    engine: ClassVar[str] = "vllm"
    layout: ClassVar[AxisLayout] = AxisLayout.NHD
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS
    shape_desc: ClassVar = "NL x [NB, 2, BS, NH, HS]"
    backend_label: ClassVar = "vLLM non-MLA flash infer"

    _AX_NB: ClassVar[int] = 0
    _AX_TWO: ClassVar[int] = 1
    _AX_BS: ClassVar[int] = 2
    _AX_NH: ClassVar[int] = 3
    _AX_HS: ClassVar[int] = 4


class VLLMFlashInferHNDSpec(PerLayer5DSpec):
    """vLLM non-MLA flash infer HND: ``NL x [NB, 2, NH, BS, HS]``."""

    abstract: ClassVar[bool] = False
    engine: ClassVar[str] = "vllm"
    layout: ClassVar[AxisLayout] = AxisLayout.HND
    is_hnd: ClassVar[bool] = True
    gpu_kv_format: ClassVar = lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS
    shape_desc: ClassVar = "NL x [NB, 2, NH, BS, HS]"
    backend_label: ClassVar = "vLLM non-MLA flash infer (HND layout)"

    _AX_NB: ClassVar[int] = 0
    _AX_TWO: ClassVar[int] = 1
    _AX_NH: ClassVar[int] = 2
    _AX_BS: ClassVar[int] = 3
    _AX_HS: ClassVar[int] = 4

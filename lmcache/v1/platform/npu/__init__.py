# SPDX-License-Identifier: Apache-2.0
"""NPU (Huawei Ascend) platform helpers."""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.npu.device_ops import NpuDeviceOps

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class NpuDeviceSpec(DeviceSpec):
    """NPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "npu"

    @property
    def torch_module_name(self) -> str:
        return "npu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        return NpuDeviceOps

    def is_available(self) -> bool:
        """Check NPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            return False

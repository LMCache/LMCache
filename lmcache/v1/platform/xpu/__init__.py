# SPDX-License-Identifier: Apache-2.0
"""XPU (Intel SYCL) platform helpers."""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base_device_ops import DeviceOps

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class XpuDeviceSpec(DeviceSpec):
    """XPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "xpu"

    @property
    def torch_module_name(self) -> str:
        return "xpu"

    @property
    def ops_cls(self) -> "type[DeviceOps]":
        # First Party
        from lmcache.v1.platform.xpu.device_ops import XpuDeviceOps

        return XpuDeviceOps

    def is_available(self) -> bool:
        """Check XPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False

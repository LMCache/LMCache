# SPDX-License-Identifier: Apache-2.0
"""HPU (Habana Gaudi) platform helpers."""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class HpuDeviceSpec(DeviceSpec):
    """HPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "hpu"

    @property
    def torch_module_name(self) -> str:
        return "hpu"

    @property
    def ops_module(self) -> str | None:
        return None

    def is_available(self) -> bool:
        """Check HPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "hpu") and torch.hpu.is_available()
        except Exception:
            return False

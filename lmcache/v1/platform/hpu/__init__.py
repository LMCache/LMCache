# SPDX-License-Identifier: Apache-2.0
"""HPU (Habana Gaudi) platform helpers."""

# Future
from __future__ import annotations

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


class HpuDeviceSpec(DeviceSpec):
    """HPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "hpu"

    @property
    def torch_module_name(self) -> str:
        return "hpu"

    @property
    def ops_cls(self) -> "type[DeviceOps]":
        # First Party
        from lmcache.v1.platform.hpu.device_ops import HpuDeviceOps

        return HpuDeviceOps

    def is_available(self) -> bool:
        """Check HPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "hpu") and torch.hpu.is_available()
        except Exception:
            return False

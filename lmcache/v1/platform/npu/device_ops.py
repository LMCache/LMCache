# SPDX-License-Identifier: Apache-2.0
"""NPU ops backend: inherit the torch baseline unchanged.

:class:`NpuDeviceOps` gives the registry a ``device_type="npu"`` entry.
All ops are inherited from :class:`DeviceOps` via MRO.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps


class NpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "npu"

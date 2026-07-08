# SPDX-License-Identifier: Apache-2.0
"""HPU ops backend: inherit the torch baseline unchanged.

:class:`HpuDeviceOps` is an empty subclass that gives the registry a
``device_type="hpu"`` entry. All ops inherit the torch baseline via
:meth:`DeviceOps.populate_module`.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base_device_ops import DeviceOps


class HpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "hpu"
    # All ops inherited from the torch baseline.

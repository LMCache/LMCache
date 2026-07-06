# SPDX-License-Identifier: Apache-2.0
"""CPU ops backend: the torch baseline, registered under ``device_type="cpu"``.

:class:`CpuDeviceOps` adds no overrides -- the inherited :class:`DeviceOps`
methods delegating to :mod:`lmcache.v1.platform._torch_ops` *are* the CPU
backend. It exists only to give the registry a concrete ``device_type="cpu"``
entry distinct from the unregistered :class:`DeviceOps` base.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base_device_ops import DeviceOps


class CpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cpu"
    # No overrides: the inherited torch baseline is the CPU backend.

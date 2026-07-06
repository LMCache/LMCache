# SPDX-License-Identifier: Apache-2.0
"""HPU ops backend: inherit the torch baseline unchanged.

:class:`HpuDeviceOps` is an empty subclass that gives the registry a
``device_type="hpu"`` entry. All ops inherit the torch baseline, matching
today's HPU path (which has no ``backend_candidates`` entry and runs every op on
the Python fallback). Add overrides later if profiling or pointer-mode callers
require them.
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

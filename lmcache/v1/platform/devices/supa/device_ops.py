# SPDX-License-Identifier: Apache-2.0
"""SUPA ops backend: inherit the torch baseline unchanged.

:class:`SupaDeviceOps` gives the registry a ``device_type="supa"`` entry.
``torch.supa`` exposes the CUDA-like surface the torch baseline depends on
(``Stream``, ``Event``, ``current_stream``, ``synchronize`` and memory stats),
so all ops are inherited from :class:`DeviceOps` via MRO.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps


class SupaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "supa"

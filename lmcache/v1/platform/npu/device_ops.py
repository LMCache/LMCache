# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU ops backend: bulk-bind the plugin's ``lmcache_ascend.c_ops``.

:class:`NpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the external Ascend plugin's compiled extension on top of the
torch baseline. All NPU-specific kernels live in the ``lmcache_ascend``
plugin; this package only wires detection and backend selection. If the
plugin is missing, a warning is logged and the instance stays on the
torch fallback (soft-fail, same as CUDA).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class NpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "npu"

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # Third Party
            import lmcache_ascend.c_ops as native
        except ImportError:
            logger.warning(
                "lmcache_ascend.c_ops plugin not found; NpuDeviceOps stays "
                "on the torch baseline for all ops."
            )
            return
        self.bind_native(native)

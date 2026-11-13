# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the SYCL extension on top of the torch baseline.  If the extension
is not built or fails to load, the cause is logged and the instance stays on
the torch fallback.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_ops import (
    DeviceOps,
    log_native_import_failure,
)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError as exc:
            log_native_import_failure("lmcache.xpu_ops", exc)
            return
        self.bind_native(sycl)

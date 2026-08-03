# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.c_ops`` extension.

:class:`CudaDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the compiled CUDA extension on top of the torch baseline.  If the
extension is missing, a warning is logged and the instance stays on the
torch fallback (soft-fail, same as XPU).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"

    logger.warning(
        " testing cuda using python fallbacks!!!! to be reverted before merging!!!"
    )

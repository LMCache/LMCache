# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.cuda_ops`` extension.

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

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.cuda_ops as native
        except ModuleNotFoundError:
            logger.warning(
                "lmcache.cuda_ops compiled extension not found; "
                "CudaDeviceOps stays on the torch baseline for all ops."
            )
            return
        except ImportError as exc:
            # The .so is present but the dynamic loader rejected it -- almost
            # always a torch/CUDA ABI mismatch between the wheel's build
            # environment and the running torch. Say so: "not found" would
            # send operators looking for a missing file instead of a version
            # skew.
            logger.warning(
                "lmcache.cuda_ops compiled extension is present but failed to "
                "load (%s); CudaDeviceOps stays on the torch baseline for all "
                "ops. Check that the wheel was built against the installed "
                "torch/CUDA versions.",
                exc,
            )
            return
        self.bind_native(native)

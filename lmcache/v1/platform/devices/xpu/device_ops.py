# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the SYCL extension on top of the torch baseline.  If the extension
is not built, a warning is logged and the instance stays on the torch
fallback.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.devices.xpu.tensor_from_ptr import (
    construct_xpu_tensor_from_data_pointer,
)

if TYPE_CHECKING:
    # Third Party
    import torch

logger = init_logger(__name__)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: "torch.dtype",
        device: "torch.device",
    ) -> "torch.Tensor":
        """Create a non-owning XPU tensor from a process-local pointer.

        The LMCache-driven transfer path passes device buffers as raw
        pointers, so the torch fallback for the block-transfer kernels can
        only run on XPU once those pointers can be turned back into tensors.
        """
        return construct_xpu_tensor_from_data_pointer(ptr, shape, dtype, device)

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError:
            logger.warning(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the "
                "torch baseline for all ops."
            )
            return
        self.bind_native(sycl)

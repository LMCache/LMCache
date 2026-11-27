# SPDX-License-Identifier: Apache-2.0
"""CPU ops backend: the torch baseline, registered under ``device_type="cpu"``.

:class:`CpuDeviceOps` owns host-pointer tensor construction and inherits the
remaining torch baseline from :class:`DeviceOps`.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import ctypes
import math

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps


class CpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cpu"

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: "torch.dtype",
        device: "torch.device",
    ) -> "torch.Tensor":
        """Create a zero-copy CPU tensor over ``ptr``."""
        del device
        total_bytes = math.prod(shape) * dtype.itemsize
        buffer_type = ctypes.c_uint8 * total_bytes
        return torch.frombuffer(buffer_type.from_address(ptr), dtype=dtype).view(*shape)

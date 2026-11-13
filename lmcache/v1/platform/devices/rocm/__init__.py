# SPDX-License-Identifier: Apache-2.0
"""ROCm platform primitives built on PyTorch's CUDA-compatible surface."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.devices.cuda import CudaDeviceSpec
from lmcache.v1.platform.ipc_policy import get_ipc_policy

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


class RocmDeviceSpec(CudaDeviceSpec):
    """ROCm device specification for the detection registry."""

    @property
    def backend_name(self) -> str:
        """Return the LMCache-specific ROCm backend selector."""
        return "rocm"

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        """Return the KV-cache IPC wrapper class; the VMM one is ROCm-specific."""
        if get_ipc_policy().use_vmm_api:
            # First Party
            from lmcache.v1.platform.devices.rocm.ipc_wrapper import (
                VmmRocmIPCWrapper,
            )

            return VmmRocmIPCWrapper
        return super().ipc_wrapper_cls

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        """Return the ROCm host-memory pinning backend."""
        # First Party
        from lmcache.v1.platform.devices.rocm.pin_memory import (
            RocmPinMemoryBackend,
        )

        return RocmPinMemoryBackend

    def is_available(self) -> bool:
        """Check ROCm availability through PyTorch's ``torch.cuda`` API."""
        try:
            # Third Party
            import torch

            return (
                torch.cuda.is_available()
                and getattr(getattr(torch, "version", None), "hip", None) is not None
            )
        except Exception:
            return False

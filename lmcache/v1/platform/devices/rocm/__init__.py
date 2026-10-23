# SPDX-License-Identifier: Apache-2.0
"""ROCm platform primitives built on PyTorch's CUDA-compatible surface."""

# First Party
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.devices.cuda import CudaDeviceSpec


class RocmDeviceSpec(CudaDeviceSpec):
    """ROCm device specification for the detection registry."""

    @property
    def backend_name(self) -> str:
        """Return the LMCache-specific ROCm backend selector."""
        return "rocm"

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

# SPDX-License-Identifier: Apache-2.0
"""ROCm platform primitives built on PyTorch's CUDA-compatible surface."""

# First Party
from lmcache.v1.platform.cuda import CudaDeviceSpec


class RocmDeviceSpec(CudaDeviceSpec):
    """ROCm device specification for the detection registry."""

    @property
    def backend_name(self) -> str:
        """Return the LMCache-specific ROCm backend selector."""
        return "rocm"

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

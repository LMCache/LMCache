# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives."""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class CudaDeviceSpec(DeviceSpec):
    """CUDA device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def torch_module_name(self) -> str:
        return "cuda"

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return CudaPinMemoryBackend

    def is_available(self) -> bool:
        """Check CUDA availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

# SPDX-License-Identifier: Apache-2.0
"""CUDA / ROCm platform primitives.

This module provides the device specification for both NVIDIA CUDA and
AMD ROCm GPUs.  On ROCm, PyTorch exposes the HIP runtime through the
``torch.cuda`` API (``torch.cuda.is_available()`` returns ``True``,
``tensor.device.type`` is ``"cuda"``), so the existing
:class:`CudaIPCWrapper` and :class:`CudaPinMemoryBackend` work on ROCm
without modification.  The ``is_rocm`` property additionally checks
``torch.version.hip`` to distinguish the two platforms for logging.
"""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class CudaDeviceSpec(DeviceSpec):
    """CUDA / ROCm device specification for the detection registry.

    Handles both NVIDIA CUDA and AMD ROCm, since PyTorch on ROCm uses
    the ``torch.cuda`` API (HIP compatibility layer).  The
    ``is_rocm`` property distinguishes the two for logging and metrics.
    """

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def torch_module_name(self) -> str:
        return "cuda"

    @property
    def ops_module(self) -> str | None:
        # lmcache.c_ops are compiled against the CUDA/HIP API and work
        # on both NVIDIA and AMD GPUs.
        return "lmcache.c_ops"

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return CudaPinMemoryBackend

    @property
    def is_rocm(self) -> bool:
        """Return True if running on AMD ROCm (HIP), False for NVIDIA CUDA."""
        try:
            # Third Party
            import torch

            return getattr(torch.version, "hip", None) is not None
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check CUDA/ROCm availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

# SPDX-License-Identifier: Apache-2.0
"""NVIDIA CUDA platform primitives.

This module provides the device specification for NVIDIA CUDA GPUs.
On ROCm, a separate :class:`RocmDeviceSpec` (in
:mod:`lmcache.v1.platform.rocm`) is selected instead, which uses
:class:`RocmPinMemoryBackend` (loading ``libamdhip64.so`` directly).

:class:`CudaIPCWrapper` is shared across both platforms because
PyTorch on ROCm uses the ``torch.cuda`` API (HIP compatibility layer),
so ``tensor.device.type == "cuda"`` and the IPC wrapper auto-discovery
selects :class:`CudaIPCWrapper` for both.

Build with ``BUILD_WITH_HIP=1`` to compile ``lmcache.c_ops`` against
the HIP runtime for ROCm.
"""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class CudaDeviceSpec(DeviceSpec):
    """NVIDIA CUDA device specification for the detection registry.

    Selected on NVIDIA systems (where ``torch.version.hip`` is ``None``).
    On AMD ROCm, :class:`RocmDeviceSpec` is selected instead.
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

    def is_available(self) -> bool:
        """Check CUDA/ROCm availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

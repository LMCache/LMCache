# SPDX-License-Identifier: Apache-2.0
"""AMD ROCm platform primitives.

On ROCm, PyTorch exposes the HIP runtime through the ``torch.cuda`` API
(``torch.cuda.is_available()`` returns ``True``, ``tensor.device.type``
is ``"cuda"``).  This means the existing :class:`CudaIPCWrapper` and
:class:`CudaPinMemoryBackend` in :mod:`lmcache.v1.platform.cuda` already
handle the common path (PyTorch-allocator-backed tensors).

This module provides a dedicated :class:`RocmDeviceSpec` so that:

* ROCm is detected **explicitly** (via ``torch.version.hip``) rather than
  silently falling through to the generic CUDA path.
* ``DEVICE_TYPE=rocm`` can be used to force selection.
* Log messages and metrics distinguish NVIDIA CUDA from AMD ROCm.
* The :class:`RocmPinMemoryBackend` loads ``libamdhip64.so`` directly
  (the CUDA backend's ``libcudart.so`` fallback does not exist on ROCm).

The ``device_type`` is ``"cuda"`` (matching what PyTorch reports on ROCm)
so the IPC-wrapper auto-discovery and tensor-device lookups are
unaffected — :class:`CudaIPCWrapper` is still selected for
``tensor.device.type == "cuda"``.  However, because the device-spec
registry keys on ``device_type`` and both ``CudaDeviceSpec`` and
``RocmDeviceSpec`` report ``"cuda"``, the registry first-wins ordering
matters.  :class:`RocmDeviceSpec` overrides :meth:`is_available` to
return ``True`` **only** when ``torch.version.hip`` is set, so on an
NVIDIA system the CUDA spec is selected and on an AMD system the ROCm
spec is selected (alphabetically ``rocm`` > ``cuda``, so ROCm wins on
AMD hardware — which is what we want).
"""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend
from lmcache.v1.platform.rocm.pin_memory import RocmPinMemoryBackend


class RocmDeviceSpec(DeviceSpec):
    """AMD ROCm device specification for the detection registry.

    Handles AMD ROCm GPUs.  On ROCm, PyTorch uses the ``torch.cuda`` API
    (HIP compatibility layer), so ``device_type`` is ``"cuda"`` to stay
    compatible with the IPC wrapper registry (which keys on
    ``tensor.device.type``).  Detection distinguishes ROCm from NVIDIA
    CUDA via ``torch.version.hip``.
    """

    @property
    def device_type(self) -> str:
        # PyTorch on ROCm reports "cuda" for tensor.device.type, so we
        # keep "cuda" here to stay compatible with the IPC wrapper
        # registry (which keys on tensor.device.type).
        return "cuda"

    @property
    def torch_module_name(self) -> str:
        return "cuda"

    @property
    def ops_module(self) -> str | None:
        # lmcache.c_ops are compiled against the HIP API when built with
        # BUILD_WITH_HIP=1 and work on AMD GPUs.
        return "lmcache.c_ops"

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return RocmPinMemoryBackend

    def is_available(self) -> bool:
        """Return True only on AMD ROCm (not NVIDIA CUDA).

        Detection: ``torch.version.hip`` is set on ROCm builds and
        ``None`` on NVIDIA builds.  This ensures the ROCm spec is only
        selected on AMD hardware, and the CUDA spec is selected on
        NVIDIA hardware.
        """
        try:
            # Third Party
            import torch

            return (
                torch.cuda.is_available()
                and getattr(torch.version, "hip", None) is not None
            )
        except Exception:
            return False

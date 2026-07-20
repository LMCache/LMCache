# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives.

Defines :class:`CpuDeviceSpec` for the device registry.  The spec's
:attr:`~CpuDeviceSpec.ipc_wrapper_cls` binds
:class:`~lmcache.v1.platform.cpu.shm.CpuShmTensorWrapper` to the
``"cpu"`` device, so the multiprocess adapter can dispatch by
``tensor.device.type`` without any if/elif chain.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper


class CpuDeviceSpec(DeviceSpec):
    """CPU device specification for the detection registry.

    Keeps ``device_type="cpu"`` aligned with the accelerator-specific
    resolution path by exposing an :attr:`ipc_wrapper_cls` binding, so
    callers dispatching on ``tensor.device.type`` never fall through
    to the bare :class:`DeviceSpec` fallback when the CPU backend is
    installed.

    Invariant:
        ``is_available()`` must inherit the base-class ``False`` (i.e.
        do **not** override it here). ``_detect_device()`` skips
        ``"cpu"`` inside its auto-detection loop to preserve the
        "cpu is the tail fallback" semantics, but that skip is
        defence-in-depth: if this class ever returned ``True`` from
        ``is_available()``, accelerators registered later in the dict
        would still be reached only because of that ``continue``. Keep
        this method inherited so the invariant holds in both layers;
        an explicit opt-in via ``DEVICE_TYPE=cpu`` is the supported
        path for forcing CPU selection.
    """

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def torch_module_name(self) -> str:
        return "cpu"

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        # First Party
        from lmcache.v1.platform.cpu.shm import CpuShmTensorWrapper

        return CpuShmTensorWrapper

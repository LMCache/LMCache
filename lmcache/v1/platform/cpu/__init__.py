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

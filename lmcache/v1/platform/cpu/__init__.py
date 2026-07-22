# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives.

Defines :class:`CpuDeviceSpec` for the device registry.  The spec's
:attr:`~CpuDeviceSpec.ipc_wrapper_cls` binds
:class:`~lmcache.v1.platform.cpu.shm.CpuShmTensorWrapper` to the
``"cpu"`` device, so the multiprocess adapter can dispatch by
``tensor.device.type`` without any if/elif chain.

:class:`CpuDeviceSpec` also participates in the ``DeviceSpec`` registry so
callers can resolve the CPU cache-context implementation via
:func:`lmcache.v1.platform.get_device_spec`. It intentionally reports
``is_available() == False`` so that ``_detect_device`` never picks it
up during auto-detection -- CPU is exclusively reached through the
``StubCPUDevice`` fallback path.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base_cache_context import BaseCacheContext
    from lmcache.v1.platform.base_device_ops import DeviceOps
    from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper


class CpuDeviceSpec(DeviceSpec):
    """CPU device specification for the detection registry.

    Keeps ``device_type="cpu"`` aligned with the accelerator-specific
    resolution path by exposing an :attr:`ipc_wrapper_cls` binding, so
    callers dispatching on ``tensor.device.type`` never fall through
    to the bare :class:`DeviceSpec` fallback when the CPU backend is
    installed.

    Also used for ``get_device_spec("cpu")`` lookups (e.g. by
    :func:`lmcache.v1.platform.cache_context.create_cache_context`).
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

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        # First Party
        from lmcache.v1.platform.cpu.cache_context import CPUCacheContext

        return CPUCacheContext(*args, **kwargs)

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.cpu.device_ops import CpuDeviceOps

        return CpuDeviceOps

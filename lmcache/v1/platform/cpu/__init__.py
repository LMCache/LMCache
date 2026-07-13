# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives.

:class:`~lmcache.v1.platform.cpu.shm.CpuShmTensorWrapper` carries a
``device_type`` ClassVar and a ``wrap`` factory classmethod, which
:func:`~lmcache.v1.platform._registry._discover_wrappers_once` picks
up at run-time -- no static ``register_kv_wrapper`` needed.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.platform.base_device_ops import DeviceOps
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.cpu.device_ops import CpuDeviceOps


class CpuDeviceSpec(DeviceSpec):
    """CPU device specification for the detection registry.

    This keeps CPU aligned with the accelerator-specific resolution path:
    callers asking for ``device_type="cpu"`` receive a concrete
    :class:`CpuDeviceOps` class through :attr:`DeviceSpec.ops_cls`, while the
    bare :class:`DeviceSpec` remains available as the fallback for
    ``device_type=""`` or deliberately stripped test registries.
    """

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def torch_module_name(self) -> str:
        return "cpu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        return CpuDeviceOps

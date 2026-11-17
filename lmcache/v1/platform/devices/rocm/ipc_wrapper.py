# SPDX-License-Identifier: Apache-2.0
"""ROCm IPC wrapper implementations.

The CUDA wrappers that go through PyTorch also serve ROCm. The VMM wrapper
calls the driver API directly, so ROCm has its own subclass that issues the
same calls through ``rocm.bindings``.
"""

# Future
from __future__ import annotations

# Standard
from types import ModuleType

# First Party
from lmcache.v1.platform.base.ipc_wire import VmmRocmIPCData
from lmcache.v1.platform.devices.cuda.ipc_wrapper import VmmCudaIPCWrapper


class VmmRocmIPCWrapper(VmmCudaIPCWrapper):
    """:class:`VmmCudaIPCWrapper` on the HIP VMM API (``rocm.bindings``)."""

    _wire_cls = VmmRocmIPCData

    @staticmethod
    def _driver() -> ModuleType:
        """Return the HIP calls under the ``cuda.bindings`` names the wrapper uses."""
        # rocm.bindings is only installed on ROCm; descriptors decode anywhere.
        # First Party
        from lmcache.v1.platform.devices.rocm import vmm_bindings

        return vmm_bindings

    @staticmethod
    def _bind_context(device_index: int) -> None:
        """No-op: HIP binds contexts implicitly to the current device."""

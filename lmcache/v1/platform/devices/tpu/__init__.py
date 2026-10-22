# SPDX-License-Identifier: Apache-2.0
"""Google TPU (TorchTPU) platform helpers.

Targets TorchTPU (``torch_tpu``), which registers ``torch.device("tpu")`` via
PyTorch's ``rename_privateuse1_backend`` mechanism, together with the
``torch.tpu`` device module.

Scope: **engine-driven** multiprocess transfer only. A TorchTPU tensor's
``data_ptr()`` addresses host-side buffer metadata rather than HBM, so neither
pointer-based kernels nor the LMCache-driven IPC handle path can address it;
:class:`~lmcache.v1.platform.devices.tpu.device_ops.TpuDeviceOps` moves whole
blocks with tensor ops through host staging instead.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps


class TpuDeviceSpec(DeviceSpec):
    """TorchTPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "tpu"

    @property
    def torch_module_name(self) -> str:
        return "tpu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.devices.tpu.device_ops import TpuDeviceOps

        return TpuDeviceOps

    def is_available(self) -> bool:
        """Check TorchTPU availability.

        Imports ``torch_tpu``, which registers the ``tpu`` device on a TPU
        host unless PyTorch's device-backend autoload already has. Availability
        is then read through ``torch.accelerator`` rather than
        ``torch.tpu.is_available()``: on PyTorch >= 2.11 TorchTPU reserves the
        ``torch.tpu`` device-module functions for the framework and raises
        ``RuntimeError`` when other packages call them.

        Returns:
            bool: ``True`` when ``torch_tpu`` imports and the current
            accelerator is an available TPU, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch_tpu  # noqa: F401
        except ImportError:
            return False
        # Third Party
        import torch

        accelerator = torch.accelerator.current_accelerator(check_available=True)
        return accelerator is not None and accelerator.type == "tpu"

    def is_lmcache_driven_available(self) -> bool:
        """Report that TorchTPU cannot serve the LMCache-driven path.

        That path ships KV tensors to the server as IPC handles to device
        memory, and a TorchTPU tensor exposes no address of its HBM buffer.

        Returns:
            bool: Always ``False``.
        """
        return False

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
from typing import TYPE_CHECKING, Any

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_ops import DeviceOps


def _tpu_is_available() -> bool:
    """``torch.tpu.is_available()`` through ``torch.accelerator``."""
    return torch.accelerator.is_available()


def _tpu_device_count() -> int:
    """``torch.tpu.device_count()`` through ``torch.accelerator``."""
    return torch.accelerator.device_count()


def _tpu_synchronize(device: int | str | torch.device | None = None) -> None:
    """``torch.tpu.synchronize()`` through ``torch.accelerator``."""
    torch.accelerator.synchronize(device)


def _tpu_current_device() -> int:
    """``torch.tpu.current_device()`` through ``torch.accelerator``."""
    return torch.accelerator.current_device_index()


def _tpu_set_device(device: int | str | torch.device) -> None:
    """``torch.tpu.set_device()`` through ``torch.accelerator``."""
    torch.accelerator.set_device_index(device)


def _tpu_current_stream(
    device: int | str | torch.device | None = None,
) -> torch.Stream:
    """``torch.tpu.current_stream()`` through ``torch.accelerator``."""
    return torch.accelerator.current_stream(device)


# On PyTorch >= 2.11 TorchTPU raises RuntimeError when code outside torch calls
# these on ``torch.tpu`` and points to ``torch.accelerator``, which reaches the
# same ``TpuDeviceGuardImpl``; so unlike Neuron's shims, these replace the
# module's own functions.
_DEVICE_MODULE_SHIMS = {
    "is_available": _tpu_is_available,
    "device_count": _tpu_device_count,
    "synchronize": _tpu_synchronize,
    "current_device": _tpu_current_device,
    "set_device": _tpu_set_device,
    "current_stream": _tpu_current_stream,
}


def _install_device_module_shims(torch_tpu_module: Any) -> None:
    """Point ``torch.tpu``'s restricted functions at ``torch.accelerator``.

    Idempotent: installing again sets the same functions.
    """
    for name, impl in _DEVICE_MODULE_SHIMS.items():
        setattr(torch_tpu_module, name, impl)


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
        ``RuntimeError`` when other packages call them. Once a TPU is found,
        also points those ``torch.tpu`` functions at ``torch.accelerator``, so
        that the generic ``torch_dev.<method>()`` call sites throughout LMCache
        work on TPU without each one needing a TPU-specific branch.

        Returns:
            bool: ``True`` when ``torch_tpu`` imports and the current
            accelerator is an available TPU, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch_tpu  # noqa: F401
        except ImportError:
            return False

        accelerator = torch.accelerator.current_accelerator(check_available=True)
        if accelerator is None or accelerator.type != "tpu":
            return False
        _install_device_module_shims(torch.tpu)  # type: ignore[attr-defined]
        return True

    def is_lmcache_driven_available(self) -> bool:
        """Report that TorchTPU cannot serve the LMCache-driven path.

        That path ships KV tensors to the server as IPC handles to device
        memory, and a TorchTPU tensor exposes no address of its HBM buffer.

        Returns:
            bool: Always ``False``.
        """
        return False

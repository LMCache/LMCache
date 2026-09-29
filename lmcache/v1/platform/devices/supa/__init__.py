# SPDX-License-Identifier: Apache-2.0
"""Biren SUPA platform helpers.

Biren GPUs are exposed to PyTorch by the ``torch_br`` package, which bridges
PyTorch's ``PrivateUse1`` backend to Biren's SUPA runtime via
``torch.utils.rename_privateuse1_backend("supa")`` and registers
``torch.supa``.  Importing ``torch_br`` therefore makes ``torch.device("supa")``
available process-wide; on hosts without the runtime the import fails and this
spec reports unavailable.

``torch.supa`` exposes a CUDA-like surface (``Stream``, ``Event`` with
inter-process handles, ``synchronize``, ``current_stream`` and memory stats),
so this backend reuses the torch baseline :class:`DeviceOps` unchanged, like
the HPU and Neuron integrations.

Scope: engine-driven multiprocess (MP) transfer only for now.  The
LMCache-driven path additionally needs a KV IPC handle wrapper and a
cross-process event IPC backend.  SUPA ``Event`` IPC exists, but tensor IPC
sharing is not wired up yet, so :meth:`SupaDeviceSpec.is_handle_transfer_available`
returns ``False`` and ``mp_transfer_mode=lmcache_driven`` fails at its
documented validation point instead of crashing deeper in the transfer path.

See ``docs/design/v1/platform/devices/supa/README.md`` for the full contract.
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

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class SupaDeviceSpec(DeviceSpec):
    """Biren SUPA device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "supa"

    @property
    def torch_module_name(self) -> str:
        return "supa"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.devices.supa.device_ops import SupaDeviceOps

        return SupaDeviceOps

    def is_available(self) -> bool:
        """Check SUPA availability without importing ``lmcache.__init__``.

        Imports ``torch_br`` to trigger the ``PrivateUse1`` -> ``supa`` rename
        that registers ``torch.supa``; without it the device module does not
        exist on ``torch`` and detection silently fails.  Hardware probing is
        confined here: a vendor-runtime error (for example a driver that
        cannot hand out a device to a co-tenant process) degrades to
        "unavailable" instead of aborting import.

        Returns:
            bool: ``True`` when ``torch.supa`` is present and reports at least
            one usable device, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch

            if not hasattr(torch, "supa"):
                # Third Party
                import torch_br  # noqa: F401 — side-effect: registers torch.supa

            return hasattr(torch, "supa") and torch.supa.is_available()
        except Exception:
            return False

    def is_handle_transfer_available(self) -> bool:
        """Report that SUPA cannot yet ship KV tensors as IPC handles.

        The base class defaults to ``True``; SUPA overrides it to ``False``
        until a KV IPC wrapper and cross-process event IPC backend are
        implemented, so ``mp_transfer_mode=lmcache_driven`` fails at its
        documented validation point.

        Returns:
            bool: Always ``False``.
        """
        return False

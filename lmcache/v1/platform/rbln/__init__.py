# SPDX-License-Identifier: Apache-2.0
"""RBLN (Rebellions NPU) platform primitives.

Registers :class:`RblnDeviceSpec` with the device-detection registry so
LMCache resolves ``torch.rbln`` as an accelerator instead of falling back
to the CPU stub.  ``torch.rbln`` is contributed by the ``torch_rbln``
package through a torch backend entry point, so it is visible on a bare
``import torch`` -- no explicit ``import torch_rbln`` is required here.

Scope: **engine-driven** multiprocess (MP) transfer only.  ``torch.rbln``
exposes device discovery and ``synchronize()`` but no ``Stream`` / ``Event``
types, so the LMCache-driven path (which needs cross-process event IPC and
an IPC handle wrapper) cannot be supported.  The spec therefore reports
:meth:`RblnDeviceSpec.is_handle_transfer_available` as ``False`` and leaves
``ipc_wrapper_cls`` / ``event_ipc_backend`` at their ``None`` defaults, so
requesting ``mp_transfer_mode=lmcache_driven`` fails fast with a clear
error rather than crashing deeper in the transfer path.

See ``docs/design/v1/platform/rbln/README.md`` for the full contract.
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


class RblnDeviceSpec(DeviceSpec):
    """RBLN device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "rbln"

    @property
    def torch_module_name(self) -> str:
        return "rbln"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps

        return RblnDeviceOps

    def is_available(self) -> bool:
        """Check RBLN availability without importing ``lmcache.__init__``.

        ``torch.rbln.is_available()`` raises (rather than returning
        ``False``) when the runtime cannot register a physical NPU -- for
        example when another process already holds the device or a stale
        allocation survives.  Detection runs on every LMCache start,
        including on hosts with no free NPU, so the exception is swallowed
        and reported as "unavailable"; letting it escape would abort import
        for every co-tenant process on the box.

        Returns:
            bool: ``True`` when ``torch.rbln`` is present and reports at
            least one usable device, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch

            return hasattr(torch, "rbln") and torch.rbln.is_available()
        except Exception:
            return False

    def is_handle_transfer_available(self) -> bool:
        """Report that RBLN cannot ship KV tensors as IPC handles.

        The base class defaults to ``True``; RBLN overrides it to ``False``
        because ``torch.rbln`` exposes no ``Event`` type, so the ordered
        cross-process publication the LMCache-driven path depends on cannot
        be expressed.  Returning ``False`` keeps
        ``mp_transfer_mode=lmcache_driven`` failing at its documented
        validation point instead of at an attribute lookup later on.

        Returns:
            bool: Always ``False``.
        """
        return False

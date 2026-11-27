# SPDX-License-Identifier: Apache-2.0
"""Enflame GCU platform primitives.

Registers :class:`GcuDeviceSpec` so LMCache resolves ``torch.gcu`` as an
accelerator instead of falling back to the CPU stub. The compiled ops surface
is assembled from the ``triton_kernel_gcu`` plugin's
``torch.ops.TRITON_GCU.lmcache`` operators; host-memory pinning uses
the GCU runtime (``topsHostRegister``); cross-process KV-cache sharing uses
``UntypedStorage._share_gcu_`` / ``_new_shared_gcu``.

Scope mirrors the CUDA backend: this package only wires device detection and
backend selection. The Triton kernels live in ``triton_kernel_gcu`` (imported
lazily by :mod:`lmcache.v1.platform.devices.gcu.device_ops`).
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.devices.gcu.pin_memory import GcuPinMemoryBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class GcuDeviceSpec(DeviceSpec):
    """Enflame GCU device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "gcu"

    @property
    def torch_module_name(self) -> str:
        return "gcu"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.devices.gcu.device_ops import GcuDeviceOps

        return GcuDeviceOps

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return GcuPinMemoryBackend

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        # First Party
        from lmcache.v1.platform.devices.gcu.ipc_wrapper import GcuIPCWrapper

        return GcuIPCWrapper

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the GCU event IPC backend.

        GCU's ``Event`` supports CUDA-style interprocess handles but its C
        extension kwlist is invisible to :func:`inspect.signature`, so the
        backend probes by direct construction instead of relying on the
        generic capability check.
        """
        backend = self._event_backend_cache
        if backend is None:
            # First Party
            from lmcache.v1.platform.devices.gcu.event_ipc import GcuEventIPCBackend

            backend = GcuEventIPCBackend()
            self._event_backend_cache = backend
        return backend

    def is_available(self) -> bool:
        """Check GCU availability without importing ``lmcache.__init__``.

        Returns:
            bool: ``True`` when ``torch.gcu`` is present and reports at least
            one usable device, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch

            return hasattr(torch, "gcu") and torch.gcu.is_available()
        except Exception:
            return False

    def is_lmcache_driven_available(self) -> bool:
        """GCU has the full IPC stack for LMCache-driven transfer.

        ``GcuIPCWrapper`` (handle-based KV sharing via
        ``UntypedStorage._share_gcu_`` / ``_new_shared_gcu``), the GCU event
        IPC backend, and the GCU cache context are all wired, so the
        lmcache-driven (IPC / SHM zero-copy) data path is available.
        """
        return True

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Instantiate the GCU cache context.

        Forwards ``*args`` / ``**kwargs`` verbatim so the call site in
        :func:`lmcache.v1.platform.cache_context.create_cache_context` stays
        backend-agnostic.
        """
        # First Party
        from lmcache.v1.platform.devices.gcu.cache_context import GcuCacheContext

        return GcuCacheContext(*args, **kwargs)

# SPDX-License-Identifier: Apache-2.0
"""XPU (Intel SYCL) platform helpers."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.devices.xpu.device_ops import XpuDeviceOps

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class XpuDeviceSpec(DeviceSpec):
    """XPU device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "xpu"

    @property
    def torch_module_name(self) -> str:
        return "xpu"

    def get_stream_handle(self, stream: object) -> int:
        """Return the SYCL queue pointer backing ``stream``.

        ``torch.xpu.Stream`` exposes the underlying ``sycl::queue *`` as
        ``sycl_queue``, the XPU analogue of CUDA's ``cuda_stream``.
        """
        stream_handle = getattr(stream, "sycl_queue", None)
        if stream_handle is None:
            raise RuntimeError("XPU stream does not expose a sycl_queue.")
        return int(stream_handle)

    @property
    def ops_cls(self) -> type[DeviceOps]:
        return XpuDeviceOps

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        """Return the SYCL host-memory backend when it can initialize."""
        try:
            # First Party
            from lmcache.v1.platform.devices.xpu.pin_memory import (
                XpuPinMemoryBackend,
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("XPU pin-memory backend is unavailable: %s", exc)
            return None

        if XpuPinMemoryBackend.is_available():
            return XpuPinMemoryBackend
        return None

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        """Return the torch-storage-IPC wrapper for XPU KV tensors."""
        # First Party
        from lmcache.v1.platform.devices.xpu.ipc_wrapper import XpuIPCWrapper

        return XpuIPCWrapper

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the torch XPU event IPC backend."""
        backend = self._event_backend_cache
        if backend is None:
            # First Party
            from lmcache.v1.platform.devices.xpu.event_ipc import (
                XpuEventIPCBackend,
            )

            backend = XpuEventIPCBackend()
            self._event_backend_cache = backend
        return backend

    def is_available(self) -> bool:
        """Check XPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Create the XPU cache context for LMCache-driven transfer."""
        # First Party
        from lmcache.v1.platform.devices.xpu.cache_context import XPUCacheContext

        return XPUCacheContext(*args, **kwargs)

    def is_lmcache_driven_available(self) -> bool:
        """Return whether the XPU lmcache-driven path may be selected.

        ``True`` only when ``LMCACHE_XPU_HANDLE_TRANSFER`` opts in and the
        torch build provides XPU storage and event IPC. The inherited
        :meth:`default_mp_transfer_mode` derives from this, so ``auto``
        selects lmcache-driven exactly when it returns ``True`` and
        engine-driven otherwise.
        """
        # First Party
        from lmcache.v1.platform.devices.xpu.ipc_wrapper import (
            is_xpu_handle_transfer_available,
        )

        return is_xpu_handle_transfer_available()

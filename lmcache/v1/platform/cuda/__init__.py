# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend
from lmcache.v1.platform.cuda.vmm_ipc import is_use_vmm_api
from lmcache.v1.platform.isolated_ipc import is_isolated_ipc

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

# ---------------------------------------------------------------------------
# Event IPC backend selection
# ---------------------------------------------------------------------------


def _select_event_ipc_backend(device_type: str) -> "EventIPCBackend":
    """Construct the event IPC backend for the current isolated-IPC setting.

    Args:
        device_type: Device-type label passed through to the backend.

    Returns:
        The timeline-semaphore backend when isolated IPC is enabled (see
        ``lmcache/v1/platform/isolated_ipc.py``), otherwise the CUDA
        interprocess event handle backend.
    """
    if is_isolated_ipc():
        # First Party
        from lmcache.v1.platform.cuda.timeline_semaphore_event_ipc import (
            TimelineSemaphoreEventIPCBackend,
        )

        return TimelineSemaphoreEventIPCBackend()

    # Third Party
    import torch

    # First Party
    from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend

    return DefaultEventIPCBackend(
        event_module=torch.cuda,
        device_type=device_type,
    )


def _select_ipc_wrapper_cls() -> "type[DeviceIPCWrapper]":
    """Return the KV-cache IPC wrapper class for the current switches.

    Three modes, one wrapper each:

    - ``use_vmm_api`` on: :class:`VmmCudaIPCWrapper` (the engine
      allocates KV through the CUDA VMM API; legacy IPC handles do not
      exist for such memory). Composes with ``isolated_ipc``: the
      fabric kind is isolation-clean (inline blob, IMEX channel device
      injection, no shared filesystem), while a POSIX-fd allocation is
      rejected at wrap time -- fd passing needs a shared path, which
      the zero-share isolated model rules out.
    - ``isolated_ipc`` alone: :class:`RawCudaIPCWrapper` (driver-level
      CUDA IPC mem handles, no shared ``/dev/shm`` assumed).
    - default: :class:`CudaIPCWrapper` (PyTorch storage IPC).

    Returns:
        The wrapper class for the current switch settings.
    """

    # First Party
    from lmcache.v1.platform.cuda.ipc_wrapper import (
        CudaIPCWrapper,
        RawCudaIPCWrapper,
        VmmCudaIPCWrapper,
    )

    if is_use_vmm_api():
        return VmmCudaIPCWrapper
    return RawCudaIPCWrapper if is_isolated_ipc() else CudaIPCWrapper


# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class CudaDeviceSpec(DeviceSpec):
    """CUDA device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def torch_module_name(self) -> str:
        return "cuda"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.cuda.device_ops import CudaDeviceOps

        return CudaDeviceOps

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the CUDA event IPC backend."""
        backend = self._event_backend_cache
        if backend is None:
            backend = _select_event_ipc_backend(self.device_type)
            self._event_backend_cache = backend
        return backend

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return CudaPinMemoryBackend

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        """Return the KV-cache IPC wrapper class (isolated-IPC aware)."""
        return _select_ipc_wrapper_cls()

    def is_available(self) -> bool:
        """Check CUDA availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            torch_version = getattr(torch, "version", None)
            if torch_version is None:
                return torch.cuda.is_available()
            return (
                torch.cuda.is_available()
                and getattr(torch_version, "cuda", None) is not None
            )
        except Exception:
            return False

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        # First Party
        from lmcache.v1.platform.cuda.cache_context import GPUCacheContext

        return GPUCacheContext(*args, **kwargs)

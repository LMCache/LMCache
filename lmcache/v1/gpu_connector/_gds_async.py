# SPDX-License-Identifier: Apache-2.0
"""Extensible dispatch shim for async GPUDirect Storage backends.

Backend modules register themselves by defining a concrete
:class:`GDSAsyncBackend` subclass. This shim discovers those backend classes,
selects the configured implementation, and re-exports the selected module's
common async IO surface under stable names for
:mod:`lmcache.v1.gpu_connector.gds_context`.

Callers must import this module rather than individual symbols, which would
retain backend bindings captured before :func:`select_backend` runs.
"""

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.gpu_connector._gds_backend import (
    AUTO_BACKEND_NAME,
    BackendName,
    GDSAsyncBackend,
    GDSBackendRegistry,
)

_registry = GDSBackendRegistry.discover(__package__)
_selected_backend: GDSAsyncBackend = _registry.default_backend()
_selection_finalized = False

# The backend surface re-exported under stable module-level names so callers
# and test monkeypatches target this module.
_EXPORTED_NAMES = (
    "AsyncHandle",
    "Submission",
    "close_driver",
    "register_handle",
    "deregister_handle",
    "register_buffer",
    "deregister_buffer",
    "register_stream",
    "deregister_stream",
)


def _bind_backend_surface(backend: GDSAsyncBackend) -> None:
    """Rebind every exported name to the given backend module."""
    backend.bind_surface(globals(), _EXPORTED_NAMES)


if TYPE_CHECKING:
    AsyncHandle: Any
    Submission: Any

    def close_driver() -> None:
        """Close the selected backend driver."""
        ...

    def register_handle(fd: int) -> Any:
        """Register an open file descriptor with the selected backend."""
        ...

    def deregister_handle(handle: Any) -> None:
        """Deregister a backend file handle."""
        ...

    def register_buffer(buf: Any) -> None:
        """Register a device tensor with the selected backend."""
        ...

    def deregister_buffer(buf: Any) -> None:
        """Deregister a device tensor from the selected backend."""
        ...

    def register_stream(raw_stream: int) -> None:
        """Register a raw GPU stream with the selected backend."""
        ...

    def deregister_stream(raw_stream: int) -> None:
        """Deregister a raw GPU stream from the selected backend."""
        ...

else:
    _bind_backend_surface(_selected_backend)


def select_backend(name: BackendName) -> str:
    """Select the process-global GDS L1 implementation.

    Args:
        name: Explicit backend name, or ``auto`` for platform selection.

    Returns:
        The resolved backend name.

    Raises:
        ValueError: If the backend is unknown or incompatible with PyTorch.
        RuntimeError: If a different backend was already selected.
    """
    global _selected_backend
    global _selection_finalized

    selected_backend = _registry.select(name)
    if _selection_finalized and selected_backend.name != _selected_backend.name:
        raise RuntimeError(
            f"GDS backend already selected as {_selected_backend.name}; "
            f"cannot switch to {selected_backend.name}"
        )
    _selected_backend = selected_backend
    _bind_backend_surface(_selected_backend)
    _selection_finalized = True
    return _selected_backend.name


def get_device_capacity(fd: int, handle: int) -> int:
    """Return the selected backend's finite slab-device capacity.

    Args:
        fd: Open slab-device descriptor.
        handle: Registered backend handle whose namespace to query.

    Returns:
        Usable backing-device capacity in bytes.

    Raises:
        RuntimeError: If the selected backend does not expose a capacity query.
    """
    return _selected_backend.get_device_capacity(fd, handle)


__all__ = (
    "AUTO_BACKEND_NAME",
    "BackendName",
    "AsyncHandle",
    "Submission",
    "close_driver",
    "register_handle",
    "deregister_handle",
    "register_buffer",
    "deregister_buffer",
    "register_stream",
    "deregister_stream",
    "select_backend",
    "get_device_capacity",
)

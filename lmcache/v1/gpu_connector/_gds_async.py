# SPDX-License-Identifier: Apache-2.0
"""Platform dispatch for the GDS async backend.

Selects the GDSContext-facing backend from cuFile, hipFile, or uGDS. Automatic
selection chooses cuFile on NVIDIA and hipFile on AMD ROCm; GDS L1 config can
explicitly select uGDS for a raw ``/dev/ugds_drvX`` slab. uGDS can be used on
either platform with a matching platform-specific build. All implementations
expose an identical API -- :class:`AsyncHandle`, :class:`Submission`, and the
``register_*`` / ``deregister_*`` / ``close_driver`` functions -- so
:mod:`lmcache.v1.gpu_connector.gds_context` remains backend-agnostic.
The uGDS-only :func:`get_ugds_device_capacity` helper is intentionally separate
from that common backend surface.

Selection is by ``torch.version.hip``: a ROCm torch build reports a non-None
HIP version. Importing this shim does not dlopen any GPU IO driver; the selected
backend binds its native library lazily on first use.

Callers must import this module rather than individual symbols, which would
retain the backend bindings captured before :func:`select_backend` runs.
"""

# Standard
from types import ModuleType
from typing import TYPE_CHECKING, Literal

# Third Party
import torch

BackendName = Literal["auto", "cufile", "hipfile", "ugds"]
_backend: ModuleType
_selected_backend: str
_selection_finalized = False

# The backend surface re-exported under stable module-level names so callers
# (and test monkeypatches) target this module.
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


def _load_backend(name: BackendName) -> tuple[str, ModuleType]:
    selected = name
    backend: ModuleType
    if selected == "auto":
        selected = "hipfile" if torch.version.hip is not None else "cufile"
    if selected == "cufile":
        # First Party
        from lmcache.v1.gpu_connector import _cufile_async

        backend = _cufile_async
    elif selected == "hipfile":
        # First Party
        from lmcache.v1.gpu_connector import _hipfile_async

        backend = _hipfile_async
    elif selected == "ugds":
        # First Party
        from lmcache.v1.gpu_connector import _ugds_async

        backend = _ugds_async
    else:
        raise ValueError(f"unsupported GDS L1 backend: {name}")
    return selected, backend


def _bind_backend_surface(backend: ModuleType) -> None:
    """Rebind every exported name to the given backend module."""
    for name in _EXPORTED_NAMES:
        globals()[name] = getattr(backend, name)


def _validate_backend_platform(selected: str) -> None:
    """Reject GDS backends incompatible with the installed PyTorch build."""
    if selected == "hipfile":
        if torch.version.hip is None:
            raise ValueError("hipfile requires a ROCm PyTorch build")
    elif selected == "cufile" and torch.version.cuda is None:
        raise ValueError(f"{selected} requires a CUDA PyTorch build")
    elif (
        selected == "ugds" and torch.version.hip is None and torch.version.cuda is None
    ):
        raise ValueError(f"{selected} requires a ROCm or CUDA PyTorch build")


if TYPE_CHECKING:
    # Static surface for type checkers; every backend exposes the same names.
    # First Party
    from lmcache.v1.gpu_connector._cufile_async import AsyncHandle as AsyncHandle
    from lmcache.v1.gpu_connector._cufile_async import Submission as Submission
    from lmcache.v1.gpu_connector._cufile_async import close_driver as close_driver
    from lmcache.v1.gpu_connector._cufile_async import (
        deregister_buffer as deregister_buffer,
    )
    from lmcache.v1.gpu_connector._cufile_async import (
        deregister_handle as deregister_handle,
    )
    from lmcache.v1.gpu_connector._cufile_async import (
        deregister_stream as deregister_stream,
    )
    from lmcache.v1.gpu_connector._cufile_async import (
        register_buffer as register_buffer,
    )
    from lmcache.v1.gpu_connector._cufile_async import (
        register_handle as register_handle,
    )
    from lmcache.v1.gpu_connector._cufile_async import (
        register_stream as register_stream,
    )
else:
    _selected_backend, _backend = _load_backend("auto")
    _bind_backend_surface(_backend)


def select_backend(name: BackendName) -> str:
    """Select the process-global GDS L1 implementation.

    Args:
        name: Explicit backend name, or ``auto`` for platform selection.

    Returns:
        The resolved backend name.

    Raises:
        ValueError: If the backend is incompatible with the PyTorch build.
        RuntimeError: If a different backend was already selected.
    """
    global _backend
    global _selected_backend
    global _selection_finalized

    selected, backend = _load_backend(name)
    if _selection_finalized and selected != _selected_backend:
        raise RuntimeError(
            f"GDS backend already selected as {_selected_backend}; "
            f"cannot switch to {selected}"
        )
    _validate_backend_platform(selected)
    _selected_backend, _backend = selected, backend
    _bind_backend_surface(_backend)
    _selection_finalized = True
    return _selected_backend


def get_ugds_device_capacity(fd: int, handle: int) -> int:
    """Return the capacity reported by the selected uGDS backend.

    Args:
        fd: Open uGDS character-device descriptor.
        handle: Registered ``uGDSHandle_t`` whose namespace to query.

    Returns:
        Usable NVMe namespace capacity in bytes.

    Raises:
        RuntimeError: If uGDS is not the selected backend or its capacity query
            fails.
    """
    if _selected_backend != "ugds":
        raise RuntimeError(
            "get_ugds_device_capacity requires the selected GDS backend to be 'ugds'"
        )
    return _backend.get_device_capacity(fd, handle)

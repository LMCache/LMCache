# SPDX-License-Identifier: Apache-2.0
"""Device detection and backend selection for LMCache.

This module detects the available hardware accelerator and exposes
:data:`torch_dev` and :data:`torch_device_type` as module-level
singletons.  It also provides :func:`get_backend` which is called by
``lmcache.__init__`` to select the appropriate ops backend.

Detection and backend selection are **registry-driven**: each platform
sub-package (``platform/cuda``, ``platform/musa``, ...) defines a
concrete :class:`~lmcache.v1.platform.base_device_info.DeviceInfo`
subclass in its ``__init__.py``.  This module auto-discovers those
subclasses via ``pkgutil.iter_modules``, instantiates them, and uses
the resulting objects for detection and backend selection.

The detection order is determined by ``pkgutil.iter_modules`` scan
order (alphabetical by sub-package name).  If a specific ordering is
needed, name the sub-packages accordingly (e.g. prefix with a digit).

Adding a new accelerator (e.g. MLU) requires **zero** edits to this
module -- just drop a new ``platform/<backend>/`` package with a
:class:`~lmcache.v1.platform.base_device_info.DeviceInfo` subclass.

This module is intentionally a leaf module -- it does NOT import from
``lmcache.__init__`` or ``lmcache.v1.platform.__init__`` to avoid
circular dependencies.
"""

# Standard
from typing import Any
import importlib
import types

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_info import DeviceInfo
from lmcache.v1.utils.subclass_discovery import discover_subclasses

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Device info registry
# ---------------------------------------------------------------------------

# the device order now is: musa > xpu > hpu > cuda > cpu
_DEVICE_REGISTRY: dict[str, DeviceInfo] = {
    info.device_type: info
    for info in sorted(
        (
            cls()
            for cls in discover_subclasses(
                "lmcache.v1.platform",
                DeviceInfo,  # type: ignore[type-abstract]
                module_filter=lambda name: not name.startswith(("_", "base")),
                require_defined_in_module=True,
                on_import_error=lambda name, exc: None,
            )
        ),
        key=lambda info: info.priority,
    )
}

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _detect_device() -> tuple[Any, str]:
    """Detect the available accelerator via the device registry.

    Returns:
        tuple[Any, str]: A tuple of (torch_device_module, device_type_string).
            When torch is not installed (CLI-only mode), returns
            ``(None, "cpu")``.
    """
    try:
        # Third Party
        import torch
    except ImportError as e:
        logger.warning("load torch failed, error is %s", e)
        return None, "cpu"  # fallback for CLI-only environments

    for info in _DEVICE_REGISTRY.values():
        if not info.is_available():
            continue

        torch_module = getattr(torch, info.torch_module_name, None)
        if torch_module is not None:
            return torch_module, info.device_type
        else:
            logger.warning(
                "device [%s] is available, but torch module [%s] is not found.",
                info.device_type,
                info.torch_module_name,
            )

    # No accelerator found -- fall back to CPU stub
    # First Party
    from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice

    return StubCPUDevice("cpu"), "cpu"


# ---------------------------------------------------------------------------
# Dynamic backend selection
# ---------------------------------------------------------------------------


def get_backend(device_type: str) -> Any | None:
    """Select the ops backend for the given device type.

    Looks up the :class:`DeviceInfo` for *device_type* in the registry
    and loads/merges its ops module on top of the Python fallback.

    Args:
        device_type: The detected device type string (e.g. ``"cuda"``).

    Returns:
        A merged :class:`types.ModuleType` (fallback + hw-specific ops),
        or ``None`` if torch / dependencies are unavailable.
    """
    try:
        # Third Party
        import torch  # noqa: F401
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("load torch failed, error is %s", e)
        return None

    try:
        default_module = importlib.import_module("lmcache.python_ops_fallback")
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning("Cannot load python_ops_fallback: %s", e)
        return None

    info = _DEVICE_REGISTRY.get(device_type)
    if info is None:
        logger.info("No DeviceInfo registered for %r, using fallback ops.", device_type)
        return default_module

    if not info.is_available():
        logger.warning("Device %s is not available, using fallback ops.", device_type)
        return default_module

    if not info.ops_module:
        # Device has no custom ops -- use fallback
        logger.info(
            "Custom ops not supported for device: %s, using fallback ops.", device_type
        )
        return default_module

    try:
        backend_module = importlib.import_module(info.ops_module)
        merged_module = types.ModuleType("lmcache.c_ops")
        merged_module.__dict__.update(default_module.__dict__)
        merged_module.__dict__.update(backend_module.__dict__)
        logger.info("Using backend: %s", info.ops_module)
        return merged_module
    except Exception as e:
        logger.warning("Failed to import backend %s: %s", info.ops_module, e)

    return default_module


torch_dev, torch_device_type = _detect_device()

logger.info("torch_dev=%s, torch_device_type=%s", torch_dev, torch_device_type)

# Attach the DeviceExt instance as ``torch_dev.ext``.
if torch_dev is not None:
    # First Party
    from lmcache.v1.platform.device_ext import DeviceExt

    torch_dev.ext = DeviceExt(torch_device_type)  # type: ignore[attr-defined]
else:
    logger.warning("torch_dev is None, skipping DeviceExt initialization.")

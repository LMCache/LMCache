# SPDX-License-Identifier: Apache-2.0
"""Cross-platform abstraction layer for LMCache.

This package centralizes platform-specific primitives. It currently
exposes :class:`EventNotifier` -- a thin wake-up primitive used to
signal background loops from other threads.  On Linux it is backed by
``os.eventfd``; on macOS / other POSIX systems it falls back to
``os.pipe``.  Callers never touch ``os.eventfd`` directly.

Accelerator- and OS-specific implementations live in dedicated sub-
packages so each can evolve independently:

* :mod:`lmcache.v1.platform.cuda` -- CUDA-backed implementations.
* :mod:`lmcache.v1.platform.cpu`  -- CPU-only fallbacks.

KV-cache IPC wrappers and ``BaseCacheContext`` subclasses are
discovered separately on first use via
:mod:`lmcache.v1.utils.subclass_discovery`, keyed by each subclass'
``device_type`` ClassVar.  Adding a new accelerator therefore
requires *zero* edits to this module -- drop a new
``platform/<backend>/`` package and it will be picked up
automatically.
"""

# Standard
from typing import Any
import importlib
import os
import types

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_info import DeviceInfo
from lmcache.v1.platform.event_notifier import HAS_EVENTFD as HAS_EVENTFD
from lmcache.v1.platform.event_notifier import EventfdNotifier as EventfdNotifier
from lmcache.v1.platform.event_notifier import EventNotifier as EventNotifier
from lmcache.v1.platform.event_notifier import PipeNotifier as PipeNotifier
from lmcache.v1.platform.event_notifier import consume_fd as consume_fd
from lmcache.v1.platform.event_notifier import (
    create_event_notifier as create_event_notifier,
)
from lmcache.v1.utils.subclass_discovery import discover_subclasses

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Device info registry
# ---------------------------------------------------------------------------


_DEVICE_REGISTRY: dict[str, DeviceInfo] = {
    info.device_type: info
    for info in [
        cls()
        for cls in discover_subclasses(
            "lmcache.v1.platform",
            DeviceInfo,  # type: ignore[type-abstract]
            module_filter=lambda name: not name.startswith(("_", "base")),
            require_defined_in_module=True,
            on_import_error=lambda name, exc: None,
        )
    ]
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

    # Check DEVICE_TYPE environment variable for forced device selection.
    env_device_type = os.environ.get("DEVICE_TYPE")
    if env_device_type is not None:
        env_device_type = env_device_type.strip().lower()
        info = _DEVICE_REGISTRY.get(env_device_type)
        if info is not None and info.is_available():
            torch_module = getattr(torch, info.torch_module_name, None)
            if torch_module is not None:
                return torch_module, info.device_type
            else:
                logger.warning(
                    "DEVICE_TYPE=%r is available but torch module [%s] not found, "
                    "falling back to auto-detection.",
                    env_device_type,
                    info.torch_module_name,
                )
        else:
            logger.warning(
                "DEVICE_TYPE=%r is not available or not registered, "
                "falling back to auto-detection.",
                env_device_type,
            )

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
# Get device info
# ---------------------------------------------------------------------------
def get_device_info(device_type: str) -> DeviceInfo | None:
    """Get the DeviceInfo for the given device type.

    Args:
        device_type: The device type string (e.g. ``"cuda"``).

    Returns:
        The DeviceInfo for the given device type, or None if not found.
    """
    return _DEVICE_REGISTRY.get(device_type)


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

    torch_dev.ext = DeviceExt(_DEVICE_REGISTRY.get(torch_device_type))
else:
    logger.warning("torch_dev is None, skipping DeviceExt initialization.")

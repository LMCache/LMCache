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
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform._device_detect import (
    current_device_spec as _current_device_spec_fn,
)
from lmcache.v1.platform._device_detect import get_device_spec as get_device_spec
from lmcache.v1.platform._device_detect import (
    get_torch_device,
)
from lmcache.v1.utils.subclass_discovery import discover_subclasses
from lmcache.v1.platform.base_device_ops import DeviceOps
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.event_notifier import HAS_EVENTFD as HAS_EVENTFD
from lmcache.v1.platform.event_notifier import EventfdNotifier as EventfdNotifier
from lmcache.v1.platform.event_notifier import EventNotifier as EventNotifier
from lmcache.v1.platform.event_notifier import PipeNotifier as PipeNotifier
from lmcache.v1.platform.event_notifier import consume_fd as consume_fd
from lmcache.v1.platform.event_notifier import (
    create_event_notifier as create_event_notifier,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Resolve device ops
# ---------------------------------------------------------------------------


_DEVICE_REGISTRY: dict[str, DeviceSpec] = {
    spec.device_type: spec
    for spec in [
        cls()
        for cls in discover_subclasses(
            "lmcache.v1.platform",
            DeviceSpec,  # type: ignore[type-abstract]
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
        spec = _DEVICE_REGISTRY.get(env_device_type)
        if spec is not None and spec.is_available():
            torch_module = getattr(torch, spec.torch_module_name, None)
            if torch_module is not None:
                return torch_module, spec.device_type
            else:
                logger.warning(
                    "DEVICE_TYPE=%r is available but torch module [%s] not found, "
                    "falling back to auto-detection.",
                    env_device_type,
                    spec.torch_module_name,
                )
        else:
            logger.warning(
                "DEVICE_TYPE=%r is not available or not registered, "
                "falling back to auto-detection.",
                env_device_type,
            )

    for spec in _DEVICE_REGISTRY.values():
        # ``cpu`` is the tail fallback: even though ``CpuDeviceSpec`` now
        # lives in the registry (so ``resolve_kv_wrapper_factory`` can bind
        # its IPC wrapper), auto-detection must still prefer accelerators
        # and let the ``StubCPUDevice`` branch below handle the no-accelerator
        # case. ``DEVICE_TYPE=cpu`` remains an explicit opt-in above.
        #
        # Defence-in-depth pairs with the ``CpuDeviceSpec`` invariant that
        # ``is_available()`` stays inherited (False); do not remove either
        # side without updating the other.
        if spec.device_type == "cpu":
            continue
        if not spec.is_available():
            continue

        torch_module = getattr(torch, spec.torch_module_name, None)
        if torch_module is not None:
            return torch_module, spec.device_type
        else:
            logger.warning(
                "device [%s] is available, but torch module [%s] is not found.",
                spec.device_type,
                spec.torch_module_name,
            )

    # No accelerator found -- fall back to CPU stub
    # First Party
    from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice

    return StubCPUDevice("cpu"), "cpu"


# ---------------------------------------------------------------------------
# KV-cache IPC wrapper resolution
# ---------------------------------------------------------------------------
def resolve_kv_wrapper_factory(device_type: str) -> Any:
    """Return the KV-cache IPC wrapper factory for *device_type*.

    Reads :attr:`DeviceSpec.ipc_wrapper_cls` off the registered spec
    and returns the class's ``wrap`` classmethod (falling back to the
    class itself when no ``wrap`` is defined) so callers can invoke
    ``factory(tensor)`` uniformly.

    Args:
        device_type: The device type string (e.g. ``"cuda"``).

    Returns:
        A callable that takes a single ``torch.Tensor`` and returns a
        wrapper instance ready for the multiprocess wire.

    Raises:
        ValueError: If no spec / wrapper is registered for *device_type*.
    """
    spec = _DEVICE_REGISTRY.get(device_type)
    wrapper_cls = spec.ipc_wrapper_cls if spec is not None else None
    if wrapper_cls is None:
        raise ValueError(
            "No KV-cache wrapper factory registered for device type %r" % device_type
        )
    return getattr(wrapper_cls, "wrap", wrapper_cls)


def resolve_device_ops_cls(device_type: str) -> type[DeviceOps]:
    """Resolve the ``DeviceOps`` class for *device_type*.

    Args:
        device_type: Device type string such as ``"cuda"`` or ``"cpu"``.

    Returns:
        The resolved :class:`DeviceOps` subclass for the requested device.
        ``"cpu"`` normally resolves through :class:`CpuDeviceSpec`, while
        ``""`` uses the bare :class:`DeviceSpec` fallback. If tests or
        CLI-only fallback paths deliberately remove the CPU spec from the
        registry to exercise or simulate the minimal baseline path, ``"cpu"``
        also falls back to the bare baseline.

    Raises:
        RuntimeError: If an accelerator device has no registered
            :class:`DeviceSpec`, since silently falling back to the torch
            baseline on accelerator hardware would mask configuration errors.
    """
    spec = _DEVICE_REGISTRY.get(device_type)
    if spec is None:
        if device_type in ("", "cpu"):
            spec = DeviceSpec()
        else:
            raise RuntimeError(
                f"No DeviceSpec registered for accelerator {device_type!r}; "
                "refusing to silently fall back to the torch baseline on "
                "accelerator hardware. Ensure the platform sub-package for this "
                "device is importable and defines a DeviceSpec with ops_cls."
            )

    return spec.ops_cls


torch_dev, torch_device_type = get_torch_device()

current_device_spec: DeviceSpec = _current_device_spec_fn()




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

Device backends are discovered from two sources:

* in-tree ``platform/<backend>/`` packages inside LMCache;
* installed wheels exposing a ``DeviceSpec`` entry point in the
  ``lmcache.v1.device_specs`` group.

KV-cache IPC wrappers and ``BaseCacheContext`` subclasses hang off the
registered ``DeviceSpec`` for each backend, so adding a new
accelerator requires *zero* edits to this module whether the backend
lives in-tree or ships from an external wheel.
"""

# Future
from __future__ import annotations

__all__ = [
    "current_device_spec",
    "DeviceSpec",
    "get_device_spec",
    "get_torch_device",
    "resolve_device_ops",
    "torch_dev",
    "torch_device_type",
    "consume_fd",
    "create_event_notifier",
    "EventfdNotifier",
    "EventNotifier",
    "HAS_EVENTFD",
    "PipeNotifier",
]

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform._device_detect import (
    _build_device_registry,
)
from lmcache.v1.platform._device_detect import (
    current_device_spec as _current_device_spec_fn,
)
from lmcache.v1.platform._device_detect import get_device_spec as get_device_spec
from lmcache.v1.platform._device_detect import (
    get_torch_device,
)
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    from lmcache.v1.platform.base.device_ops import DeviceOps

# First Party
from lmcache.v1.platform.event_notifier import HAS_EVENTFD as HAS_EVENTFD
from lmcache.v1.platform.event_notifier import EventfdNotifier as EventfdNotifier
from lmcache.v1.platform.event_notifier import EventNotifier as EventNotifier
from lmcache.v1.platform.event_notifier import PipeNotifier as PipeNotifier
from lmcache.v1.platform.event_notifier import consume_fd as consume_fd
from lmcache.v1.platform.event_notifier import (
    create_event_notifier as create_event_notifier,
)

# ---------------------------------------------------------------------------
# KV-cache IPC wrapper resolution
# ---------------------------------------------------------------------------
_DEVICE_REGISTRY: dict[str, DeviceSpec] = _build_device_registry()


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


# Fallback device spec for "" / "cpu" when not in registry (cached to preserve
# singleton semantics on get_ops()).
_FALLBACK_CPU_SPEC: DeviceSpec = DeviceSpec()


def _resolve_device_spec(device_type: str) -> DeviceSpec:
    """Resolve the :class:`DeviceSpec` for *device_type*.

    ``"cpu"`` normally resolves through the registry; ``""`` uses the bare
    :class:`DeviceSpec` fallback.  If ``"cpu"`` is absent from the registry
    (e.g. tests strip it), it also falls back.

    Raises:
        RuntimeError: If an accelerator device has no registered spec.
    """
    dev_spec = _DEVICE_REGISTRY.get(device_type)
    if dev_spec is not None:
        return dev_spec
    if device_type in ("", "cpu"):
        return _FALLBACK_CPU_SPEC
    raise RuntimeError(
        f"No DeviceSpec registered for accelerator {device_type!r}; "
        "refusing to silently fall back to the torch baseline on "
        "accelerator hardware. Ensure the backend is installed and "
        "defines a DeviceSpec, either from an in-tree platform "
        "sub-package or an lmcache.v1.device_specs entry point."
    )


def resolve_device_ops(device_type: str) -> DeviceOps:
    """Resolve the :class:`DeviceOps` **instance** for *device_type*.

    Returns a cached singleton via :meth:`DeviceSpec.get_ops`.  Callers
    hitting the same *device_type* twice always get the same instance, so
    any state set on it (native handles, cached lookups) is shared across
    the process.
    """
    return _resolve_device_spec(device_type).get_ops()


torch_dev, torch_device_type = get_torch_device()

current_device_spec: DeviceSpec = _current_device_spec_fn()

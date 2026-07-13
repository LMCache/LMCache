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

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform._device_detect import (
    current_device_spec as _current_device_spec_fn,
)
from lmcache.v1.platform._device_detect import get_device_spec as get_device_spec
from lmcache.v1.platform._device_detect import (
    get_torch_device,
)
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.event_notifier import HAS_EVENTFD as HAS_EVENTFD
from lmcache.v1.platform.event_notifier import EventfdNotifier as EventfdNotifier
from lmcache.v1.platform.event_notifier import EventNotifier as EventNotifier
from lmcache.v1.platform.event_notifier import PipeNotifier as PipeNotifier
from lmcache.v1.platform.event_notifier import consume_fd as consume_fd
from lmcache.v1.platform.event_notifier import (
    create_event_notifier as create_event_notifier,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base_device_ops import DeviceOps

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Resolve device ops
# ---------------------------------------------------------------------------
def _resolve_spec(device_type: str) -> DeviceSpec:
    """Resolve the :class:`DeviceSpec` for *device_type*.

    ``"cpu"`` normally resolves through :class:`CpuDeviceSpec`, while
    ``""`` uses the bare :class:`DeviceSpec` fallback. If tests or CLI-only
    fallback paths deliberately remove the CPU spec from the registry to
    exercise or simulate the minimal baseline path, ``"cpu"`` also falls
    back to the bare baseline.

    Raises:
        RuntimeError: If an accelerator device has no registered spec,
            since silently falling back to the torch baseline on accelerator
            hardware would mask configuration errors.
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
    return spec


def resolve_device_ops_cls(device_type: str) -> type[DeviceOps]:
    """Resolve the ``DeviceOps`` class for *device_type*.

    Args:
        device_type: Device type string such as ``"cuda"`` or ``"cpu"``.

    Returns:
        The resolved :class:`DeviceOps` subclass for the requested device.
        See :func:`_resolve_spec` for resolution / error semantics.
    """
    return _resolve_spec(device_type).ops_cls


def resolve_device_ops(device_type: str) -> DeviceOps:
    """Resolve the :class:`DeviceOps` **instance** for *device_type*.

    Returns a cached singleton via :meth:`DeviceSpec.get_ops`. Callers
    hitting the same *device_type* twice always get the same instance, so
    any state set on it (native handles, cached lookups) is shared across
    the process.

    See :func:`_resolve_spec` for resolution / error semantics.
    """
    return _resolve_spec(device_type).get_ops()


torch_dev, torch_device_type = get_torch_device()

current_device_spec: DeviceSpec = _current_device_spec_fn()

# First Party
# Backward-compat: tests and internal code may reference _DEVICE_REGISTRY.
# Delegates to the cached registry in _device_detect.
from lmcache.v1.platform._device_detect import _build_device_registry  # noqa: E402

_DEVICE_REGISTRY: dict[str, DeviceSpec] = _build_device_registry()

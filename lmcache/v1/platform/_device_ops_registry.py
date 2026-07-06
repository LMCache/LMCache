# SPDX-License-Identifier: Apache-2.0
"""Device-type-keyed registry for :class:`DeviceOps` subclasses.

Mirrors the discovery used by :mod:`lmcache.v1.platform.cache_context`: the
first :func:`get_device_ops_cls` call scans the ``platform`` package two levels
deep (``platform/<backend>/device_ops.py``) and indexes every concrete
:class:`DeviceOps` subclass by its ``device_type`` ClassVar. Adding a backend
needs *zero* edits here -- just drop a new ``platform/<backend>/device_ops.py``.

Resolution is fail-fast: a requested accelerator with no registered class
raises :class:`RuntimeError` instead of silently degrading to the torch
baseline. Only ``cpu``/``""`` legitimately fall back to the base.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_device_ops import DeviceOps
from lmcache.v1.utils.subclass_discovery import discover_subclasses

logger = init_logger(__name__)

# ``device_type -> DeviceOps`` subclass, populated lazily on first use.
_CACHE: dict[str, type[DeviceOps]] = {}
_DISCOVERED: bool = False


def _discover_once() -> None:
    """Populate :data:`_CACHE` on first use."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    # First Party
    import lmcache.v1.platform as platform_pkg

    for cls in discover_subclasses(
        platform_pkg,
        DeviceOps,  # type: ignore[type-abstract]
        module_filter=lambda short_name: short_name == "device_ops",
        levels=[2, 2],
    ):
        device_type = getattr(cls, "device_type", "")
        if not device_type:
            logger.warning(
                "Skipping %s: empty device_type ClassVar; concrete DeviceOps "
                "subclasses must override it.",
                cls.__name__,
            )
            continue
        existing = _CACHE.get(device_type)
        if existing is not None and existing is not cls:
            logger.warning(
                "Multiple DeviceOps classes claim device_type=%r (%s vs %s); "
                "keeping the first.",
                device_type,
                existing.__name__,
                cls.__name__,
            )
            continue
        _CACHE.setdefault(device_type, cls)

    _DISCOVERED = True


def get_device_ops_cls(device_type: str) -> type[DeviceOps]:
    """Return the :class:`DeviceOps` subclass for *device_type*.

    ``cpu``/``""`` resolve to ``CpuDeviceOps``/``DeviceOps``; a registered
    accelerator returns its class; an accelerator with no registered class
    raises :class:`RuntimeError`.
    """
    _discover_once()
    cls = _CACHE.get(device_type)
    if cls is not None:
        return cls
    if device_type in ("", "cpu"):
        # The torch baseline is a legitimate CPU backend.
        return DeviceOps
    raise RuntimeError(
        "No DeviceOps class registered for accelerator %r. Make sure "
        "``lmcache.v1.platform.%s.device_ops`` ships a DeviceOps subclass with "
        "the matching ``device_type`` ClassVar and its compiled module is "
        "importable." % (device_type, device_type)
    )


def snapshot_device_ops() -> dict[str, type[DeviceOps]]:
    """Return a shallow copy of the registry for test fixtures."""
    _discover_once()
    return dict(_CACHE)


def restore_device_ops(state: dict[str, type[DeviceOps]]) -> None:
    """Replace the registry with *state* and mark discovery complete."""
    global _DISCOVERED
    _CACHE.clear()
    _CACHE.update(state)
    _DISCOVERED = True

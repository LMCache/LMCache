# SPDX-License-Identifier: Apache-2.0
"""Universal platform registry.

Scans ``platform/base/`` to discover every class defined there that
subclasses :class:`lmcache.v1.platform.base._base.PlatformBase`, then
scans device sub-packages for concrete subclasses of each.  The
registry is indexed by ``(base_class, device_type)`` pairs.

``device_type`` is purely a *subclass* attribute naming the concrete
device a given implementation serves (``"cuda"``/``"cpu"``).

Adding a new base class: drop a ``.py`` file in ``platform/base/`` that
defines a :class:`PlatformBase` subclass — done.  Adding a new device
implementation: drop a subclass file in ``platform/<device>/`` whose
class sets a ``device_type`` ClassVar — done.  No other code changes
needed in either case.

Thin convenience wrappers (:func:`get_kv_wrapper_factory`,
:func:`register_availability`, :func:`is_available`) sit on top of the
core :func:`get_impl` / :func:`get_all_impls` lookups.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Dict
import importlib
import inspect
import pkgutil
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Public sentinel used by callers who want the always-available
# fall-back regardless of the running ``torch_device_type``.
DEFAULT_BACKEND: str = "cpu"

# Core data structure: {base_class: {device_type: concrete_class}}
_REGISTRY: Dict[type, Dict[str, type]] = {}

# Per-backend availability predicate (e.g. CUDA's ``is_available``).
# Missing entry == always available.
_AVAILABILITY: Dict[str, Callable[[], bool]] = {}

_DISCOVERED: bool = False
_DISCOVERY_LOCK = threading.Lock()


def _collect_base_classes() -> list[type]:
    """Scan ``platform/base/`` and collect marker-declared base classes.

    A class qualifies as a base class iff both of these are true:

    * it is defined in a module directly under ``platform/base/`` (so
      imported names are excluded), and
    * it subclasses :class:`PlatformBase`.

    Returns:
        List of base classes discovered in ``platform/base/``.
    """
    # First Party
    from lmcache.v1.platform.base._base import PlatformBase
    import lmcache.v1.platform.base as base_pkg

    base_classes: list[type] = []
    pkg_path = getattr(base_pkg, "__path__", None)
    if pkg_path is None:
        return base_classes

    for _, module_name, is_pkg in pkgutil.iter_modules(pkg_path):
        if is_pkg:
            continue
        full_name = f"{base_pkg.__name__}.{module_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            logger.warning("Failed to import base module %s", full_name, exc_info=True)
            continue

        for _, cls in inspect.getmembers(mod, inspect.isclass):
            # Only classes actually defined in this module (not imports).
            if cls.__module__ != mod.__name__:
                continue
            if cls is PlatformBase or not issubclass(cls, PlatformBase):
                continue
            base_classes.append(cls)
            _REGISTRY.setdefault(cls, {})

    return base_classes


def _discover_all_once() -> None:
    """Populate the registry on first use (thread-safe, runs at most once).

    Walks ``platform/base/`` to collect every base class, then walks
    ``platform/`` two levels deep to find concrete subclasses for each
    base class and registers them keyed by ``device_type``.

    Subclasses with an empty ``device_type`` are skipped with a warning.
    When two subclasses claim the same ``(base_class, device_type)`` pair
    the first one wins and a warning is emitted.  The special
    ``_is_default_wrapper`` ClassVar (used by
    :class:`~lmcache.v1.platform.base.ipc_wrapper.DeviceIPCWrapper`
    subclasses) allows multiple implementations per device_type while
    only auto-registering the default one.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    with _DISCOVERY_LOCK:
        if _DISCOVERED:
            return

        # First Party
        from lmcache.v1.utils.subclass_discovery import discover_subclasses
        import lmcache.v1.platform as platform_pkg

        base_classes = _collect_base_classes()

        for base_cls in base_classes:
            sub_cls: type
            for sub_cls in discover_subclasses(
                platform_pkg,
                base_cls,
                levels=[2, 2],
                include_abstract=False,
            ):
                # Respect _is_default_wrapper: if a class has this attribute
                # and it is False, skip it (it is a non-default sibling that
                # deliberately opts out of auto-registration).
                is_default = getattr(sub_cls, "_is_default_wrapper", None)
                if is_default is not None and not is_default:
                    continue

                device_type: str = getattr(sub_cls, "device_type", "")
                if not device_type:
                    logger.warning(
                        "Skipping %s: empty device_type ClassVar; subclasses "
                        "of %s must override device_type.",
                        sub_cls.__name__,
                        base_cls.__name__,
                    )
                    continue

                existing = _REGISTRY[base_cls].get(device_type)
                if existing is not None and existing is not sub_cls:
                    logger.warning(
                        "Multiple %s subclasses claim device_type=%r "
                        "(%s vs %s); keeping the first.",
                        base_cls.__name__,
                        device_type,
                        existing.__name__,
                        sub_cls.__name__,
                    )
                    continue

                _REGISTRY[base_cls][device_type] = sub_cls

        _DISCOVERED = True


def get_impl(base_class: type, device_type: str) -> type:
    """Get the concrete implementation of *base_class* for *device_type*.

    This is the primary lookup API.  Example::

        from lmcache.v1.platform._registry import get_impl
        from lmcache.v1.platform.base.cache_context import BaseCacheContext

        cls = get_impl(BaseCacheContext, "cuda")

    Args:
        base_class: A base class defined in ``platform/base/``.
        device_type: The ``torch.device.type`` string (e.g. ``"cuda"``).

    Returns:
        The concrete subclass registered for ``(base_class, device_type)``.

    Raises:
        ValueError: If *base_class* is not in the registry or no
            implementation is registered for *device_type*.
    """
    _discover_all_once()
    table = _REGISTRY.get(base_class)
    if table is None:
        raise ValueError(
            "Base class %r is not registered.  Make sure it subclasses "
            "PlatformBase and is defined in a module directly under "
            "platform/base/." % base_class
        )
    cls = table.get(device_type)
    if cls is None:
        raise ValueError(
            "No %s implementation registered for device_type=%r"
            % (base_class.__name__, device_type)
        )
    return cls


def get_all_impls(base_class: type) -> Dict[str, type]:
    """Return all registered implementations for *base_class*.

    Args:
        base_class: A base class defined in ``platform/base/``.

    Returns:
        A shallow copy of the ``{device_type: concrete_class}`` mapping.
    """
    _discover_all_once()
    return dict(_REGISTRY.get(base_class, {}))


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def register_availability(device_type: str, predicate: Callable[[], bool]) -> None:
    """Register an availability predicate for a device type.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).
        predicate: A zero-argument callable returning ``True`` when the
            device is available.
    """
    _AVAILABILITY[device_type] = predicate


def is_available(device_type: str) -> bool:
    """Check whether a device type is available.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).

    Returns:
        ``True`` if the device is available or no predicate is registered,
        ``False`` otherwise.
    """
    pred = _AVAILABILITY.get(device_type)
    if pred is None:
        return True
    try:
        return bool(pred())
    except Exception:
        return False


def get_kv_wrapper_factory(device_type: str) -> Callable[..., Any]:
    """Pick the KV-cache wrapper factory for ``device_type``.

    Triggers lazy auto-discovery on first call.  A missing entry means no
    :class:`~lmcache.v1.platform.base.ipc_wrapper.DeviceIPCWrapper`
    subclass declared ``device_type`` for the requested backend.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).

    Returns:
        The registered KV-cache wrapper factory for the device type.

    Raises:
        ValueError: If no factory is registered for the device type.
    """
    # Imported lazily: DeviceIPCWrapper's module imports torch at the top
    # level, so a module-level import here would require torch even in
    # environments that only use the registry for pin-memory or cache-context
    # lookups.
    # First Party
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

    cls = get_impl(DeviceIPCWrapper, device_type)
    return getattr(cls, "wrap", cls)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def snapshot() -> Dict[str, Any]:
    """Return a deep-copy of the registry tables.

    Test suites use this to install backend overrides without leaking
    state across tests; pair with :func:`restore` in a ``finally`` /
    fixture teardown clause.

    Returns:
        A dict with keys ``"registry"``, ``"availability"`` and
        ``"discovered"``.
    """
    return {
        "registry": {k: dict(v) for k, v in _REGISTRY.items()},
        "availability": dict(_AVAILABILITY),
        "discovered": _DISCOVERED,
    }


def restore(state: Dict[str, Any]) -> None:
    """Restore registry tables to a previously :func:`snapshot`-ed state.

    Args:
        state: A snapshot dict as returned by :func:`snapshot`.
    """
    global _DISCOVERED
    _REGISTRY.clear()
    for k, v in state.get("registry", {}).items():
        _REGISTRY[k] = dict(v)
    _AVAILABILITY.clear()
    _AVAILABILITY.update(state.get("availability", {}))
    _DISCOVERED = bool(state.get("discovered", False))


def reset_for_tests() -> None:
    """Wipe registry tables and force re-discovery on next access.

    Intended **only** for test fixtures: clears every registered entry
    and flips :data:`_DISCOVERED` back to ``False`` so the next
    :func:`get_impl` call re-runs the scan and re-populates the table
    from the live ``platform`` sub-packages.
    """
    global _DISCOVERED
    _REGISTRY.clear()
    _AVAILABILITY.clear()
    _DISCOVERED = False

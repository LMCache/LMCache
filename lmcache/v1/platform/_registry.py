# SPDX-License-Identifier: Apache-2.0
"""Platform backend registry.

Two independent registries live here:

**Universal 3-D registry** (``get_impl`` / ``resolve_impl``)
    Each accelerator sub-package ships concrete subclasses of the ABC
    base classes defined under ``lmcache/v1/platform/base/``.
    :func:`_discover_all_once` auto-discovers those base classes and
    their implementations and indexes them by
    ``(base_class, device_type, impl_key)``.

    * :func:`get_impl` performs a strict lookup and raises ``ValueError``
      when nothing is registered for the requested triple.
    * :func:`resolve_impl` is the policy-aware lookup: it re-raises for
      abstract base classes (required capabilities) and falls back to the
      base class itself only when the base class is concrete.

**IPC wrapper registry** (``get_kv_wrapper_factory``)
    Legacy path retained for backward compatibility.  Concrete
    :class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
    subclasses are discovered via :func:`_discover_wrappers_once` and
    indexed by ``device_type``.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Dict
import abc
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Public sentinel used by callers who want the always-available
# fall-back regardless of the running ``torch_device_type``.
DEFAULT_BACKEND: str = "cpu"


# ---------------------------------------------------------------------------
# Universal 3-D registry
# ---------------------------------------------------------------------------

# _REGISTRY[base_class][device_type][impl_key] = concrete_class
# Populated lazily by _discover_all_once().
_REGISTRY: Dict[type, Dict[str, Dict[str, type]]] = {}

_ALL_DISCOVERED: bool = False
_ALL_DISCOVERY_LOCK = threading.Lock()


def _discover_base_classes() -> list[type[abc.ABC]]:
    """Return all registry base classes from ``lmcache.v1.platform.base``.

    A class qualifies when it is **directly defined** in a non-private
    module directly under the ``base`` sub-package **and** subclasses
    :class:`abc.ABC`.

    Returns:
        List of discovered base classes (all subclass abc.ABC).
    """
    # Standard
    import importlib
    import inspect
    import pkgutil

    base_pkg = importlib.import_module("lmcache.v1.platform.base")
    base_classes: list[type[abc.ABC]] = []

    for _, short_name, is_pkg in pkgutil.iter_modules(base_pkg.__path__):  # type: ignore[attr-defined]
        if is_pkg or short_name.startswith("_"):
            continue
        full_name = "lmcache.v1.platform.base.%s" % short_name
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:
            logger.warning(
                "Failed to import %s during base-class discovery: %s", full_name, exc
            )
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module.__name__
                and issubclass(obj, abc.ABC)
                and obj is not abc.ABC
            ):
                base_classes.append(obj)

    return base_classes


def _discover_all_once() -> None:
    """Populate :data:`_REGISTRY` on first use.

    Walks ``lmcache.v1.platform.base`` for ABC base classes, then scans
    each device sub-package (depth 2) for concrete implementations.
    Each implementation must set ``device_type`` (a non-empty string
    ClassVar) to be indexed; ``impl_key`` is optional and defaults to
    ``"default"``.

    Multiple implementations claiming the same
    ``(base_class, device_type, impl_key)`` triple trigger a warning;
    the first one wins.
    """
    global _ALL_DISCOVERED
    if _ALL_DISCOVERED:
        return

    with _ALL_DISCOVERY_LOCK:
        if _ALL_DISCOVERED:
            return

        # First Party
        from lmcache.v1.utils.subclass_discovery import discover_subclasses
        import lmcache.v1.platform as platform_pkg

        base_classes = _discover_base_classes()
        if not base_classes:
            logger.warning(
                "No registry base classes found under lmcache.v1.platform.base; "
                "universal registry will be empty."
            )

        for base_cls in base_classes:
            _REGISTRY.setdefault(base_cls, {})
            for impl_cls in discover_subclasses(
                platform_pkg,
                base_cls,  # type: ignore[type-abstract]
                module_filter=lambda name: not name.startswith(("_", "base")),
                require_defined_in_module=True,
                on_import_error=lambda name, exc: logger.warning(
                    "Failed to import %s during registry discovery: %s", name, exc
                ),
                levels=[2, 2],
            ):
                _register_impl(base_cls, impl_cls)

        _ALL_DISCOVERED = True


def _register_impl(base_cls: type, impl_cls: type) -> None:
    """Register *impl_cls* in the 3-D table under *base_cls*.

    The implementation must carry a non-empty ``device_type`` ClassVar.
    ``impl_key`` defaults to ``"default"`` when absent.  Duplicate
    ``(device_type, impl_key)`` entries trigger a warning; the first
    registration wins.

    Args:
        base_cls: The registry base class this implementation belongs to.
        impl_cls: The concrete implementation class to register.
    """
    device_type: str = getattr(impl_cls, "device_type", "")
    if not device_type:
        logger.warning(
            "Skipping %s: empty device_type ClassVar; concrete subclasses of "
            "%s must set device_type.",
            impl_cls.__name__,
            base_cls.__name__,
        )
        return

    impl_key: str = getattr(impl_cls, "impl_key", "default") or "default"

    device_table = _REGISTRY.setdefault(base_cls, {})
    key_table = device_table.setdefault(device_type, {})

    existing = key_table.get(impl_key)
    if existing is not None and existing is not impl_cls:
        logger.warning(
            "Multiple %s subclasses registered for (device_type=%r, impl_key=%r): "
            "%s vs %s; keeping the first.",
            base_cls.__name__,
            device_type,
            impl_key,
            existing.__name__,
            impl_cls.__name__,
        )
        return

    key_table[impl_key] = impl_cls


def get_impl(base_class: type, device_type: str, impl_key: str = "default") -> type:
    """Return the registered implementation for the given triple.

    Triggers lazy auto-discovery on the first call.  Raises
    ``ValueError`` when nothing is registered for the requested
    ``(base_class, device_type, impl_key)``; it never falls back.

    Args:
        base_class: The registry base class (e.g. ``PinMemoryBackend``).
        device_type: The device type string (e.g. ``"cuda"``).
        impl_key: The implementation key (default: ``"default"``).

    Returns:
        The concrete implementation class.

    Raises:
        ValueError: If no implementation is registered for the triple.
    """
    _discover_all_once()

    device_table = _REGISTRY.get(base_class)
    if device_table is None:
        raise ValueError(
            "Base class %r is not registered in the universal registry. "
            "Ensure it is defined in a module under lmcache/v1/platform/base/ "
            "and subclasses abc.ABC." % base_class
        )

    key_table = device_table.get(device_type)
    if key_table is None:
        raise ValueError(
            "No %s implementation registered for device_type=%r."
            % (base_class.__name__, device_type)
        )

    impl = key_table.get(impl_key)
    if impl is None:
        raise ValueError(
            "No %s implementation registered for (device_type=%r, impl_key=%r)."
            % (base_class.__name__, device_type, impl_key)
        )

    return impl


def resolve_impl(base_class: type, device_type: str, impl_key: str = "default") -> type:
    """Resolve an implementation class for callers using fallback policy.

    This API first performs strict lookup via :func:`get_impl`.
    If strict lookup fails:

    * Abstract base classes (``inspect.isabstract(base_class)``) re-raise
      the original ``ValueError`` to preserve fail-fast semantics for
      required capabilities.
    * Concrete base classes fall back to ``base_class`` itself, which
      lets optional capabilities provide a built-in default implementation.

    Args:
        base_class: The registry base class.
        device_type: The device type string.
        impl_key: The implementation key (default: ``"default"``).

    Returns:
        The concrete implementation class, or ``base_class`` when fallback
        is allowed by the abstractness rule.

    Raises:
        ValueError: If no implementation is registered and ``base_class``
            is abstract.
    """
    try:
        return get_impl(base_class, device_type, impl_key)
    except ValueError:
        # Standard
        import inspect

        if inspect.isabstract(base_class):
            raise
        return base_class


def reset_registry_for_tests() -> None:
    """Wipe the universal 3-D registry and force re-discovery.

    Intended **only** for test fixtures.  Clears every registered
    implementation and flips :data:`_ALL_DISCOVERED` back to ``False``
    so the next :func:`get_impl` / :func:`resolve_impl` call re-runs
    :func:`_discover_all_once`.
    """
    global _ALL_DISCOVERED
    _REGISTRY.clear()
    _ALL_DISCOVERED = False


def snapshot_registry() -> Dict[str, Any]:
    """Return a copy of the universal registry state for test isolation.

    Returns:
        A dict with keys ``"registry"`` and ``"discovered"``.
    """
    return {
        "registry": {
            base: {dt: dict(key_table) for dt, key_table in device_table.items()}
            for base, device_table in _REGISTRY.items()
        },
        "discovered": _ALL_DISCOVERED,
    }


def restore_registry(state: Dict[str, Any]) -> None:
    """Restore the universal registry to a previously snapshotted state.

    Args:
        state: A snapshot dict as returned by :func:`snapshot_registry`.
    """
    global _ALL_DISCOVERED
    _REGISTRY.clear()
    for base, device_table in state.get("registry", {}).items():
        _REGISTRY[base] = {dt: dict(kt) for dt, kt in device_table.items()}
    _ALL_DISCOVERED = bool(state.get("discovered", False))


# ---------------------------------------------------------------------------
# IPC wrapper registry (legacy -- retained for backward compatibility)
# ---------------------------------------------------------------------------

# KV-cache IPC wrapper factory per device type.  Populated lazily on
# first :func:`get_kv_wrapper_factory` call by scanning the
# ``platform`` package for
# :class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
# subclasses.  Tests substitute entries via
# :func:`snapshot` / :func:`restore`.
_KV_WRAPPER_FACTORIES: Dict[str, Callable[..., Any]] = {}

# Guard so discovery only runs once (lazy init).  The lock plus the
# double-checked flag below keep the first concurrent caller from
# racing a second one through the scan and emitting duplicate
# "multiple wrappers claim device_type=..." warnings.
_WRAPPERS_DISCOVERED: bool = False
_DISCOVERY_LOCK = threading.Lock()


def _discover_wrappers_once() -> None:
    """Populate :data:`_KV_WRAPPER_FACTORIES` on first use.

    Walks ``lmcache.v1.platform`` two levels deep for
    :class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
    subclasses.  Each subclass is indexed by its *device_type*
    ClassVar, and its *wrap* factory is stored as the KV-wrapper
    factory — but only when ``_is_default_wrapper`` is ``True``
    (so e.g. :class:`~lmcache.v1.platform.cuda.ipc_wrapper.RawCudaIPCWrapper`
    is skipped in favour of
    :class:`~lmcache.v1.platform.cuda.ipc_wrapper.CudaIPCWrapper`).

    Subclasses with an empty *device_type* or ``_is_default_wrapper ==
    False`` are skipped.  Multiple subclasses claiming the same
    *device_type* trigger a warning; the first one wins.
    """
    global _WRAPPERS_DISCOVERED
    # Fast path: avoid the lock once discovery is done (the common case).
    if _WRAPPERS_DISCOVERED:
        return

    with _DISCOVERY_LOCK:
        # Re-check under the lock: another thread may have run the
        # scan while we were waiting.
        if _WRAPPERS_DISCOVERED:
            return

        # First Party
        from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper
        from lmcache.v1.utils.subclass_discovery import discover_subclasses
        import lmcache.v1.platform as platform_pkg

        for cls in discover_subclasses(
            platform_pkg,
            DeviceIPCWrapper,  # type: ignore[type-abstract]
            levels=[2, 2],
        ):
            _register_discovered_wrapper(cls)

        _WRAPPERS_DISCOVERED = True


def _register_discovered_wrapper(cls: type) -> None:
    """Index *cls* in :data:`_KV_WRAPPER_FACTORIES` by its device_type.

    Only registers when ``_is_default_wrapper`` is ``True`` so sibling
    subclasses (e.g. ``RawCudaIPCWrapper`` vs ``CudaIPCWrapper``) can
    share a ``device_type`` without colliding.
    """
    if not getattr(cls, "_is_default_wrapper", False):
        return

    device_type: str = getattr(cls, "device_type", "")
    if not device_type:
        logger.warning(
            "Skipping %s: empty device_type ClassVar; concrete "
            "DeviceIPCWrapper subclasses must override it.",
            cls.__name__,
        )
        return

    factory = getattr(cls, "wrap", cls)
    existing = _KV_WRAPPER_FACTORIES.get(device_type)
    if existing is not None and existing is not factory:
        logger.warning(
            "Multiple KV-wrapper classes claim device_type=%r "
            "(%s vs %s); keeping the first.",
            device_type,
            getattr(existing, "__name__", str(existing)),
            cls.__name__,
        )
        return

    _KV_WRAPPER_FACTORIES[device_type] = factory


def register_kv_wrapper(device_type: str, factory: Callable[..., Any]) -> None:
    """Register a KV-cache IPC wrapper factory for ``device_type``.

    This is the manual registration path kept for backward
    compatibility.  New backends should instead set ``device_type``
    and ``wrap`` on their :class:`DeviceIPCWrapper` subclass and let
    :func:`_discover_wrappers_once` handle registration.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).
        factory: A callable that takes a single ``torch.Tensor`` and
            returns a wrapper instance ready for the multiprocess wire.
    """
    _KV_WRAPPER_FACTORIES[device_type] = factory


def get_kv_wrapper_factory(device_type: str) -> Callable[..., Any]:
    """Pick the KV-cache wrapper factory for ``device_type``.

    Triggers lazy auto-discovery on first call (see
    :func:`_discover_wrappers_once`).  A missing entry means no
    :class:`~lmcache.v1.platform.base_ipc_wrapper.DeviceIPCWrapper`
    subclass declared *device_type* for the requested backend.

    Args:
        device_type: The device type string (e.g., ``"cuda"``).

    Returns:
        The registered KV-cache wrapper factory for the device type.

    Raises:
        ValueError: If no factory is registered for the device type.
    """
    _discover_wrappers_once()
    factory = _KV_WRAPPER_FACTORIES.get(device_type)
    if factory is None:
        raise ValueError(
            "No KV-cache wrapper factory registered for device type %r" % device_type
        )
    return factory


def snapshot() -> Dict[str, Any]:
    """Return a deep-copy of the registry tables.

    Test suites use this to install backend overrides without leaking
    state across tests; pair with :func:`restore` in a ``finally`` /
    fixture teardown clause.

    The lazy-discovery flag is captured alongside the tables: if a test
    snapshots *before* discovery runs and restores *after*, the next
    caller still re-runs discovery and picks up the auto-registered
    backends, instead of seeing a stale "already discovered, table is
    empty" view.

    Returns:
        A dict with keys ``"kv_wrapper"``, and ``"discovered"``.
    """
    return {
        "kv_wrapper": dict(_KV_WRAPPER_FACTORIES),
        "discovered": _WRAPPERS_DISCOVERED,
    }


def restore(state: Dict[str, Any]) -> None:
    """Restore registry tables to a previously :func:`snapshot`-ed state.

    Args:
        state: A snapshot dict as returned by :func:`snapshot`.
    """
    global _WRAPPERS_DISCOVERED
    _KV_WRAPPER_FACTORIES.clear()
    _KV_WRAPPER_FACTORIES.update(state.get("kv_wrapper", {}))
    _WRAPPERS_DISCOVERED = bool(state.get("discovered", False))


def reset_for_tests() -> None:
    """Wipe registry tables and force re-discovery on next access.

    Intended **only** for test fixtures: clears every registered KV
    wrapper and flips
    :data:`_WRAPPERS_DISCOVERED` back to ``False`` so the next
    :func:`get_kv_wrapper_factory` call re-runs the
    :func:`_discover_wrappers_once` scan and re-populates the table
    from the live ``platform`` sub-packages.

    This is the recommended replacement for callers that previously
    hand-mutated module-private globals; pair with an ``autouse``
    pytest fixture to guarantee every test starts and ends with a
    clean slate (see ``tests/v1/multiprocess/conftest.py``).
    """
    global _WRAPPERS_DISCOVERED
    _KV_WRAPPER_FACTORIES.clear()
    _WRAPPERS_DISCOVERED = False

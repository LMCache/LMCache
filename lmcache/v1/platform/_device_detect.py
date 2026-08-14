# SPDX-License-Identifier: Apache-2.0
"""Device detection helpers, decoupled from ``lmcache.v1.platform``.

Kept in its own module (rather than in ``platform/__init__.py``) so that
peers such as :mod:`lmcache.v1.platform.torch_ops` can import the
detection primitives at the top of the file without introducing an
import cycle -- ``platform/__init__.py`` itself pulls in
``base_device_ops``, which in turn pulls in ``torch_ops``, so any name
that ``torch_ops`` needs from the platform package must live *outside*
that init chain.

The registry of :class:`DeviceSpec` subclasses and the detected torch
device module are both built lazily on first access and then cached
process-wide.  Registry entries may come from LMCache's in-tree
``platform/<backend>/`` packages or from installed wheels publishing a
``DeviceSpec`` entry point in the ``lmcache.v1.device_specs`` group.
"""

# Standard
from functools import lru_cache
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any
import inspect
import os

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

logger = init_logger(__name__)

_DEVICE_SPEC_ENTRYPOINT_GROUP = "lmcache.v1.device_specs"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_device_registry() -> "dict[str, DeviceSpec]":
    """Discover and instantiate every :class:`DeviceSpec` subclass.

    Discovery is deferred until first use so importing this module
    stays cheap and side-effect free.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec
    from lmcache.v1.utils.subclass_discovery import discover_subclasses

    registry: dict[str, DeviceSpec] = {}

    for cls in discover_subclasses(
        "lmcache.v1.platform",
        DeviceSpec,  # type: ignore[type-abstract]
        module_filter=lambda name: not name.startswith(("_", "base")),
        require_defined_in_module=True,
        on_import_error=lambda name, exc: None,
    ):
        _register_device_spec(
            registry,
            cls(),
            source=f"{cls.__module__}.{cls.__qualname__}",
        )

    for entry_point, spec in _iter_entry_point_device_specs(DeviceSpec):
        _register_device_spec(
            registry,
            spec,
            source=(
                f"entry point {entry_point.group}:{entry_point.name}"
                f" ({entry_point.value})"
            ),
        )

    return registry


def _iter_entry_point_device_specs(
    base_class: "type[DeviceSpec]",
) -> "list[tuple[Any, DeviceSpec]]":
    """Load ``DeviceSpec`` objects exposed through installed entry points."""
    discovered: list[tuple[Any, DeviceSpec]] = []

    for entry_point in _select_entry_points(_DEVICE_SPEC_ENTRYPOINT_GROUP):
        try:
            loaded = entry_point.load()
        except Exception as exc:
            logger.warning(
                "Failed to load device entry point %s:%s (%s): %s",
                entry_point.group,
                entry_point.name,
                entry_point.value,
                exc,
            )
            continue

        spec = _coerce_entry_point_device_spec(entry_point, loaded, base_class)
        if spec is not None:
            discovered.append((entry_point, spec))

    return discovered


def _select_entry_points(group: str) -> list[Any]:
    """Return entry points from *group* across Python 3.10-3.13 APIs."""
    entry_points = importlib_metadata.entry_points()
    select = getattr(entry_points, "select", None)
    if callable(select):
        return list(select(group=group))

    legacy_get = getattr(entry_points, "get", None)
    if callable(legacy_get):
        return list(legacy_get(group, ()))

    return []


def _coerce_entry_point_device_spec(
    entry_point: Any,
    loaded: Any,
    base_class: "type[DeviceSpec]",
) -> "DeviceSpec | None":
    """Normalize an entry point target to a concrete ``DeviceSpec`` instance."""
    if inspect.isclass(loaded):
        try:
            is_subclass = issubclass(loaded, base_class)
        except TypeError:
            is_subclass = False
        if not is_subclass:
            logger.warning(
                "Skipping device entry point %s:%s (%s): target is not a "
                "DeviceSpec subclass.",
                entry_point.group,
                entry_point.name,
                entry_point.value,
            )
            return None
        if inspect.isabstract(loaded):
            logger.warning(
                "Skipping device entry point %s:%s (%s): DeviceSpec subclass "
                "is abstract.",
                entry_point.group,
                entry_point.name,
                entry_point.value,
            )
            return None
        return loaded()

    if isinstance(loaded, base_class):
        return loaded

    logger.warning(
        "Skipping device entry point %s:%s (%s): target must resolve to a "
        "DeviceSpec subclass or instance.",
        entry_point.group,
        entry_point.name,
        entry_point.value,
    )
    return None


def _register_device_spec(
    registry: "dict[str, DeviceSpec]",
    spec: "DeviceSpec",
    *,
    source: str,
) -> None:
    """Insert *spec* into *registry* unless a device_type collision exists."""
    device_type = spec.device_type
    if not device_type:
        logger.warning("Skipping DeviceSpec from %s: device_type is empty.", source)
        return

    existing = registry.get(device_type)
    if existing is not None:
        logger.warning(
            "Skipping DeviceSpec from %s: device_type %r is already provided by %s.",
            source,
            device_type,
            f"{type(existing).__module__}.{type(existing).__qualname__}",
        )
        return

    registry[device_type] = spec


def _get_platform_device_registry() -> "dict[str, DeviceSpec]":
    """Return the shared device registry owned by ``lmcache.v1.platform``."""
    # First Party
    import lmcache.v1.platform as platform_pkg

    return platform_pkg._DEVICE_REGISTRY


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

    registry = _get_platform_device_registry()

    # Check DEVICE_TYPE environment variable for forced device selection.
    env_device_type = os.environ.get("DEVICE_TYPE")
    if env_device_type is not None:
        env_device_type = env_device_type.strip().lower()
        spec = registry.get(env_device_type)
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

    for spec in registry.values():
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
# Public functions
# ---------------------------------------------------------------------------


def get_device_spec(device_type: str) -> "DeviceSpec | None":
    """Return the :class:`DeviceSpec` registered for *device_type*, if any."""
    return _get_platform_device_registry().get(device_type)


@lru_cache(maxsize=1)
def get_torch_device() -> tuple[Any, str]:
    """Return the cached ``(torch_dev, torch_device_type)`` pair.

    Lazy + memoized so that peers like :mod:`torch_ops` can safely
    import this helper at module top level: no work is performed until
    the tuple is actually needed.
    """
    torch_dev, torch_device_type = _detect_device()
    logger.info("torch_dev=%s, torch_device_type=%s", torch_dev, torch_device_type)
    return torch_dev, torch_device_type


@lru_cache(maxsize=1)
def current_device_spec() -> "DeviceSpec":
    """Return the :class:`DeviceSpec` for the detected device.

    Falls back to a bare ``DeviceSpec()`` (no-op / all False semantics)
    when no accelerator sub-package matches.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

    _, device_type = get_torch_device()
    spec = get_device_spec(device_type)
    if spec is None:
        if device_type != "cpu":
            logger.warning(
                "No DeviceSpec registered for %r; using fallback"
                " with no-op capabilities.",
                device_type,
            )
        return DeviceSpec()
    return spec

# SPDX-License-Identifier: Apache-2.0
"""Device detection helpers, decoupled from ``lmcache.v1.platform``.

Kept in its own module (rather than in ``platform/__init__.py``) so that
peers such as :mod:`lmcache.v1.platform.torch_ops` can import the
detection primitives at the top of the file without introducing an
import cycle -- ``platform/__init__.py`` itself pulls in
``base_device_ops``, which in turn pulls in ``torch_ops``, so any name
that ``torch_ops`` needs from the platform package must live *outside*
that init chain.

The registry combines the :class:`DeviceSpec` subclasses shipped with
LMCache and external subclasses published through the
``lmcache.device_plugins`` Python entry-point group. The registry and the
detected torch device module are both built lazily on first access and then
cached process-wide.
"""

# Standard
from functools import lru_cache
from typing import TYPE_CHECKING, Any
import importlib.metadata
import os

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

logger = init_logger(__name__)

DEVICE_PLUGIN_ENTRY_POINT_GROUP = "lmcache.device_plugins"
DEVICE_BACKEND_ENV_VAR = "LMCACHE_DEVICE_BACKEND"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_spec_name(name: str, *, field_name: str) -> None:
    """Validate a DeviceSpec identifier used in env vars and registries."""
    if not isinstance(name, str) or not name or name != name.lower():
        raise ValueError(f"{field_name} must be a non-empty lowercase string")


def _load_external_device_specs() -> "list[DeviceSpec]":
    """Load external device specifications from Python entry points.

    Each entry point must use the backend's unique ``backend_name`` as its
    name and resolve to a concrete :class:`DeviceSpec` subclass with a
    no-argument constructor. Invalid or broken plugins are skipped so that an
    unrelated installed package cannot prevent LMCache from starting.

    Returns:
        Device specifications loaded successfully from installed plugins.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

    specs: list[DeviceSpec] = []
    entry_points = importlib.metadata.entry_points(
        group=DEVICE_PLUGIN_ENTRY_POINT_GROUP
    )
    for entry_point in sorted(entry_points, key=lambda item: (item.name, item.value)):
        try:
            spec_cls = entry_point.load()
            if (
                not isinstance(spec_cls, type)
                or spec_cls is DeviceSpec
                or not issubclass(spec_cls, DeviceSpec)
            ):
                raise TypeError("entry point must resolve to a DeviceSpec subclass")

            spec = spec_cls()
            if not isinstance(spec, DeviceSpec):
                raise TypeError("entry point must construct a DeviceSpec instance")

            _validate_spec_name(spec.device_type, field_name="device_type")
            _validate_spec_name(spec.backend_name, field_name="backend_name")
            if spec.backend_name != entry_point.name:
                raise ValueError(
                    "entry-point name must match DeviceSpec.backend_name "
                    f"({entry_point.name!r} != {spec.backend_name!r})"
                )
        except Exception as exc:
            logger.warning(
                "Failed to load LMCache device plugin %r (%s): %s",
                entry_point.name,
                entry_point.value,
                exc,
            )
            continue
        specs.append(spec)
    return specs


@lru_cache(maxsize=1)
def _build_backend_registry() -> "dict[str, DeviceSpec]":
    """Discover and index registered specs by unique backend name.

    Discovery is deferred until first use so importing this module stays cheap
    and side-effect free. ``backend_name`` is LMCache's explicit backend
    selector, so collisions are resolved deterministically and logged.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec
    from lmcache.v1.utils.subclass_discovery import discover_subclasses

    registry: dict[str, DeviceSpec] = {}
    specs = [
        *(
            cls()
            for cls in discover_subclasses(
                "lmcache.v1.platform",
                DeviceSpec,  # type: ignore[type-abstract]
                module_filter=lambda name: not name.startswith(("_", "base")),
                require_defined_in_module=True,
                on_import_error=lambda name, exc: None,
            )
        ),
        *_load_external_device_specs(),
    ]
    for spec in specs:
        _validate_spec_name(spec.device_type, field_name="device_type")
        _validate_spec_name(spec.backend_name, field_name="backend_name")
        if spec.backend_name in registry:
            logger.warning(
                "Ignoring DeviceSpec %s because backend_name %r is already "
                "registered by %s",
                type(spec).__qualname__,
                spec.backend_name,
                type(registry[spec.backend_name]).__qualname__,
            )
            continue
        registry[spec.backend_name] = spec
    return registry


@lru_cache(maxsize=1)
def _build_device_registry() -> "dict[str, tuple[DeviceSpec, ...]]":
    """Group registered specs by torch-facing ``device_type``."""
    registry: dict[str, list[DeviceSpec]] = {}
    for spec in _build_backend_registry().values():
        registry.setdefault(spec.device_type, []).append(spec)
    return {device_type: tuple(specs) for device_type, specs in registry.items()}


def _get_env_choice(env_var_name: str) -> "str | None":
    """Return a normalized explicit env choice, or ``None`` when unset."""
    value = os.environ.get(env_var_name)
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _resolve_explicit_backend(
    torch: Any,
    backend_name: str,
) -> "tuple[Any, DeviceSpec]":
    """Resolve an explicitly requested backend by unique backend name."""
    spec = _build_backend_registry().get(backend_name)
    if spec is None:
        available_backends = ", ".join(sorted(_build_backend_registry()))
        raise RuntimeError(
            f"{DEVICE_BACKEND_ENV_VAR}={backend_name!r} is not registered. "
            f"Available backends: {available_backends or '<none>'}."
        )
    if not spec.is_available():
        raise RuntimeError(
            f"{DEVICE_BACKEND_ENV_VAR}={backend_name!r} is registered but not "
            "available on this host."
        )

    torch_module = getattr(torch, spec.torch_module_name, None)
    if torch_module is None:
        raise RuntimeError(
            f"{DEVICE_BACKEND_ENV_VAR}={backend_name!r} resolved to "
            f"device_type={spec.device_type!r}, but torch has no module "
            f"{spec.torch_module_name!r}."
        )
    return torch_module, spec


def _resolve_device_type_candidates(
    torch: Any,
    device_type: str,
) -> "tuple[Any, DeviceSpec] | None":
    """Resolve a ``device_type`` to exactly one available spec, if possible."""
    candidates = _build_device_registry().get(device_type, ())
    if not candidates:
        return None

    available_candidates: list[tuple[Any, DeviceSpec]] = []
    for spec in candidates:
        if not spec.is_available():
            continue

        torch_module = getattr(torch, spec.torch_module_name, None)
        if torch_module is None:
            logger.warning(
                "backend [%s] for device [%s] is available, but torch module "
                "[%s] is not found.",
                spec.backend_name,
                spec.device_type,
                spec.torch_module_name,
            )
            continue
        available_candidates.append((torch_module, spec))

    if len(available_candidates) == 1:
        return available_candidates[0]

    if len(available_candidates) > 1:
        backend_names = ", ".join(
            sorted(spec.backend_name for _, spec in available_candidates)
        )
        raise RuntimeError(
            f"Multiple LMCache backends are available for device_type "
            f"{device_type!r}: {backend_names}. Set "
            f"{DEVICE_BACKEND_ENV_VAR}=<backend_name> to choose one explicitly."
        )
    return None


def _detect_device() -> "tuple[Any, str, str | None]":
    """Detect the available accelerator via the device registry.

    Returns:
        tuple[Any, str, str | None]: ``(torch_device_module, device_type,
            backend_name)``. When torch is not installed (CLI-only mode),
            returns ``(None, "cpu", None)``.
    """
    try:
        # Third Party
        import torch
    except ImportError as e:
        logger.warning("load torch failed, error is %s", e)
        return None, "cpu", None  # fallback for CLI-only environments

    env_backend_name = _get_env_choice(DEVICE_BACKEND_ENV_VAR)
    if env_backend_name is not None:
        torch_module, spec = _resolve_explicit_backend(torch, env_backend_name)
        return torch_module, spec.device_type, spec.backend_name

    env_device_type = _get_env_choice("DEVICE_TYPE")
    if env_device_type is not None:
        resolved = _resolve_device_type_candidates(torch, env_device_type)
        if resolved is not None:
            torch_module, spec = resolved
            return torch_module, spec.device_type, spec.backend_name
        logger.warning(
            "DEVICE_TYPE=%r is not available or not registered, "
            "falling back to auto-detection.",
            env_device_type,
        )

    for device_type in _build_device_registry():
        resolved = _resolve_device_type_candidates(torch, device_type)
        if resolved is not None:
            torch_module, spec = resolved
            return torch_module, spec.device_type, spec.backend_name

    # No accelerator found -- fall back to CPU stub
    # First Party
    from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice

    return StubCPUDevice("cpu"), "cpu", None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_device_spec(device_type: str) -> "DeviceSpec | None":
    """Return the resolved :class:`DeviceSpec` for *device_type*, if any."""
    specs = _build_device_registry().get(device_type, ())
    if not specs:
        return None
    if len(specs) == 1:
        return specs[0]

    env_backend_name = _get_env_choice(DEVICE_BACKEND_ENV_VAR)
    if env_backend_name is not None:
        spec = _build_backend_registry().get(env_backend_name)
        if spec is not None and spec.device_type == device_type:
            return spec

    available_specs = [spec for spec in specs if spec.is_available()]
    if len(available_specs) == 1:
        return available_specs[0]

    default_specs = [spec for spec in specs if spec.backend_name == device_type]
    if len(default_specs) == 1:
        return default_specs[0]
    return None


@lru_cache(maxsize=1)
def get_torch_device() -> tuple[Any, str]:
    """Return the cached ``(torch_dev, torch_device_type)`` pair.

    Lazy + memoized so that peers like :mod:`torch_ops` can safely
    import this helper at module top level: no work is performed until
    the tuple is actually needed.
    """
    torch_dev, torch_device_type, _ = _detect_device()
    logger.info("torch_dev=%s, torch_device_type=%s", torch_dev, torch_device_type)
    return torch_dev, torch_device_type


@lru_cache(maxsize=1)
def current_device_spec() -> "DeviceSpec":
    """Return the :class:`DeviceSpec` for the detected device.

    Falls back to a bare ``DeviceSpec()`` (no-op / all False semantics)
    when no registered accelerator backend matches.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

    _, device_type, backend_name = _detect_device()
    spec = _build_backend_registry().get(backend_name) if backend_name else None
    if spec is None:
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

# SPDX-License-Identifier: Apache-2.0
"""Registry + lazy module discovery for the kv_format package.

This module is the *single source of truth* for "which formats /
engines do we know about?". It owns:

* :data:`_SPECS_BY_ID` — ``format_id`` (str) -> Spec class (1-to-1).
* :data:`_SPECS_BY_GPU_KV_FORMAT` — ``GPUKVFormat`` enum -> Spec class
  (kept in sync; many call sites only have the C++ enum value).
* :data:`_DETECTORS_BY_ENGINE` — ``EngineType`` -> Detector instance.

Specs and detectors register themselves automatically via the
``__init_subclass__`` hooks on their base classes. This module merely
*triggers* the imports of every module under ``specs/`` and
``detectors/`` lazily on first use, mirroring the
``storage_controllers`` / ``l2_adapters`` / ``record_strategies``
discovery pattern already used elsewhere in the codebase.
"""

# Standard
from typing import TYPE_CHECKING, Optional
import importlib
import pkgutil

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.base import KVFormatSpec
    from lmcache.v1.gpu_connector.kv_format.detection_base import EngineDetector
    from lmcache.v1.gpu_connector.kv_format.types import (
        DiscoverableKVCache,
    )
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


_SPECS_BY_ID: dict[str, type["KVFormatSpec"]] = {}
_SPECS_BY_GPU_KV_FORMAT: dict["lmc_ops.GPUKVFormat", type["KVFormatSpec"]] = {}
_DETECTORS_BY_ENGINE: dict[EngineType, "EngineDetector"] = {}

_LOADED = False
_FAILED_SPEC_MODULES: list[tuple[str, str]] = []
_FAILED_DETECTOR_MODULES: list[tuple[str, str]] = []


# ----------------------------------------------------------------------
# Internal registration hooks. Called from base classes' __init_subclass__.
# ----------------------------------------------------------------------
def register_spec_class(cls: type["KVFormatSpec"]) -> None:
    """Register a concrete :class:`KVFormatSpec` subclass.

    Invoked automatically by :meth:`KVFormatSpec.__init_subclass__`;
    user code should never call it directly.
    """
    fid = cls.format_id
    existing = _SPECS_BY_ID.get(fid)
    if existing is not None and existing is not cls:
        raise RuntimeError(
            "Duplicate KVFormatSpec format_id %r: %s vs %s" % (fid, existing, cls)
        )
    _SPECS_BY_ID[fid] = cls

    gpu_fmt = cls.gpu_kv_format
    existing_enum = _SPECS_BY_GPU_KV_FORMAT.get(gpu_fmt)
    if existing_enum is not None and existing_enum is not cls:
        raise RuntimeError(
            "Duplicate KVFormatSpec gpu_kv_format %r: %s vs %s"
            % (gpu_fmt, existing_enum, cls)
        )
    _SPECS_BY_GPU_KV_FORMAT[gpu_fmt] = cls


def register_detector_class(cls: type["EngineDetector"]) -> None:
    """Register an :class:`EngineDetector` subclass (instantiates it)."""
    engine = cls.engine
    if engine in _DETECTORS_BY_ENGINE and not isinstance(
        _DETECTORS_BY_ENGINE[engine], cls
    ):
        raise RuntimeError(
            "Duplicate EngineDetector for engine %r: %s vs %s"
            % (engine, type(_DETECTORS_BY_ENGINE[engine]), cls)
        )
    _DETECTORS_BY_ENGINE[engine] = cls()


# ----------------------------------------------------------------------
# Lazy discovery. Idempotent.
# ----------------------------------------------------------------------
def _discover(subpackage: str, failures: list[tuple[str, str]]) -> None:
    pkg = importlib.import_module("lmcache.v1.gpu_connector.kv_format." + subpackage)
    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(pkg.__path__):
        if mod_name.startswith("_"):
            continue
        full = pkg.__name__ + "." + mod_name
        try:
            importlib.import_module(full)
        except Exception as e:  # noqa: BLE001 — log and continue.
            failures.append((full, repr(e)))
            logger.warning(
                "Failed to load kv_format.%s module %s: %s",
                subpackage,
                full,
                e,
            )


def ensure_loaded() -> None:
    """Idempotently trigger lazy discovery of all spec and detector modules.

    Walks the ``specs/`` and ``detectors/`` subpackages once, importing
    every public module so that the ``__init_subclass__`` hooks register
    their classes. Subsequent calls are no-ops. Failures from individual
    modules are logged at WARNING and recorded in
    :data:`_FAILED_SPEC_MODULES` / :data:`_FAILED_DETECTOR_MODULES`
    instead of being raised, so a broken third-party plugin can never
    take down the whole registry.
    """
    global _LOADED
    if _LOADED:
        return
    _discover("specs", _FAILED_SPEC_MODULES)
    _discover("detectors", _FAILED_DETECTOR_MODULES)
    _LOADED = True
    logger.debug(
        "kv_format registry loaded: specs=%s detectors=%s",
        list(_SPECS_BY_ID.keys()),
        list(_DETECTORS_BY_ENGINE.keys()),
    )


# ----------------------------------------------------------------------
# Public lookup API.
# ----------------------------------------------------------------------
def all_format_ids() -> list[str]:
    """Return every registered ``format_id`` (triggers lazy discovery)."""
    ensure_loaded()
    return list(_SPECS_BY_ID.keys())


def all_gpu_kv_formats() -> list["lmc_ops.GPUKVFormat"]:
    """Return every registered ``GPUKVFormat`` enum value."""
    ensure_loaded()
    return list(_SPECS_BY_GPU_KV_FORMAT.keys())


def get_spec_class_by_id(format_id: str) -> Optional[type["KVFormatSpec"]]:
    """Look up the spec class by its string ``format_id``.

    Returns ``None`` when no such format is registered.
    """
    ensure_loaded()
    return _SPECS_BY_ID.get(format_id)


def get_spec_class(
    fmt: "lmc_ops.GPUKVFormat",
) -> Optional[type["KVFormatSpec"]]:
    """Look up the spec class by C++ ``GPUKVFormat`` enum value.

    Returns ``None`` when no spec is registered for ``fmt``.
    """
    ensure_loaded()
    return _SPECS_BY_GPU_KV_FORMAT.get(fmt)


def get_spec(
    kv_caches: "DiscoverableKVCache",
    fmt: "lmc_ops.GPUKVFormat",
) -> "KVFormatSpec":
    """Construct the spec instance for ``fmt`` bound to ``kv_caches``.

    Args:
        kv_caches: The KV cache structure the spec will introspect.
        fmt: The C++ ``GPUKVFormat`` enum value identifying the layout.

    Returns:
        A fresh :class:`KVFormatSpec` instance bound to ``kv_caches``.

    Raises:
        ValueError: If ``fmt`` has no registered spec (the error
            message includes the loaded format ids and any module
            import failures, to aid debugging missing-plugin cases).
    """
    cls = get_spec_class(fmt)
    if cls is None:
        raise ValueError(
            "Unknown GPU KV Format: %r. Loaded specs: %s. Failed spec "
            "modules: %s" % (fmt, list(_SPECS_BY_ID.keys()), _FAILED_SPEC_MODULES)
        )
    return cls(kv_caches)


def get_detector(engine: EngineType) -> Optional["EngineDetector"]:
    """Return the registered detector instance for ``engine``, or ``None``."""
    ensure_loaded()
    return _DETECTORS_BY_ENGINE.get(engine)


def supported_engines() -> list[EngineType]:
    """Return every ``EngineType`` that has a registered detector."""
    ensure_loaded()
    return list(_DETECTORS_BY_ENGINE.keys())


def loaded_format_ids() -> list[str]:
    """Snapshot of currently-loaded format_ids (does not trigger load)."""
    return list(_SPECS_BY_ID.keys())


def failed_module_reports() -> dict[str, list[tuple[str, str]]]:
    """Return module names and repr(exception) for any modules that
    failed to import during discovery. Mainly for diagnostics in error
    paths.
    """
    return {
        "specs": list(_FAILED_SPEC_MODULES),
        "detectors": list(_FAILED_DETECTOR_MODULES),
    }


# ----------------------------------------------------------------------
# Test-only: tear down a registration.
# ----------------------------------------------------------------------
def unregister_spec(format_id: str) -> None:
    """Drop a registered spec by ``format_id``. Used by tests."""
    cls = _SPECS_BY_ID.pop(format_id, None)
    if cls is not None:
        _SPECS_BY_GPU_KV_FORMAT.pop(cls.gpu_kv_format, None)

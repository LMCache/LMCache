# SPDX-License-Identifier: Apache-2.0
"""Discover async GDS implementations by module name and import them on demand.

Add a public module exporting a Backend subclass to the gds_backends package;
no factory, interface, context, or configuration registry needs updating.
"""

# Standard
import importlib
import pkgutil

# First Party
from lmcache.v1.gpu_connector import gds_backends
from lmcache.v1.gpu_connector._gds_async import GDSBackend


def available_backends() -> tuple[str, ...]:
    """List backend module names without importing their implementations."""
    return tuple(
        sorted(
            module.name
            for module in pkgutil.iter_modules(gds_backends.__path__)
            if not module.name.startswith("_")
        )
    )


def create_backend(name: str) -> GDSBackend:
    """Construct and validate a backend without loading its native driver.

    Explicit selection imports only the requested implementation. ``auto``
    imports candidates in name order until one accepts default selection.
    Unknown names and incompatible environments raise ValueError. No backend
    instance is cached, and discovery does not import optional dependencies.
    """
    names = available_backends()
    if name == "auto":
        for candidate in names:
            backend_class = _load_backend_class(candidate)
            if backend_class.is_default():
                break
        else:
            raise ValueError("no default GDS backend for this environment")
    else:
        if name not in names:
            raise ValueError(f"unsupported GDS L1 backend: {name}")
        backend_class = _load_backend_class(name)
    backend = backend_class()
    backend.validate_environment()
    return backend


def _load_backend_class(name: str) -> type[GDSBackend]:
    module = importlib.import_module(f"{gds_backends.__name__}.{name}")
    try:
        backend_class = module.Backend
    except AttributeError as error:
        raise TypeError(f"GDS backend {name!r} must export a Backend class") from error
    if not isinstance(backend_class, type) or not issubclass(backend_class, GDSBackend):
        raise TypeError(f"GDS backend {name!r} must export a GDSBackend subclass")
    return backend_class

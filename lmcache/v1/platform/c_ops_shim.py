# SPDX-License-Identifier: Apache-2.0
"""Builder for the backward-compat ``lmcache.c_ops`` shim module.

This module contains the *construction* logic only. Registration into
``sys.modules`` and attaching the shim as an attribute on the top-level
``lmcache`` package remains the responsibility of ``lmcache/__init__.py``,
because the parent-attribute mount point (``lmcache.c_ops = shim``) is
part of the top-level package's public contract and must be set from the
top-level module's own globals.
"""

# Future
from __future__ import annotations

# Standard
import importlib
import types

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import resolve_device_ops

logger = init_logger(__name__)


def build_c_ops_shim(device_type: str) -> types.ModuleType:
    """Build a live ``lmcache.c_ops`` view over the resolved
    :class:`DeviceOps` **instance** for *device_type*.

    Calls :meth:`DeviceOps.ensure_native` on the singleton to lazily bind
    the compiled extension, then returns a module whose attribute access
    resolves through:

    1. Real attributes copied from the compiled native module (if any)
       -- so plan types like ``KernelGroupSpec`` and native pybind
       callables are always the real thing, matching the pre-refactor
       ``lmcache.c_ops`` contract.
    2. A ``__getattr__`` fallback to the resolved :class:`DeviceOps`
       instance -- for ops that only exist as torch-baseline instance
       methods (no native counterpart).

    The PEP 562 module-level ``__getattr__`` hook is used so that later
    ``setattr(shim, ...)`` patches (e.g. from tests) remain visible for
    attributes not shadowed by the native module.

    The caller is responsible for registering the returned module into
    ``sys.modules["lmcache.c_ops"]`` and mounting it as ``lmcache.c_ops``
    on the top-level package.
    """
    ops = resolve_device_ops(device_type)
    ops.ensure_native()

    # Best-effort: import the compiled native module directly. When
    # ``ensure_native`` already imported it, this returns the cached
    # entry from ``sys.modules``; otherwise Python's import machinery
    # loads the ``.so`` fresh. Missing / unbuilt -> ``None`` and we
    # fall back to the torch baseline via ``ops``.
    native_mod: types.ModuleType | None = None
    try:
        native_mod = importlib.import_module("lmcache.c_ops")
    except Exception as exc:
        logger.debug("Native lmcache.c_ops not importable: %s", exc)

    shim = types.ModuleType("lmcache.c_ops")
    if native_mod is not None:
        # Copy every public symbol from the native module onto the shim
        # so ``shim.KernelGroupSpec`` etc. resolve without any indirection.
        for _name in dir(native_mod):
            if _name.startswith("_"):
                continue
            setattr(shim, _name, getattr(native_mod, _name))

    shim.__getattr__ = lambda name: getattr(ops, name)  # type: ignore[method-assign]
    shim.__dir__ = lambda: sorted(  # type: ignore[method-assign]
        set(vars(shim).keys()) | set(dir(ops))
    )
    return shim

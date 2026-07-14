# SPDX-License-Identifier: Apache-2.0

# Standard
import importlib
import sys
import types

# First Party
from lmcache.logging import init_logger

# --------------------------
# Backend instance & Device detection
# --------------------------
from lmcache.v1.platform import torch_dev as torch_dev
from lmcache.v1.platform import torch_device_type as torch_device_type

try:
    # First Party
    from lmcache._version import __version__
except ImportError:
    __version__ = "unknown"

logger = init_logger(__name__)

__all__ = ["__version__", "torch_dev", "torch_device_type"]


# --------------------------
# Backward-compat ``lmcache.c_ops`` shim
# --------------------------
def _install_c_ops_shim() -> None:
    """Register ``lmcache.c_ops`` as a live view over the resolved
    :class:`DeviceOps` **instance** for the current device.

    Calls :meth:`DeviceOps.ensure_native` on the singleton to lazily bind
    the compiled extension, then installs a module whose attribute access
    resolves through:

    1. Real attributes copied from the compiled native module (if any)
       -- so plan types like ``KernelGroupSpec`` and native pybind
       callables are always the real thing, matching the pre-refactor
       ``lmcache.c_ops`` contract.
    2. A ``__getattr__`` fallback to the resolved :class:`DeviceOps`
       instance -- for ops that only exist as torch-baseline instance
       methods (no native counterpart).

    The PEP 562 module-level ``__getattr__`` hook is used so that later
    ``setattr(ops, ...)`` patches (e.g. from tests) remain visible for
    attributes not shadowed by the native module.
    """
    # First Party
    from lmcache.v1.platform import resolve_device_ops

    ops = resolve_device_ops(torch_device_type)
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
    sys.modules["lmcache.c_ops"] = shim
    globals()["c_ops"] = shim  # parent attr for IMPORT_FROM bytecode


try:
    _install_c_ops_shim()
except Exception as exc:
    logger.warning(
        "No compute backend loaded; CLI-only mode (torch/numba not installed). "
        "Reason: %s",
        exc,
    )

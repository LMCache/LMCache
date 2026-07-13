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


# Sentinel used by the shim to distinguish "attribute missing" from
# "attribute is legitimately ``None``".
_MISSING = object()


# --------------------------
# Backward-compat ``lmcache.c_ops`` shim
# --------------------------
def _install_c_ops_shim() -> None:
    """Register ``lmcache.c_ops`` as a live view over the resolved
    :class:`DeviceOps` **instance** for the current device.

    Calls :meth:`DeviceOps.ensure_native` on the singleton to lazily bind
    the compiled extension, then installs a module whose ``__getattr__``
    forwards to that instance. Since all ops are instance methods, calls
    like ``c_ops.multi_layer_kv_transfer(...)`` resolve to bound methods
    on the singleton, giving us proper polymorphism while keeping the
    existing module-level API surface untouched.

    If the compiled ``lmcache.c_ops`` native module is importable, keep a
    direct reference to it as a **secondary fallback**: whenever the
    resolved ``ops`` instance yields a pure-Python ``NativePlanType``
    stub (e.g. because ``_bind_native`` didn't rebind that class-level
    alias for whatever reason), transparently fall back to the native
    module. This preserves the ``lmcache.c_ops`` contract for callers
    that construct plan types like ``KernelGroupSpec`` / ``StagingCopy``.

    The PEP 562 module-level ``__getattr__`` hook is used so that later
    ``setattr(ops, ...)`` patches (e.g. from tests) remain visible.
    """
    # First Party
    from lmcache.v1.platform import resolve_device_ops
    from lmcache.v1.platform.ops_types import NativePlanType

    ops = resolve_device_ops(torch_device_type)
    ops.ensure_native()

    # Best-effort: keep a direct reference to the compiled native module
    # as a fallback for names the ops instance couldn't provide natively.
    native_mod: types.ModuleType | None = None
    try:
        native_mod = importlib.import_module("lmcache.c_ops")
    except Exception as exc:
        logger.debug("Native lmcache.c_ops not importable: %s", exc)

    def _shim_getattr(name: str):
        val = getattr(ops, name, _MISSING)
        if val is _MISSING or (
            isinstance(val, type) and issubclass(val, NativePlanType)
        ):
            if native_mod is not None:
                native_val = getattr(native_mod, name, _MISSING)
                if native_val is not _MISSING:
                    return native_val
        if val is _MISSING:
            raise AttributeError(name)
        return val

    shim = types.ModuleType("lmcache.c_ops")
    shim.__getattr__ = _shim_getattr  # type: ignore[method-assign]
    shim.__dir__ = lambda: dir(ops)  # type: ignore[method-assign]
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

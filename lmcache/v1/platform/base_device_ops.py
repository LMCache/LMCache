# SPDX-License-Identifier: Apache-2.0
"""The unified per-device ops abstraction (the ``lmcache.c_ops`` surface).

:class:`DeviceOps` owns the full op contract with a device-agnostic torch
baseline from :mod:`lmcache.v1.platform.torch_ops`.  Each op is a real
``staticmethod`` on the class, resolved via normal MRO.  Accelerator subclasses
override individual ops or call :func:`bind_native` (usually via
:meth:`ensure_native`) to bulk-bind a compiled ``.so``.

The class itself is the single source of truth: op names are discovered by
iterating :meth:`DeviceOps.iter_ops` / :meth:`DeviceOps.iter_native_types`,
so there is no parallel string list to keep in sync.

:func:`bind_native` is a class-decorator factory: ``bind_native(module)(cls)``
overwrites every staticmethod op found in *module* onto *cls*.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable, ClassVar, Iterator

# First Party
from lmcache.v1.platform import ops_types, torch_ops
from lmcache.v1.platform.ops_types import (
    BatchStep,
    EngineKVFormat,
    KernelGroupSpec,
    LaunchVar,
    PageBufferShapeDesc,
    StagingCopy,
    TransferDirection,
    set_shape_desc_dtype,
)

# ─── Decorator ─────────────────────────────────────────────────────────


def bind_native(module: object) -> Callable[[type[DeviceOps]], type[DeviceOps]]:
    """Class-decorator factory: bulk-bind native ops from *module* onto a class.

    Usage::

        @bind_native(compiled_module)
        class MyDeviceOps(DeviceOps): ...

    Or applied lazily inside ``ensure_native()``::

        bind_native(native)(cls)

    The class itself is the SSOT: for every op name yielded by
    :meth:`DeviceOps.iter_ops`, if *module* exports it, the class attribute
    is overwritten with ``staticmethod(fn)``.  Types yielded by
    :meth:`DeviceOps.iter_native_types` are also rebound (for pybind enum
    identity).

    Args:
    module: The compiled native module or object containing the native ops.

    Returns:
    A decorator function that takes a DeviceOps subclass and returns it
    with native ops bound.
    """

    def decorator(cls: type[DeviceOps]) -> type[DeviceOps]:
        for name in cls.iter_ops():
            fn = getattr(module, name, None)
            if fn is not None:
                setattr(cls, name, staticmethod(fn))
        for type_name in cls.iter_native_types():
            t = getattr(module, type_name, None)
            if t is not None:
                setattr(cls, type_name, t)
        # Maintain GPUKVFormat alias
        ekf = getattr(module, "EngineKVFormat", None)
        if ekf is not None:
            cls.GPUKVFormat = ekf  # type: ignore[attr-defined]
        return cls

    return decorator


# ─── Base class ────────────────────────────────────────────────────────


class DeviceOps:
    """Strategy base: per-device ops resolved via MRO.

    Every op is a ``staticmethod`` on the class delegating to the torch
    baseline in :mod:`~lmcache.v1.platform.torch_ops`.
    Accelerator subclasses either:

    - Override individual ops (e.g. MUSA overrides one transfer op).
    - Call ``bind_native(module)(cls)`` in :meth:`ensure_native` to
      bulk-overwrite all ops the compiled extension exports.

    The ``lmcache.c_ops`` shim reads attributes directly off the resolved
    class after :meth:`ensure_native` has been called.
    """

    device_type: ClassVar[str] = ""  # base is unregistered
    _native_bound: ClassVar[bool] = False

    # Names that are class metadata / non-op helpers rather than ops.
    # ``set_shape_desc_dtype`` is a Python-only ``staticmethod`` that must not
    # be treated as a native op (native modules do not export it).
    _NON_OP_STATICMETHODS: ClassVar[frozenset[str]] = frozenset(
        {"set_shape_desc_dtype"}
    )
    # Type attributes that are aliases; ``bind_native`` should not rebind them
    # as native types (``GPUKVFormat`` is a back-compat alias managed manually).
    _ALIAS_TYPES: ClassVar[frozenset[str]] = frozenset({"GPUKVFormat"})

    # ── Shared types (explicit for static analysis) ────────────────────
    TransferDirection = TransferDirection
    EngineKVFormat = EngineKVFormat
    GPUKVFormat = EngineKVFormat  # back-compat alias
    PageBufferShapeDesc = PageBufferShapeDesc
    StagingCopy = StagingCopy
    LaunchVar = LaunchVar
    BatchStep = BatchStep
    KernelGroupSpec = KernelGroupSpec
    set_shape_desc_dtype = staticmethod(set_shape_desc_dtype)

    # ── Introspection (SSOT for op / type names) ─────────────────────

    @classmethod
    def iter_ops(cls) -> Iterator[str]:
        """Yield the names of every op defined on the class.

        An op is any public ``staticmethod`` on the class, excluding helpers
        listed in :attr:`_NON_OP_STATICMETHODS`.  The class hierarchy is the
        single source of truth: adding a new op to :class:`DeviceOps` (or a
        subclass) automatically exposes it here without touching any parallel
        list.
        """
        seen: set[str] = set()
        for klass in cls.__mro__:
            for name, raw in vars(klass).items():
                if name.startswith("_") or name in seen:
                    continue
                if name in cls._NON_OP_STATICMETHODS:
                    seen.add(name)
                    continue
                # ``staticmethod`` in the class dict wraps the callable.
                if isinstance(raw, staticmethod):
                    seen.add(name)
                    yield name

    @classmethod
    def iter_native_types(cls) -> Iterator[str]:
        """Yield the names of native-rebindable type attributes on the class.

        A native type is any public class attribute whose value is a ``type``,
        excluding aliases listed in :attr:`_ALIAS_TYPES`.
        """
        seen: set[str] = set()
        for klass in cls.__mro__:
            for name, raw in vars(klass).items():
                if name.startswith("_") or name in seen:
                    continue
                if name in cls._ALIAS_TYPES:
                    seen.add(name)
                    continue
                if isinstance(raw, type):
                    seen.add(name)
                    yield name

    # ── Lazy native binding ───────────────────────────────────────────

    @classmethod
    def ensure_native(cls) -> None:
        """Attempt to load and bind the compiled native extension.

        Subclasses override this to ``importlib.import_module(...)`` then
        call ``bind_native(native)(cls)``.  Guarded by ``_native_bound``
        so it runs at most once per class.
        """


# ─── Populate baseline from torch_ops / ops_types ──────────────────────
# ``torch_ops`` and ``ops_types`` are the source of truth for the op /
# type surface.  Rather than restate every name on the class, mirror them
# once at import time: any function defined in ``torch_ops`` becomes a
# staticmethod on ``DeviceOps``, and any class defined in ``ops_types``
# becomes a class attribute.  Attributes already set on the class body
# (aliases, non-op helpers) are preserved.


def _bind_base_device_ops(
    module: object,
    cls: type,
    kind: str,  # "callable" or "type"
) -> None:
    """Copy public members of *module* onto *cls* without clobbering existing
    class-body definitions.

    Only members whose ``__module__`` matches ``module.__name__`` are copied,
    so re-exports (e.g. ``set_shape_desc_dtype`` re-exported by ``torch_ops``
    from ``ops_types``) are attributed to their defining module only.
    """
    mod_name = getattr(module, "__name__", None)
    for name in dir(module):
        if name.startswith("_"):
            continue
        if name in vars(cls):  # respect manual class-body definitions
            continue
        obj = getattr(module, name)
        if getattr(obj, "__module__", None) != mod_name:
            continue
        if kind == "callable":
            if callable(obj) and not isinstance(obj, type):
                setattr(cls, name, staticmethod(obj))
        elif kind == "type":
            if isinstance(obj, type):
                setattr(cls, name, obj)


"""
Bind every public function from torch_ops as a staticmethod on DeviceOps
"""
_bind_base_device_ops(torch_ops, DeviceOps, kind="callable")

"""
Bind every public class from ops_types as a class attribute on DeviceOps
"""
_bind_base_device_ops(ops_types, DeviceOps, kind="type")

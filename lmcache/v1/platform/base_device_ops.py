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
from lmcache.logging import init_logger
from lmcache.v1.platform import torch_ops as _ops
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

logger = init_logger(__name__)

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
        bound_ops = 0
        bound_types = 0
        for name in cls.iter_ops():
            fn = getattr(module, name, None)
            if fn is not None:
                setattr(cls, name, staticmethod(fn))
                bound_ops += 1
        for type_name in cls.iter_native_types():
            t = getattr(module, type_name, None)
            if t is not None:
                setattr(cls, type_name, t)
                bound_types += 1
        # Maintain GPUKVFormat alias
        ekf = getattr(module, "EngineKVFormat", None)
        if ekf is not None:
            cls.GPUKVFormat = ekf  # type: ignore[attr-defined]
        logger.debug(
            "bind_native: %s <- %s (%d ops, %d types)",
            cls.__name__,
            getattr(module, "__name__", repr(module)),
            bound_ops,
            bound_types,
        )
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

    #: All op names on this class (auto-populated at end of module).
    OPS: ClassVar[frozenset[str]]
    #: All native type names on this class (auto-populated at end of module).
    NATIVE_TYPES: ClassVar[frozenset[str]]

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

    # ── Ops: memory alloc / free ─────────────────────────────────────
    alloc_hugepage_pinned_numa_ptr = staticmethod(_ops.alloc_hugepage_pinned_numa_ptr)
    alloc_hugepage_pinned_ptr = staticmethod(_ops.alloc_hugepage_pinned_ptr)
    alloc_numa_ptr = staticmethod(_ops.alloc_numa_ptr)
    alloc_pinned_numa_ptr = staticmethod(_ops.alloc_pinned_numa_ptr)
    alloc_pinned_ptr = staticmethod(_ops.alloc_pinned_ptr)
    alloc_shm_pinned_ptr = staticmethod(_ops.alloc_shm_pinned_ptr)
    free_hugepage_pinned_numa_ptr = staticmethod(_ops.free_hugepage_pinned_numa_ptr)
    free_hugepage_pinned_ptr = staticmethod(_ops.free_hugepage_pinned_ptr)
    free_numa_ptr = staticmethod(_ops.free_numa_ptr)
    free_pinned_numa_ptr = staticmethod(_ops.free_pinned_numa_ptr)
    free_pinned_ptr = staticmethod(_ops.free_pinned_ptr)
    free_shm_pinned_ptr = staticmethod(_ops.free_shm_pinned_ptr)

    # ── Ops: KV transfer ─────────────────────────────────────────────
    execute_object_group_transfer = staticmethod(_ops.execute_object_group_transfer)
    multi_layer_block_kv_transfer = staticmethod(_ops.multi_layer_block_kv_transfer)
    multi_layer_kv_transfer = staticmethod(_ops.multi_layer_kv_transfer)
    multi_layer_kv_transfer_unilateral = staticmethod(_ops.multi_layer_kv_transfer_unilateral)
    single_layer_kv_transfer = staticmethod(_ops.single_layer_kv_transfer)
    single_layer_kv_transfer_sgl = staticmethod(_ops.single_layer_kv_transfer_sgl)

    # ── Ops: KV reshape ──────────────────────────────────────────────
    load_and_reshape_flash = staticmethod(_ops.load_and_reshape_flash)
    reshape_and_cache_back_flash = staticmethod(_ops.reshape_and_cache_back_flash)

    # ── Ops: codec ───────────────────────────────────────────────────
    calculate_cdf = staticmethod(_ops.calculate_cdf)
    decode_fast_new = staticmethod(_ops.decode_fast_new)
    decode_fast_prefsum = staticmethod(_ops.decode_fast_prefsum)
    encode_fast_new = staticmethod(_ops.encode_fast_new)

    # ── Ops: format query ────────────────────────────────────────────
    is_cross_layer = staticmethod(_ops.is_cross_layer)
    is_kv_list = staticmethod(_ops.is_kv_list)
    is_layer_list = staticmethod(_ops.is_layer_list)
    is_mla = staticmethod(_ops.is_mla)

    # ── Ops: async / event recording ─────────────────────────────────
    drain_recorded_completions = staticmethod(_ops.drain_recorded_completions)
    drain_recorded_events = staticmethod(_ops.drain_recorded_events)
    record_completion_on_stream = staticmethod(_ops.record_completion_on_stream)
    record_event_on_stream = staticmethod(_ops.record_event_on_stream)

    # ── Ops: misc ────────────────────────────────────────────────────
    batched_memcpy = staticmethod(_ops.batched_memcpy)
    get_gpu_pci_bus_id = staticmethod(_ops.get_gpu_pci_bus_id)
    lmcache_memcpy_async = staticmethod(_ops.lmcache_memcpy_async)
    rotary_embedding_k_fused = staticmethod(_ops.rotary_embedding_k_fused)

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

        Subclasses override this to ``lmcache.c_ops`` then call
        ``bind_native(native)(cls)``.  Guarded by ``_native_bound``
        so it runs at most once per class.
        """


# Freeze the op/type name sets for visibility and fast membership checks.
DeviceOps.OPS = frozenset(DeviceOps.iter_ops())
DeviceOps.NATIVE_TYPES = frozenset(DeviceOps.iter_native_types())

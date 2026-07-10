# SPDX-License-Identifier: Apache-2.0
"""The unified per-device ops abstraction (the ``lmcache.c_ops`` surface).

:class:`DeviceOps` owns the full op contract (:data:`OPS`) with a
device-agnostic torch baseline from :mod:`lmcache.v1.platform._torch_ops`.
Each op is a real ``staticmethod`` on the class, resolved via normal MRO.
Accelerator subclasses override individual ops or call :func:`bind_native`
(usually via :meth:`_ensure_native`) to bulk-bind a compiled ``.so``.

:func:`bind_native` is a class-decorator factory: ``bind_native(module)(cls)``
overwrites every OPS-listed staticmethod found in *module* onto *cls*.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform import _torch_ops
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

# ─── Contract ──────────────────────────────────────────────────────────

#: Every name a device provides natively or inherits from the torch baseline.
#: Single source of truth for the parity test, bind_native, and the c_ops shim.
OPS: frozenset[str] = frozenset(
    {
        # ── Memory alloc / free ───────────────────────────────
        "alloc_hugepage_pinned_numa_ptr",
        "alloc_hugepage_pinned_ptr",
        "alloc_numa_ptr",
        "alloc_pinned_numa_ptr",
        "alloc_pinned_ptr",
        "alloc_shm_pinned_ptr",
        "free_hugepage_pinned_numa_ptr",
        "free_hugepage_pinned_ptr",
        "free_numa_ptr",
        "free_pinned_numa_ptr",
        "free_pinned_ptr",
        "free_shm_pinned_ptr",
        # ── KV transfer ───────────────────────────────────────
        "execute_object_group_transfer",
        "multi_layer_block_kv_transfer",
        "multi_layer_kv_transfer",
        "multi_layer_kv_transfer_unilateral",
        "single_layer_kv_transfer",
        "single_layer_kv_transfer_sgl",
        # ── KV reshape ────────────────────────────────────────
        "load_and_reshape_flash",
        "reshape_and_cache_back_flash",
        # ── Codec ─────────────────────────────────────────────
        "calculate_cdf",
        "decode_fast_new",
        "decode_fast_prefsum",
        "encode_fast_new",
        # ── Format query ──────────────────────────────────────
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
        # ── Async / event recording ───────────────────────────
        "drain_recorded_completions",
        "drain_recorded_events",
        "record_completion_on_stream",
        "record_event_on_stream",
        # ── Misc ──────────────────────────────────────────────
        "batched_memcpy",
        "get_gpu_pci_bus_id",
        "lmcache_memcpy_async",
        "rotary_embedding_k_fused",
    }
)

#: Types that :func:`bind_native` rebinds from the compiled module.
_NATIVE_TYPES: tuple[str, ...] = (
    "TransferDirection",
    "EngineKVFormat",
    "PageBufferShapeDesc",
    "StagingCopy",
    "LaunchVar",
    "BatchStep",
    "KernelGroupSpec",
)


# ─── Decorator ─────────────────────────────────────────────────────────


def bind_native(module: object):
    """Class-decorator factory: bulk-bind native ops from *module* onto a class.

    Usage::

        @bind_native(compiled_module)
        class MyDeviceOps(DeviceOps): ...

    Or applied lazily inside ``_ensure_native()``::

        bind_native(native)(cls)

    For each name in :data:`OPS`, if *module* exports it, the class attribute
    is overwritten with ``staticmethod(fn)``.  Types listed in
    :data:`_NATIVE_TYPES` are also rebound (for pybind enum identity).
    """

    def decorator(cls: type[DeviceOps]) -> type[DeviceOps]:
        for name in OPS:
            fn = getattr(module, name, None)
            if fn is not None:
                setattr(cls, name, staticmethod(fn))
        for type_name in _NATIVE_TYPES:
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

_ops = _torch_ops  # alias for line-length compliance


class DeviceOps:
    """Strategy base: per-device ops resolved via MRO.

    Every op in :data:`OPS` is a ``staticmethod`` on the class delegating to
    the torch baseline in :mod:`~lmcache.v1.platform._torch_ops`.
    Accelerator subclasses either:

    - Override individual ops (e.g. MUSA overrides one transfer op).
    - Call ``bind_native(module)(cls)`` in :meth:`_ensure_native` to
      bulk-overwrite all ops the compiled extension exports.

    The ``lmcache.c_ops`` shim reads attributes directly off the resolved
    class after :meth:`_ensure_native` has been called.
    """

    device_type: ClassVar[str] = ""  # base is unregistered
    _native_bound: ClassVar[bool] = False

    # ── Shared types ──────────────────────────────────────────────────
    TransferDirection = TransferDirection
    EngineKVFormat = EngineKVFormat
    GPUKVFormat = EngineKVFormat  # back-compat alias
    PageBufferShapeDesc = PageBufferShapeDesc
    set_shape_desc_dtype = staticmethod(set_shape_desc_dtype)
    StagingCopy = StagingCopy
    LaunchVar = LaunchVar
    BatchStep = BatchStep
    KernelGroupSpec = KernelGroupSpec

    # ── Memory alloc / free ──────────────────────────────────────────
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

    # ── KV transfer ───────────────────────────────────────────────
    execute_object_group_transfer = staticmethod(_ops.execute_object_group_transfer)
    multi_layer_block_kv_transfer = staticmethod(_ops.multi_layer_block_kv_transfer)
    multi_layer_kv_transfer = staticmethod(_ops.multi_layer_kv_transfer)
    multi_layer_kv_transfer_unilateral = staticmethod(
        _ops.multi_layer_kv_transfer_unilateral
    )
    single_layer_kv_transfer = staticmethod(_ops.single_layer_kv_transfer)
    single_layer_kv_transfer_sgl = staticmethod(_ops.single_layer_kv_transfer_sgl)

    # ── KV reshape ────────────────────────────────────────────────
    load_and_reshape_flash = staticmethod(_ops.load_and_reshape_flash)
    reshape_and_cache_back_flash = staticmethod(_ops.reshape_and_cache_back_flash)

    # ── Codec ─────────────────────────────────────────────────────
    calculate_cdf = staticmethod(_ops.calculate_cdf)
    decode_fast_new = staticmethod(_ops.decode_fast_new)
    decode_fast_prefsum = staticmethod(_ops.decode_fast_prefsum)
    encode_fast_new = staticmethod(_ops.encode_fast_new)

    # ── Format query ──────────────────────────────────────────────
    is_cross_layer = staticmethod(_ops.is_cross_layer)
    is_kv_list = staticmethod(_ops.is_kv_list)
    is_layer_list = staticmethod(_ops.is_layer_list)
    is_mla = staticmethod(_ops.is_mla)

    # ── Async / event recording ───────────────────────────────────
    drain_recorded_completions = staticmethod(_ops.drain_recorded_completions)
    drain_recorded_events = staticmethod(_ops.drain_recorded_events)
    record_completion_on_stream = staticmethod(_ops.record_completion_on_stream)
    record_event_on_stream = staticmethod(_ops.record_event_on_stream)

    # ── Misc ──────────────────────────────────────────────────────
    batched_memcpy = staticmethod(_ops.batched_memcpy)
    get_gpu_pci_bus_id = staticmethod(_ops.get_gpu_pci_bus_id)
    lmcache_memcpy_async = staticmethod(_ops.lmcache_memcpy_async)
    rotary_embedding_k_fused = staticmethod(_ops.rotary_embedding_k_fused)

    # ── Lazy native binding ───────────────────────────────────────────

    @classmethod
    def _ensure_native(cls) -> None:
        """Lazily load and bind native ops. No-op on the base class.

        Subclasses override this to import their compiled extension and
        call ``bind_native(native)(cls)``.  Guarded by ``_native_bound``
        so it runs at most once per class.
        """

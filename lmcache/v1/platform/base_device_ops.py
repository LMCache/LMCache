# SPDX-License-Identifier: Apache-2.0
"""The unified per-device ops abstraction (the ``lmcache.c_ops`` surface).

:class:`DeviceOps` owns the full op contract (:data:`OPS`) with a
device-agnostic torch baseline migrated into
:mod:`lmcache.v1.platform._torch_ops`. The base class *is* the CPU/torch
backend: every op is an explicit thin method delegating to ``_torch_ops`` so
the contract stays visible and grep-able. Accelerator subclasses set
:attr:`device_type` and either override hot ops in Python or bind a whole
compiled module via :meth:`DeviceOps._bind_native`, whose native callables
shadow the baseline at instance level while unbound ops keep the torch
implementation.

The base itself has an empty ``device_type`` so it is never registered as a
device -- it is pure baseline + contract. Concrete devices live in
``platform/<backend>/device_ops.py`` and are discovered by ``device_type``.
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

#: The complete ops contract: every name a device provides natively or inherits
#: from the torch baseline. Single source of truth for the parity/contract test,
#: :meth:`DeviceOps._bind_native`, and the ``lmcache.c_ops`` shim.
OPS: frozenset[str] = frozenset(
    {
        "alloc_pinned_numa_ptr",
        "free_pinned_numa_ptr",
        "alloc_pinned_ptr",
        "free_pinned_ptr",
        "alloc_shm_pinned_ptr",
        "free_shm_pinned_ptr",
        "alloc_hugepage_pinned_ptr",
        "free_hugepage_pinned_ptr",
        "alloc_hugepage_pinned_numa_ptr",
        "free_hugepage_pinned_numa_ptr",
        "alloc_numa_ptr",
        "free_numa_ptr",
        "batched_memcpy",
        "lmcache_memcpy_async",
        "multi_layer_kv_transfer",
        "multi_layer_kv_transfer_unilateral",
        "multi_layer_block_kv_transfer",
        "single_layer_kv_transfer",
        "single_layer_kv_transfer_sgl",
        "load_and_reshape_flash",
        "reshape_and_cache_back_flash",
        "encode_fast_new",
        "decode_fast_new",
        "decode_fast_prefsum",
        "calculate_cdf",
        "rotary_embedding_k_fused",
        "get_gpu_pci_bus_id",
        "record_completion_on_stream",
        "drain_recorded_completions",
        "record_event_on_stream",
        "drain_recorded_events",
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
        "execute_object_group_transfer",
    }
)


class DeviceOps:
    """Strategy base: the torch baseline plus the shared op/type contract.

    Concrete subclasses set :attr:`device_type` and override only the ops they
    accelerate; everything else inherits the torch baseline below. The base
    itself has no ``device_type`` (empty), so it is never registered as a
    device.
    """

    device_type: ClassVar[str] = ""  # base is unregistered

    # Shared types are real class attributes (devices share identical enums).
    TransferDirection = TransferDirection
    EngineKVFormat = EngineKVFormat
    GPUKVFormat = EngineKVFormat  # back-compat alias
    PageBufferShapeDesc = PageBufferShapeDesc
    set_shape_desc_dtype = staticmethod(set_shape_desc_dtype)

    # Object-group transfer plan types (native-only; stubs on the baseline).
    StagingCopy = StagingCopy
    LaunchVar = LaunchVar
    BatchStep = BatchStep
    KernelGroupSpec = KernelGroupSpec

    def alloc_pinned_numa_ptr(self, *a, **k):
        return _torch_ops.alloc_pinned_numa_ptr(*a, **k)

    alloc_pinned_numa_ptr.__wrapped__ = _torch_ops.alloc_pinned_numa_ptr  # type: ignore[attr-defined]

    def free_pinned_numa_ptr(self, *a, **k):
        return _torch_ops.free_pinned_numa_ptr(*a, **k)

    free_pinned_numa_ptr.__wrapped__ = _torch_ops.free_pinned_numa_ptr  # type: ignore[attr-defined]

    def alloc_pinned_ptr(self, *a, **k):
        return _torch_ops.alloc_pinned_ptr(*a, **k)

    alloc_pinned_ptr.__wrapped__ = _torch_ops.alloc_pinned_ptr  # type: ignore[attr-defined]

    def free_pinned_ptr(self, *a, **k):
        return _torch_ops.free_pinned_ptr(*a, **k)

    free_pinned_ptr.__wrapped__ = _torch_ops.free_pinned_ptr  # type: ignore[attr-defined]

    def alloc_shm_pinned_ptr(self, *a, **k):
        return _torch_ops.alloc_shm_pinned_ptr(*a, **k)

    alloc_shm_pinned_ptr.__wrapped__ = _torch_ops.alloc_shm_pinned_ptr  # type: ignore[attr-defined]

    def free_shm_pinned_ptr(self, *a, **k):
        return _torch_ops.free_shm_pinned_ptr(*a, **k)

    free_shm_pinned_ptr.__wrapped__ = _torch_ops.free_shm_pinned_ptr  # type: ignore[attr-defined]

    def alloc_hugepage_pinned_ptr(self, *a, **k):
        return _torch_ops.alloc_hugepage_pinned_ptr(*a, **k)

    alloc_hugepage_pinned_ptr.__wrapped__ = _torch_ops.alloc_hugepage_pinned_ptr  # type: ignore[attr-defined]

    def free_hugepage_pinned_ptr(self, *a, **k):
        return _torch_ops.free_hugepage_pinned_ptr(*a, **k)

    free_hugepage_pinned_ptr.__wrapped__ = _torch_ops.free_hugepage_pinned_ptr  # type: ignore[attr-defined]

    def alloc_hugepage_pinned_numa_ptr(self, *a, **k):
        return _torch_ops.alloc_hugepage_pinned_numa_ptr(*a, **k)

    alloc_hugepage_pinned_numa_ptr.__wrapped__ = (  # type: ignore[attr-defined]
        _torch_ops.alloc_hugepage_pinned_numa_ptr
    )

    def free_hugepage_pinned_numa_ptr(self, *a, **k):
        return _torch_ops.free_hugepage_pinned_numa_ptr(*a, **k)

    free_hugepage_pinned_numa_ptr.__wrapped__ = _torch_ops.free_hugepage_pinned_numa_ptr  # type: ignore[attr-defined]

    def alloc_numa_ptr(self, *a, **k):
        return _torch_ops.alloc_numa_ptr(*a, **k)

    alloc_numa_ptr.__wrapped__ = _torch_ops.alloc_numa_ptr  # type: ignore[attr-defined]

    def free_numa_ptr(self, *a, **k):
        return _torch_ops.free_numa_ptr(*a, **k)

    free_numa_ptr.__wrapped__ = _torch_ops.free_numa_ptr  # type: ignore[attr-defined]

    def batched_memcpy(self, *a, **k):
        return _torch_ops.batched_memcpy(*a, **k)

    batched_memcpy.__wrapped__ = _torch_ops.batched_memcpy  # type: ignore[attr-defined]

    def lmcache_memcpy_async(self, *a, **k):
        return _torch_ops.lmcache_memcpy_async(*a, **k)

    lmcache_memcpy_async.__wrapped__ = _torch_ops.lmcache_memcpy_async  # type: ignore[attr-defined]

    def multi_layer_kv_transfer(self, *a, **k):
        return _torch_ops.multi_layer_kv_transfer(*a, **k)

    multi_layer_kv_transfer.__wrapped__ = _torch_ops.multi_layer_kv_transfer  # type: ignore[attr-defined]

    def multi_layer_kv_transfer_unilateral(self, *a, **k):
        return _torch_ops.multi_layer_kv_transfer_unilateral(*a, **k)

    multi_layer_kv_transfer_unilateral.__wrapped__ = (  # type: ignore[attr-defined]
        _torch_ops.multi_layer_kv_transfer_unilateral
    )

    def multi_layer_block_kv_transfer(self, *a, **k):
        return _torch_ops.multi_layer_block_kv_transfer(*a, **k)

    multi_layer_block_kv_transfer.__wrapped__ = _torch_ops.multi_layer_block_kv_transfer  # type: ignore[attr-defined]

    def single_layer_kv_transfer(self, *a, **k):
        return _torch_ops.single_layer_kv_transfer(*a, **k)

    single_layer_kv_transfer.__wrapped__ = _torch_ops.single_layer_kv_transfer  # type: ignore[attr-defined]

    def single_layer_kv_transfer_sgl(self, *a, **k):
        return _torch_ops.single_layer_kv_transfer_sgl(*a, **k)

    single_layer_kv_transfer_sgl.__wrapped__ = _torch_ops.single_layer_kv_transfer_sgl  # type: ignore[attr-defined]

    def load_and_reshape_flash(self, *a, **k):
        return _torch_ops.load_and_reshape_flash(*a, **k)

    load_and_reshape_flash.__wrapped__ = _torch_ops.load_and_reshape_flash  # type: ignore[attr-defined]

    def reshape_and_cache_back_flash(self, *a, **k):
        return _torch_ops.reshape_and_cache_back_flash(*a, **k)

    reshape_and_cache_back_flash.__wrapped__ = _torch_ops.reshape_and_cache_back_flash  # type: ignore[attr-defined]

    def encode_fast_new(self, *a, **k):
        return _torch_ops.encode_fast_new(*a, **k)

    encode_fast_new.__wrapped__ = _torch_ops.encode_fast_new  # type: ignore[attr-defined]

    def decode_fast_new(self, *a, **k):
        return _torch_ops.decode_fast_new(*a, **k)

    decode_fast_new.__wrapped__ = _torch_ops.decode_fast_new  # type: ignore[attr-defined]

    def decode_fast_prefsum(self, *a, **k):
        return _torch_ops.decode_fast_prefsum(*a, **k)

    decode_fast_prefsum.__wrapped__ = _torch_ops.decode_fast_prefsum  # type: ignore[attr-defined]

    def calculate_cdf(self, *a, **k):
        return _torch_ops.calculate_cdf(*a, **k)

    calculate_cdf.__wrapped__ = _torch_ops.calculate_cdf  # type: ignore[attr-defined]

    def rotary_embedding_k_fused(self, *a, **k):
        return _torch_ops.rotary_embedding_k_fused(*a, **k)

    rotary_embedding_k_fused.__wrapped__ = _torch_ops.rotary_embedding_k_fused  # type: ignore[attr-defined]

    def get_gpu_pci_bus_id(self, *a, **k):
        return _torch_ops.get_gpu_pci_bus_id(*a, **k)

    get_gpu_pci_bus_id.__wrapped__ = _torch_ops.get_gpu_pci_bus_id  # type: ignore[attr-defined]

    def record_completion_on_stream(self, *a, **k):
        return _torch_ops.record_completion_on_stream(*a, **k)

    record_completion_on_stream.__wrapped__ = _torch_ops.record_completion_on_stream  # type: ignore[attr-defined]

    def drain_recorded_completions(self, *a, **k):
        return _torch_ops.drain_recorded_completions(*a, **k)

    drain_recorded_completions.__wrapped__ = _torch_ops.drain_recorded_completions  # type: ignore[attr-defined]

    def record_event_on_stream(self, *a, **k):
        return _torch_ops.record_event_on_stream(*a, **k)

    record_event_on_stream.__wrapped__ = _torch_ops.record_event_on_stream  # type: ignore[attr-defined]

    def drain_recorded_events(self, *a, **k):
        return _torch_ops.drain_recorded_events(*a, **k)

    drain_recorded_events.__wrapped__ = _torch_ops.drain_recorded_events  # type: ignore[attr-defined]

    def is_cross_layer(self, *a, **k):
        return _torch_ops.is_cross_layer(*a, **k)

    is_cross_layer.__wrapped__ = _torch_ops.is_cross_layer  # type: ignore[attr-defined]

    def is_kv_list(self, *a, **k):
        return _torch_ops.is_kv_list(*a, **k)

    is_kv_list.__wrapped__ = _torch_ops.is_kv_list  # type: ignore[attr-defined]

    def is_layer_list(self, *a, **k):
        return _torch_ops.is_layer_list(*a, **k)

    is_layer_list.__wrapped__ = _torch_ops.is_layer_list  # type: ignore[attr-defined]

    def is_mla(self, *a, **k):
        return _torch_ops.is_mla(*a, **k)

    is_mla.__wrapped__ = _torch_ops.is_mla  # type: ignore[attr-defined]

    def execute_object_group_transfer(self, *a, **k):
        return _torch_ops.execute_object_group_transfer(*a, **k)

    execute_object_group_transfer.__wrapped__ = _torch_ops.execute_object_group_transfer  # type: ignore[attr-defined]

    def _bind_native(self, module: object) -> None:
        """Bind a compiled module's native ops over the torch baseline.

        Iterates :data:`OPS` and, for each name the module exports, sets the
        native callable as an instance attribute that shadows the base method
        with zero extra dispatch hops. Names absent from the module keep the
        torch baseline; native symbols absent from :data:`OPS` are ignored.
        """
        for name in OPS:
            fn = getattr(module, name, None)
            if fn is not None:
                setattr(self, name, fn)

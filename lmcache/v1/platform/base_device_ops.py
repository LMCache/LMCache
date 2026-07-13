# SPDX-License-Identifier: Apache-2.0
"""The unified per-device ops abstraction (the ``lmcache.c_ops`` surface).

:class:`DeviceOps` is a strategy base class whose **every op is an instance
method** with a real Python body delegating to
:mod:`lmcache.v1.platform._torch_impl`. Accelerator subclasses override
individual methods with native kernels via normal OO polymorphism.

Design goals (addressing review feedback on the previous static-method /
reflective-binding design):

* **Full method list in the base class**: readers and IDEs see the entire
  contract in one place, no reflection required.
* **Instance methods, not staticmethods**: subclasses can override single ops
  through standard Python MRO and can hold per-instance native handles bound
  in :meth:`DeviceOps.ensure_native` (e.g. attributes captured from a
  compiled ``.so`` module).
* **The ``lmcache.c_ops`` shim** forwards attribute access to a resolved
  ``DeviceOps`` singleton instance, so existing module-level call sites
  (``lmc_ops.multi_layer_kv_transfer(...)``) keep working unchanged.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar
import inspect

if TYPE_CHECKING:
    # Third Party
    import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import _torch_impl, ops_types
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


class DeviceOps:
    """Strategy base: per-device ops resolved via normal instance MRO.

    Every op is an ``instance`` method delegating to the torch baseline in
    :mod:`~lmcache.v1.platform._torch_impl`. Accelerator subclasses either:

    - Override individual methods (e.g. MUSA overrides
      :meth:`multi_layer_block_kv_transfer`).
    - Rebind all ops in :meth:`ensure_native` to a compiled ``.so`` module
      via :meth:`_bind_native`.

    The ``lmcache.c_ops`` shim forwards attribute access to a resolved
    singleton instance so module-level call sites keep working.
    """

    device_type: ClassVar[str] = ""  # base is unregistered

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

    def __init__(self) -> None:
        self._native_bound: bool = False

    # ── Lazy native binding ───────────────────────────────────────────

    def ensure_native(self) -> None:
        """Attempt to load and bind the compiled native extension.

        Subclasses override this to import their own ``lmcache.c_ops`` and
        call :meth:`_bind_native`. The base class is a no-op (pure torch).
        Guarded by ``_native_bound`` so it runs at most once per instance.
        """

    def _bind_native(self, module: object) -> None:
        """Rebind every op / native type on *self* to the *module*'s export.

        Op names are discovered by reflecting the class body (every public
        callable declared on :class:`DeviceOps` except :meth:`ensure_native`),
        so the class body is the single source of truth: add a method here
        and it is automatically eligible for native rebinding. If *module*
        exports a symbol with the same name, the instance attribute is
        overwritten with the module's callable (bound as a plain attribute,
        not a method -- native functions have no ``self`` argument). Native
        types (``EngineKVFormat``, ...) are rebound too for pybind enum
        identity.
        """
        bound_ops = 0
        bound_types = 0
        skip = {"ensure_native"}
        for name, member in vars(DeviceOps).items():
            if name.startswith("_") or name in skip:
                continue
            # Only rebind real methods declared in the class body; skip
            # class-level type aliases (``EngineKVFormat`` etc.) and
            # ``staticmethod`` wrappers -- those are handled below.
            if not inspect.isfunction(member):
                continue
            fn = getattr(module, name, None)
            if fn is not None:
                setattr(self, name, fn)
                bound_ops += 1
        for type_name in (
            "TransferDirection",
            "EngineKVFormat",
            "PageBufferShapeDesc",
            "StagingCopy",
            "LaunchVar",
            "BatchStep",
            "KernelGroupSpec",
        ):
            t = getattr(module, type_name, None)
            if t is not None:
                setattr(self, type_name, t)
                bound_types += 1
        ekf = getattr(module, "EngineKVFormat", None)
        if ekf is not None:
            self.GPUKVFormat = ekf  # back-compat alias
        logger.debug(
            "_bind_native: %s <- %s (%d ops, %d types)",
            type(self).__name__,
            getattr(module, "__name__", repr(module)),
            bound_ops,
            bound_types,
        )

    # ── Ops (all delegate to the torch baseline) ──────────────────────

    def alloc_pinned_numa_ptr(self, size, numa_id=0):
        return _torch_impl.alloc_pinned_numa_ptr(size, numa_id)

    def free_pinned_numa_ptr(self, ptr, size=None):
        return _torch_impl.free_pinned_numa_ptr(ptr, size)

    def alloc_pinned_ptr(self, size, device_id=0):
        return _torch_impl.alloc_pinned_ptr(size, device_id)

    def free_pinned_ptr(self, ptr):
        return _torch_impl.free_pinned_ptr(ptr)

    def batched_memcpy(self, src_ptrs, dst_ptrs, sizes):
        return _torch_impl.batched_memcpy(src_ptrs, dst_ptrs, sizes)

    def alloc_shm_pinned_ptr(self, size, shm_name=""):
        return _torch_impl.alloc_shm_pinned_ptr(size, shm_name)

    def free_shm_pinned_ptr(self, ptr, size=0, shm_name=""):
        return _torch_impl.free_shm_pinned_ptr(ptr, size, shm_name)

    def alloc_hugepage_pinned_ptr(self, size, device_id=0):
        return _torch_impl.alloc_hugepage_pinned_ptr(size, device_id)

    def free_hugepage_pinned_ptr(self, ptr, size=0):
        return _torch_impl.free_hugepage_pinned_ptr(ptr, size)

    def alloc_hugepage_pinned_numa_ptr(self, size, numa_id=0):
        return _torch_impl.alloc_hugepage_pinned_numa_ptr(size, numa_id)

    def free_hugepage_pinned_numa_ptr(self, ptr, size=0):
        return _torch_impl.free_hugepage_pinned_numa_ptr(ptr, size)

    def alloc_numa_ptr(self, size, numa_id=0):
        return _torch_impl.alloc_numa_ptr(size, numa_id)

    def free_numa_ptr(self, ptr, size=None):
        return _torch_impl.free_numa_ptr(ptr, size)

    def multi_layer_kv_transfer(self, *args, **kwargs):
        return _torch_impl.multi_layer_kv_transfer(*args, **kwargs)

    def multi_layer_kv_transfer_unilateral(self, *args, **kwargs):
        return _torch_impl.multi_layer_kv_transfer_unilateral(*args, **kwargs)

    def is_cross_layer(self, engine_kv_format):
        return _torch_impl.is_cross_layer(engine_kv_format)

    def is_kv_list(self, engine_kv_format):
        return _torch_impl.is_kv_list(engine_kv_format)

    def is_layer_list(self, engine_kv_format):
        return _torch_impl.is_layer_list(engine_kv_format)

    def is_mla(self, engine_kv_format):
        return _torch_impl.is_mla(engine_kv_format)

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: "torch.Tensor | list",
        lmcache_objects_ptrs: "list[int] | list[torch.Tensor]",
        block_ids: "torch.Tensor | list[int]",
        device: "torch.device | str",
        direction: ops_types.TransferDirection,
        shape_desc: ops_types.PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: ops_types.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        return _torch_impl.multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )

    def execute_object_group_transfer(self, *args, **kwargs):
        return _torch_impl.execute_object_group_transfer(*args, **kwargs)

    def single_layer_kv_transfer(self, *args, **kwargs):
        return _torch_impl.single_layer_kv_transfer(*args, **kwargs)

    def single_layer_kv_transfer_sgl(self, *args, **kwargs):
        return _torch_impl.single_layer_kv_transfer_sgl(*args, **kwargs)

    def load_and_reshape_flash(self, *args, **kwargs):
        return _torch_impl.load_and_reshape_flash(*args, **kwargs)

    def reshape_and_cache_back_flash(self, *args, **kwargs):
        return _torch_impl.reshape_and_cache_back_flash(*args, **kwargs)

    def lmcache_memcpy_async(self, *args, **kwargs):
        return _torch_impl.lmcache_memcpy_async(*args, **kwargs)

    def encode_fast_new(self, cdf, input_sym, output_buffer, output_lengths):
        return _torch_impl.encode_fast_new(
            cdf, input_sym, output_buffer, output_lengths
        )

    def decode_fast_new(self, cdf, bytestreams, lengths, output):
        return _torch_impl.decode_fast_new(cdf, bytestreams, lengths, output)

    def decode_fast_prefsum(self, cdf, bytestreams, lengths_prefsum, output):
        return _torch_impl.decode_fast_prefsum(
            cdf, bytestreams, lengths_prefsum, output
        )

    def calculate_cdf(self, input_tensor, num_bins):
        return _torch_impl.calculate_cdf(input_tensor, num_bins)

    def rotary_embedding_k_fused(self, *args, **kwargs):
        return _torch_impl.rotary_embedding_k_fused(*args, **kwargs)

    def get_gpu_pci_bus_id(self, device_id=0):
        return _torch_impl.get_gpu_pci_bus_id(device_id)

    def record_completion_on_stream(self, *args, **kwargs):
        return _torch_impl.record_completion_on_stream(*args, **kwargs)

    def drain_recorded_completions(self):
        return _torch_impl.drain_recorded_completions()

    def record_event_on_stream(self, *args, **kwargs):
        return _torch_impl.record_event_on_stream(*args, **kwargs)

    def drain_recorded_events(self):
        return _torch_impl.drain_recorded_events()

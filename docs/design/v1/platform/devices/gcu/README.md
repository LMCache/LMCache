# GCU (Enflame) platform support

This package wires Enflame GCU into LMCache's platform abstraction so that
``torch.gcu`` is detected as a first-class accelerator. All GCU-specific
behaviour lives under ``lmcache/v1/platform/devices/gcu/``; the only shared-file
change is the ``GcuIPCData`` wire descriptor registered in
``lmcache/v1/platform/base/ipc_wire.py``, which defaults to the existing
CUDA/torch behaviour when GCU is absent.

## Package layout

| Module | Responsibility |
|---|---|
| ``__init__.py`` | ``GcuDeviceSpec`` registration (detection + backend selection). |
| ``device_ops.py`` | ``GcuDeviceOps``: forwards the unified ``DeviceOps`` surface to the ``triton_kernel_gcu`` plugin's ``torch.ops.TRITON_GCU.lmcache`` operators. |
| ``cache_context.py`` | ``GcuCacheContext``: ``GPUCacheContext`` with CPU-resident pointer tables and a cupy-free observability stream. |
| ``event_ipc.py`` | ``GcuEventIPCBackend``: GCU interprocess events with a direct-probe capability check. |
| ``ipc_wrapper.py`` | ``GcuIPCWrapper``: KV-cache tensor sharing via ``UntypedStorage._share_gcu_`` / ``_new_shared_gcu``. |
| ``pin_memory.py`` | ``GcuPinMemoryBackend``: host pinning via ``topsHostRegister`` / ``topsHostUnregister``. |

## Device ops

``GcuDeviceOps`` keeps the ops on the torch baseline until ``triton_kernel_gcu``
is imported (soft-fail when the plugin is absent, matching CUDA/XPU). The three
signature adaptations to the Triton operators are:

1. ``paged_memory_device`` is reduced to the integer ``paged_device_index``.
2. ``multi_layer_block_kv_transfer`` expands ``PageBufferShapeDesc`` into the
   scalar layout fields plus a ``dtype_code``.
3. ``multi_layer_kv_transfer`` forwards ``block_stride_elems``.

``tensor_from_ptr`` uses the ``__cuda_array_interface__`` round-trip (GCU has
no ``_construct_storage_from_data_pointer``).

## Cache context

``GcuCacheContext`` is a self-contained ``BaseCacheContext`` implementation
(like the MUSA/NPU contexts) rather than a ``GPUCacheContext`` subclass, so it
touches no CUDA source. Its staging-buffer and pointer-table machinery mirrors
the CUDA context; the one piece that differs is the observability stream: a
cupy-free ``StubStream`` whose ``ptr`` is the real ``gcuStream_t``; GDS DMA on
the topsfile path is synchronous, so no stream-ordering callback is needed.

## IPC

``GcuIPCWrapper`` mirrors ``CudaIPCWrapper`` but rides
``UntypedStorage._share_gcu_`` / ``_new_shared_gcu``. Its wire descriptor is
``GcuIPCData`` (same 8-tuple handle shape as ``CudaIPCData``), registered in
``lmcache/v1/platform/base/ipc_wire.py``. Device identity prefers
``get_device_properties(idx).uuid`` and falls back to the ordinal with a warning
when torch_gcu exposes no physical identifier.

## Host memory pinning

``GcuPinMemoryBackend`` prefers ``torch.gcu.gcurt().topsHostRegister`` and falls
back to loading ``libtopsrt.so`` via ``ctypes``. The lazy allocator registers its
slab with the ``cudaHostRegisterPortable`` flag so multi-GCU D2H/H2D works
(GCU rejects non-portable registrations from a different device).

## Operator requirements

The ``triton_kernel_gcu`` plugin must register its operators under
``torch.ops.TRITON_GCU.lmcache``: the KV transfer family
(``multi_layer_kv_transfer``, ``single_layer_kv_transfer``,
``multi_layer_block_kv_transfer``), the CacheGen codec (``calculate_cdf``,
``encode_fast_new``, ``decode_fast_new``, ``decode_fast_prefsum``), rotary
(``rotary_embedding_k_fused``, ``rotary_embedding_k_fused_strided``),
``lmcache_memcpy_async``, ``batched_memcpy``, the ``alloc_*``/``free_*`` host ops,
``record_*``/``drain_*`` and ``get_gpu_pci_bus_id``.

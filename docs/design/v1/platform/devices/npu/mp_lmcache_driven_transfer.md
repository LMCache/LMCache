# NPU Platform: LMCache-Driven MP Transfer

`lmcache/v1/platform/devices/npu/` implements the Ascend NPU side of the MP-mode
`lmcache_driven` handle-transfer path. It mirrors the MUSA platform (the
precedent for a CuPy-less platform on this path); all kernels stay in the
external `lmcache_ascend` plugin and are layered on by `NpuDeviceOps` via
`DeviceOps.bind_native`.

## Components

| Module | Responsibility |
|---|---|
| `event_ipc.py` | `NpuEventIPCBackend` — torch_npu implements the CUDA-style interprocess event ABI (`interprocess=True`, `ipc_handle`, `from_ipc_handle`), so the shared `DefaultEventIPCBackend` adapter applies, with two NPU overrides: `export_event` serializes with `device` pinned as the thread's current device (CANN, unlike CUDA, derives the handle from the current device at export time — callers on multi-device threads cannot rely on their ambient device) and keeps the source event alive in a bounded cache (CANN invalidates a handle once its source event is destroyed). `check_event_support` fails closed on builds lacking the ABI. |
| `ipc_wrapper.py` | `NpuIPCWrapper` — plane-aggregating KV IPC wrapper bound on `NpuDeviceSpec.ipc_wrapper_cls` (moved upstream from the plugin). One wrapper per registered layer; `to_tensor()` yields a bare tensor or a plane tuple. See `ipc_wrapper.md`. |
| `cache_context.py` | `NpuCacheContext` (subclass of `BaseCacheContext`) — imports worker KV mappings from `NpuIPCWrapper` and owns `_TempNpuBuffer` staging and the transfer stream. `get_kernel_group_kv_pointers` returns per-layer plane **views** — the zero-copy IPC-imported tensors themselves, or tuples of plane tensors for `NL_X_NP_X_NB_BS_ONE_HS` — not a device-resident pointer table: per-plane widths are unrecoverable from the summed `shape_desc.hs`, and the torch fallback consumes per-layer structures directly. The Phase-2 native fast path must revisit this method if it needs a real pointer table. `_NpuHostCallbackStream` adapts the torch_npu stream to the `cupy_stream` contract (`.ptr` from `npu_stream`; `launch_host_func` degrades to synchronize-then-run). |
| `device_ops.py` | `NpuDeviceOps` binds `lmcache_ascend.c_ops` and keeps completion/event recording stream-ordered: `_synchronize_npu_stream_pointer` (via `acl.rt.synchronize_stream`) runs before the immediate-enqueue fallback, preserving the `finish_write` storage-ownership contract until the plugin ships a native `aclrtLaunchCallback` recorder. |

## Shared-code additions (B-lite)

Three touches outside `lmcache/v1/platform/devices/npu/` are sanctioned for this
path; all other NPU-specific code lives in the plugin or under `npu/`:

- `torch_ops._tensor_from_npu_ptr` (plus its `npu` dispatch arm in
  `_tensor_from_ptr`) and the `NL_X_NP_X_NB_BS_ONE_HS` branch of the
  `multi_layer_block_kv_transfer` fallback
  (`_transfer_per_layer_mla_tuple`) let the torch fallback reconstruct NPU
  tensors from raw pointers and transfer per-layer plane tuples.
- `kv_wrap.wrap_one_kv_cache` dispatches per-layer values (tensor **or**
  plane sequence) via `gpu_connector.utils.get_device`, and
  `wrap_kv_caches` wraps one value per layer. The plane-aggregating
  wrapper's `to_tensor()` yields a bare tensor or a plane tuple, which
  server-side format detection consumes directly.
- `_normalize_lmcache_objects` honors an explicit `device` for pointer-mode
  inputs, so reconstructed object chunk views alias device-resident staging
  buffers instead of defaulting to CPU.

## Staging layout

`_TempNpuBuffer` allocates one flat `uint8` buffer per
`max_batch_size` chunks with two offset maps — `(batch, kernel_group)` and
`(batch, object_group)`. Per-layer MLA-family formats
(`NL_X_NB_BS_HS`, `NL_X_NP_X_NB_BS_ONE_HS`) stage as rank-3
`[L, slots, W]`; other formats use the rank-4
`(kv_size, L, slots, W)` layout. Kernel-group buffers are contiguous inside
their object group (the staging memcpy contract).

## Enabling

The path is opt-in: set `LMCACHE_MP_TRANSFER_MODE=lmcache_driven`.
AUTO still routes `npu` to engine-driven. The worker-side wrapper
(`NpuIPCWrapper`) is bound upstream on `NpuDeviceSpec.ipc_wrapper_cls`;
no plugin-side registration patch is involved.

vLLM-Ascend's model runner hands per-layer plane tuples/lists directly in
its `kv_caches` dict (`_reshape_kv_cache_tensors`: `(K, V)` pairs, MLA/DSA
3-/4-tuples, arity-1 tuples, `_adjust_kv_layout` plane lists). Those values
reach `wrap_kv_caches` unchanged, one wrapper per layer preserves the
structure across the wire, and the server classifies each layer from its
own entry — no layout hint is required from the engine.

Spec: `docs/superpowers/specs/2026-09-02-ascend-mp-lmcache-driven-transfer-design.md`.

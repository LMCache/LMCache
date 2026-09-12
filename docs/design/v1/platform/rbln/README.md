# RBLN Device Backend

Design notes for `lmcache/v1/platform/rbln/` -- the device-registry entry for
Rebellions NPUs. The engine is
[vllm-rbln](https://github.com/RBLN-SW/vllm-rbln).

## Scope

| path | supported | how it gets there |
|---|---|---|
| multiprocess, engine-driven | yes | `gather_paged_kv_to_cpu` / `scatter_cpu_to_paged_kv`, dispatching to `RblnDeviceOps.multi_layer_block_kv_transfer` |
| multiprocess, LMCache-driven | **no** | refused up front |
| in-process | not yet | `CreateGPUConnector` still raises for `rbln`; a follow-up adds the connector |

Multiprocess is the mode that matters first, and it is self-contained: the MP
path builds no GPU connector at all, resolving layouts through
`normalize_kv_and_discover_format` and moving KV through `RblnDeviceOps`.

`torch.rbln` comes from
[torch-rbln](https://github.com/RBLN-SW/torch-rbln) through a torch backend
entry point, so it is visible on a bare `import torch` -- LMCache never imports
it explicitly. It provides device discovery, `set_device()` and `synchronize()`,
but no `Stream` / `Event` types. The LMCache-driven path publishes KV buffers
across processes by exporting a device IPC handle and ordering the handoff with
a cross-process event, which cannot be expressed without an event type.
`RblnDeviceSpec` therefore overrides `is_handle_transfer_available()` to `False`
and leaves `ipc_wrapper_cls` / `event_ipc_backend` at their `None` defaults, so
`mp_transfer_mode=lmcache_driven` fails at its documented validation point
instead of crashing later on an attribute lookup. `mp_transfer_mode=auto`
already routes every non-CUDA device to the engine-driven context, so the
default needs no special casing.

## The 6-D KV cache

This is what makes RBLN unusual. vllm-rbln allocates each layer as

```
[2, num_blocks, num_kv_heads, 1, block_size, head_size]
```

-- HND with an **extra singleton axis between heads and block tokens**, which
the RBLN attention backend requires. Every other supported engine hands
LMCache a 5-D (or 4-D / 3-D) per-layer tensor.

Axis 3 is always 1, so the tensor is byte- and stride-identical to a 5-D
`[2, NB, NH, BS, HS]` layout. It is nonetheless **registered as its own
`EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS` (15)**, so detection reports what
vLLM-RBLN actually allocated instead of reshaping a device's KV cache to fit
another format's rank.

The alternative -- squeezing the axis during discovery so it classifies as the
existing `NL_X_TWO_NB_NH_BS_HS` (6) -- was rejected. Reshaping inside detection
requires a device hook, which makes discovery depend on process-global device
state rather than being a pure function of `(kv_caches, layout_hints)`, and it
forces a second device-specific rule (the HND override below) for a format that
is HND by definition. Registering the format removes both.

Consequences of the format being first-class:

- **Detection is device-independent.** The 6-D branch in `detectors/vllm.py`
  keys off the shape signature alone (`ndim == 6`, `shape[0] == 2`,
  `shape[3] == 1`), so the same input classifies identically on any host, and
  the detector needs no `rbln` entry in its device table.

  ```python
  if (list_depth == 1 and tensor_ndim == 6
          and first_tensor.shape[0] == 2 and first_tensor.shape[3] == 1):
      return lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, kv_caches
  ```

- **The reported layout is never consulted.** vllm-rbln does not set vLLM's KV
  cache layout, so `get_kv_cache_layout()` returns the NHD default, which under
  the shared 5-D format would have classified the cache as
  `NL_X_TWO_NB_BS_NH_HS` -- the wrong axis order for every transfer. Format 15
  is HND by definition, so the hint plays no part. `detectors/vllm.py` still
  forces HND for `cpu` (vLLM's CPU attention backend misreports its layout),
  but that table no longer has an `rbln` entry.

- **The squeeze happens where bytes move.** `RblnDeviceOps.multi_layer_block_kv_transfer`
  accepts only format 15 and the MLA layout (below) and applies
  `squeeze_singleton_axis` at entry on the HND side, so `kv_ops.py` keeps
  indexing a 5-D tensor. `kv_layout.py` therefore exports the strict squeeze
  plus the `is_rbln_kv_layout` predicate -- no tolerant pass-through variant,
  since the detected format has already established what the caller holds.

- **No transfer kernel handles format 15.** RBLN has no compiled
  block-transfer extension in tree, and the CUDA / SYCL kernels never see an
  RBLN cache, so their `default:` arm rejecting the format is correct rather
  than a gap. `csrc` therefore carries only the enum value, its
  `is_layer_list` classification, and the two pybind registrations.

The multiprocess path reaches the layout through `compute_kv_layout` / gather /
scatter, all of which resolve it via `normalize_kv_and_discover_format` and
never touch a connector -- which is why the format must be recognised by
detection rather than by a connector.

## The MLA layout

vllm-rbln's MLA attention backend
(`vllm_rbln/v1/attention/backends/mla`) allocates each layer as

```
[num_blocks, block_size, head_size]
```

-- a single latent plane with no K/V split and no head axis. Unlike the 6-D
HND cache this is not an RBLN-only shape: the vLLM detector already classifies
it as the existing `EngineKVFormat.NL_X_NB_BS_HS`, so no new format is
registered and detection needs no RBLN knowledge. Chunks stay in the canonical
single-plane wire layout `[L, T, HS]`, so -- as with HND -- a chunk stored
from an RBLN MLA cache is byte-compatible with every other device.

What is RBLN-specific is the **DMA shape**, not the layout. The shared torch
MLA path issues one `index_select` / `index_copy_` per layer. On torch-rbln
each of those is a separate v2v submission (the index is read back to the
host to build the copy descriptors), the gather result / `.to(device)` window
is a fresh device allocation per chunk, and every one of those ops carries a
whole-layer CPU fallback behind it (`submit_or_fallback`) should the runtime
reject a copy. Measured on a real vLLM-RBLN DeepSeek-V3 KV cache the fallback
never fires and the shared path is correct, so the RBLN sequence exists for
cost, not correctness: `kv_ops.py` fans the `L * B` blocks of a chunk into
(or out of) a persistent per-thread *device* staging buffer with one batched
`_foreach_copy_` of whole contiguous blocks -- direct `memcpy_v2v`, no index
tensor, no fallback path -- and crosses the device boundary in one contiguous
DMA. With one block per chunk this is on par with the shared path (the extra
d2d hop cancels the saved submissions); with two or more blocks per chunk it
is 3.5-5x faster (61-layer DeepSeek-V3, 137-549 MiB chunks: 10-41 ms vs
37-213 ms per chunk).

How the two layouts relate inside the backend:

- **No transpose to hoist, so the staging moves to the device.** The HND
  sequence exists to move the head<->token transpose to the host, which is why
  its staging buffer is host memory. MLA has no head axis and its chunk is the
  blocks laid end to end, so the only cost left is the number of DMAs; its
  staging buffer is device memory (per thread, like the HND host buffer) so
  that the block-granular copies stay on the device and the chunk crosses the
  boundary once. The paged layers must be contiguous for those whole-block
  copies to be direct; `validate_mla_layers` pins that too.
- **Nothing to squeeze, so the rank is pinned instead.**
  `validate_mla_layers` in `kv_layout.py` mirrors `squeeze_singleton_axis`'s
  strictness -- the detected format has already established what the caller
  holds, so any non-3-D tensor is a layout drift and fails loudly at the
  transfer boundary -- but returns the tensors unchanged.
- **Per-format addressing, shared bookkeeping.**
  `RblnDeviceOps.multi_layer_block_kv_transfer` dispatches with
  `lmcache_native.is_mla(engine_kv_format)`; the chunk/block bookkeeping
  (blocks-per-chunk split, global prefix skip translated to a per-chunk
  offset, direction handling) is shared between the layouts. The `kv_ops`
  names carry the split: `gather_blocks_to_chunk_hnd` / `_mla` and
  `scatter_chunk_to_blocks_hnd` / `_mla`.

Both layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: `VLLM_RBLN_USE_DEVICE_TENSOR=1`). With the default compile-mode
allocation the per-layer tensors are `meta`, and any transfer -- this
backend's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".

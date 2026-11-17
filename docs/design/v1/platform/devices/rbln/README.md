# RBLN Device Backend

Design notes for `lmcache/v1/platform/devices/rbln/` -- the device-registry entry for
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
it explicitly. The LMCache-driven path publishes KV buffers across processes by
exporting a device IPC handle and ordering the handoff with a cross-process
event. LMCache has no RBLN tensor IPC wrapper or event IPC backend for that
handoff, so `RblnDeviceSpec` reports `is_lmcache_driven_available()` as `False`
and leaves `ipc_wrapper_cls` / `event_ipc_backend` at their `None` defaults, so
`mp_transfer_mode=lmcache_driven` fails at its documented validation point
instead of crashing later on an attribute lookup. `mp_transfer_mode=auto`
resolves to the device spec's declared default, which `RblnDeviceSpec`
derives from that `False` capability as engine-driven, so the default
needs no special casing.

`torch.rbln` provides `Stream` / `Event` types for ordering within a
process. Their presence does not supply the cross-process IPC support above.

## The 6-D KV cache

This is what makes RBLN unusual. Since
[vllm-rbln#1114](https://github.com/RBLN-SW/vllm-rbln/pull/1114), vllm-rbln's
default attention kernels (`rbln_custom_ops`) allocate each layer as

```
[num_blocks, 2, num_kv_heads, 1, block_size, head_size]
```

-- blocks-first HND with an **extra singleton axis between heads and block
tokens**, which the RBLN attention backend requires. Every other supported
engine hands LMCache a 5-D (or 4-D / 3-D) per-layer tensor.

The `rbln_triton_ops` kernels allocate the same layout **K/V-first**,
`[2, num_blocks, num_kv_heads, 1, block_size, head_size]`, registered as
`EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS` (15). Everything below applies to it
with the K/V axis leading; the differences are listed under
[The K/V-first layout](#the-kv-first-layout).

Axis 3 is always 1, so the tensor is byte- and stride-identical to a 5-D
`[NB, 2, NH, BS, HS]` layout. It is nonetheless **registered as its own
`EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS` (19)**, so detection reports what
vLLM-RBLN actually allocated instead of reshaping a device's KV cache to fit
another format's rank.

The alternative -- squeezing the axis during discovery so it classifies as the
existing `NL_X_NB_TWO_NH_BS_HS` (7) -- was rejected. Reshaping a device's KV
cache inside a detector puts device knowledge in engine discovery, and the
format is HND by definition, so it needs no layout hint either. Registering the
format keeps the detector to a shape check.

Consequences of the format being first-class:

- **Detection is device-independent.** The 6-D branch in `detectors/vllm.py`
  keys off the shape signature alone (`ndim == 6`, `shape[1] == 2`,
  `shape[3] == 1`), so the same input classifies identically on any host, and
  the detector needs no `rbln` entry in its device table.

  ```python
  if (list_depth == 1 and tensor_ndim == 6
          and first_tensor.shape[1] == 2 and first_tensor.shape[3] == 1):
      return lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS, kv_caches
  ```

- **The reported layout is never consulted.** vllm-rbln does not set vLLM's KV
  cache layout, so `get_kv_cache_layout()` returns the NHD default, which under
  the shared 5-D format would have classified the cache as
  `NL_X_NB_TWO_BS_NH_HS` -- the wrong axis order for every transfer. Format 19
  is HND by definition, so the hint plays no part. `detectors/vllm.py` still
  forces HND for `cpu` (vLLM's CPU attention backend misreports its layout),
  but that table has no `rbln` entry.

- **The squeeze happens where bytes move.** `RblnDeviceOps.multi_layer_block_kv_transfer`
  classifies the format by its `KVFormatSpec` layout facts, not its enum
  value: a per-layer blocks-first HND layout (`is_layer_list`, `is_hnd`, not
  `is_two_major`, not `is_fused_packed`) takes the HND sequence, a per-layer
  `[NB, BS, HS]` MLA layout (`is_layer_list`, `is_mla`, not `is_pbs_fused` /
  `is_kv_second_tuple` / `requires_blocked_scale_repacking`) takes the MLA one
  (below), and any other layout is refused with a `ValueError` naming the
  format and its shape. The HND side applies `squeeze_singleton_axis` at entry, so `hnd_ops.py` indexes a 5-D tensor.
  `kv_layout.py` therefore exports the strict squeezes plus the
  `is_rbln_kv_layout` predicate -- no tolerant pass-through variant.
  `NL_X_NB_TWO_NH_BS_HS` (7) shares format 19's layout facts (the singleton
  axis is the only difference), but vLLM-RBLN never allocates it, so a 5-D
  cache fails loudly in the squeeze rather than being accepted. The same holds
  for `NL_X_TWO_NB_NH_BS_HS` (6) against format 15.

  ```python
  spec = get_spec_class(engine_kv_format)
  is_hnd = (spec.is_layer_list and spec.is_hnd
            and not spec.is_two_major and not spec.is_fused_packed)
  ```

- **No transfer kernel handles format 19.** RBLN has no compiled
  block-transfer extension in tree, and the CUDA / SYCL kernels never see an
  RBLN cache, so their `default:` arm rejecting the format is correct rather
  than a gap. `csrc` therefore carries only the enum value, its
  `is_layer_list` / `is_hnd` classification, and the pybind registration
  (likewise for format 15).

The multiprocess path reaches the layout through `compute_kv_layout` / gather /
scatter, all of which resolve it via `normalize_kv_and_discover_format` and
never touch a connector -- which is why the format must be recognised by
detection rather than by a connector.

**Transfer.** HND restores use `hnd_ops.restore_hnd`. Blocks-first
layers are transposed as an addressing view to `[2, NB, NH, BS, HS]`;
K/V-first layers already have that form. Neither view moves bytes. Row `r`
in staging corresponds to `layers[r % L][r // L][block]`.

For each restored block:

```
host chunk -> token-major device staging -> compiled swap
           -> head-major device staging -> paged blocks
```

The first and last steps use `torch._foreach_copy_`. Each chiplet shard's
swap is `head.copy_(tokens.permute(0, 2, 1, 3))` on the device. Restores skip
the global block prefix, including a prefix ending inside a chunk. A
multi-block chunk's token window across layers is strided on the host, so
restores pair each contiguous `[BS, NH*HS]` layer slice with its staging row.
One-block chunks group contiguous spans per shard and K/V half. This avoids
an extra host transpose or temporary host tensors on restore.

HND stores keep `kv_ops.gather_blocks_to_chunk_hnd` and its K/V-first
variant. Contiguous paged blocks land in per-thread host staging, then a
host transpose writes canonical `[2, L, T, NH*HS]` chunks. Stores do not
submit or compile device swaps and may run on asynchronous commit workers.
They write every requested block; trailing partial chunks leave unused tokens
untouched. There is no host-staged restore path.

## The K/V-first layout

Format 15 (`[2, NB, NH, 1, BS, HS]`, `rbln_triton_ops`) uses
`squeeze_singleton_axis_kv_first`, then the same store and restore paths as
blocks-first HND. Each paged copy addresses one contiguous `[NH, BS, HS]`
K or V plane. Identical KV produces byte-identical canonical chunks from
both layouts.

**`num_blocks == 2` is ambiguous.** `[2, 2, NH, 1, BS, HS]` fits both layouts.
The detector checks the K/V-first branch first, so it keeps that reading; a
layout hint from vLLM-RBLN would be needed to tell them apart. A real KV cache
has far more than two blocks, so this only matters for synthetic inputs.

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

- **No head axis to swap.** MLA keeps its persistent per-thread device
  staging and chunk-sized host copies. The paged layers must be contiguous
  for its whole-block copies to be direct; `validate_mla_layers` pins that.
- **Nothing to squeeze, so the rank is pinned instead.**
  `validate_mla_layers` in `kv_layout.py` mirrors `squeeze_singleton_axis`'s
  strictness -- the detected format has already established what the caller
  holds, so any non-3-D tensor is a layout drift and fails loudly at the
  transfer boundary -- but returns the tensors unchanged.
- **Per-format addressing, shared bookkeeping.**
  `RblnDeviceOps.multi_layer_block_kv_transfer` dispatches on the format
  spec's layout facts (see above). HND restores run per block in `hnd_ops`;
  HND stores and MLA keep chunk-oriented operations in `kv_ops`.

All RBLN layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: `VLLM_RBLN_USE_DEVICE_TENSOR=1`). With the default compile-mode
allocation the per-layer tensors are `meta`, and any transfer -- this
backend's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".


## HND requirements and staging lifetime

`RblnDeviceOps` checks HND requirements before any copy, on every call:
bf16 KV/chunks, head size a multiple of 64, and a
`torch_rbln::copy_strided_view` kernel registered for `PrivateUse1`.
Unsupported dtype/head size raises `ValueError`; a missing compiled copy
raises `RuntimeError`. These checks apply to stores too, so the first store
refuses a cache that cannot later be restored. There is no native extension
or transfer-path configuration, and no host-staged restore fallback. Paged
layers and canonical host chunks must be contiguous.

Placement is optional. When `torch.rbln.set_device_alloc` and
`DevicePlacement` are available, `staging.py` divides the `2L` rows over
`min(num_chiplet, L)` shards, using
`get_device_properties(device).num_chiplet`. Each shard owns a head-major
`[rows, NH, BS, HS]` tensor and a token-major `[rows, BS, NH, HS]` tensor,
placed whole on its chiplet. Without either placement API, the restore set
uses one unplaced shard and no device-properties query is made. A warning
names the missing APIs once per process and states that staging is not
spread over chiplets. The unplaced pair occupies chiplet 0, so all restore
staging memory must fit there alongside that chiplet's model and KV cache.

One restore set is shared process-wide for the latest geometry. It is
physically reserved on the first HND transfer in either direction, so a first
store surfaces an allocation error before publishing chunks. Allocation does
not copy or compile. Errors propagate and a failed set is not cached.
Worker-side engine-driven KV registration has no DeviceSpec/DeviceOps hook
receiving the caches, so this is not a startup reservation; deployments must
leave room when sizing KV.

Total bf16 device staging is `2 * 2L * NH * BS * HS * 2 bytes`, independent
of commit-worker count. For Qwen3-1.7B (`L=28`, `NH=8`, `BS=1024`, `HS=128`),
that is 234,881,024 bytes (about 235 MB, 224 MiB), spread across chiplets when
placement is available or held on chiplet 0 otherwise. Stores additionally
retain their per-thread host landing buffers from `kv_ops`.

## HND submission and caller contract

The shared restore set's lock covers the whole restore call, including all
shard swaps and the final blocking paged-block copies. This prevents another
restore from submitting a swap or reusing staging while the call still uses
its bytes; no separate swap lock is needed. Stores use per-thread host buffers
and never enter the restore swap program. Only the loaded model's latest KV
geometry is retained; concurrent geometry changes are outside the contract.

Restores run serially on the model-driving thread's current stream. Device
swap programs compile only during a restore on that thread, so the first
cache hit pays the one-time compilation cost for its geometry. Subsequent
restores reuse the programs. The plain copies respect stream ordering and
finish before the restore lock is released. Callers still own ordering
relative to model execution and completion/publication through the transfer
context.

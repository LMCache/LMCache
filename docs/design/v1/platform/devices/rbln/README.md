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
  format and its shape. The HND side applies `squeeze_singleton_axis` at entry, so `kv_ops.py` indexes a 5-D tensor.
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

**Transfer.** Both HND directions use the device buffers in `kv_ops`. Blocks-first
layers are transposed as an addressing view to `[2, NB, NH, BS, HS]`;
K/V-first layers already have that form. Neither view moves bytes. Row `r`
in staging corresponds to `layers[r % L][r // L][block]`.

For each restored block:

```
host chunk -> token-major device staging -> compiled swap
           -> head-major device staging -> paged blocks
```

The first and last steps use `torch._foreach_copy_`. The swap is
`head.copy_(tokens.permute(0, 2, 1, 3))` over the whole staging on the
device, run as the program built when the transfer was prepared (see
[Preparing transfers](#preparing-transfers)). Restores skip
the global block prefix, including a prefix ending inside a chunk. A
one-block chunk is one contiguous host copy. A multi-block chunk's token
window across layers is strided on the host, so restores pair each
contiguous `[BS, NH*HS]` layer slice with its staging row. This avoids an
extra host transpose or temporary host tensors on restore.

Stores reverse the device data flow. They batch paged blocks into head-major
staging, swap it into token-major staging on the device, then copy directly
into the canonical host chunk: one contiguous copy for a one-block chunk,
one per row for a multi-block chunk's strided token window. Stores write
every requested block; a partial chunk leaves its unused tokens untouched.

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
cost, not correctness: per block, `kv_ops.py` gathers the `L` layer slices
into (or fans them out of) an `[L, BS, HS]` *device* staging buffer with one
batched `_foreach_copy_` of whole contiguous blocks -- direct `memcpy_v2v`,
no index tensor, no fallback path. With one block per chunk, which is how
RBLN runs LMCache, the block then crosses the device boundary in one
contiguous DMA. A multi-block chunk's token window is strided across layers
on the host, so each block moves as one contiguous DMA per layer. Staging
holds one block, rather than one chunk, the same way HND's does; that gives
up the advantage a chunk-sized buffer had for multi-block chunks (3.5-5x
over the shared path on a 61-layer DeepSeek-V3).

How the two layouts relate inside the backend:

- **No head axis to swap.** MLA stages one block of every layer and needs
  no program; only its staging is prepared. The paged layers must be
  contiguous for its whole-block copies to be direct; `validate_mla_layers`
  pins that.
- **Nothing to squeeze, so the rank is pinned instead.**
  `validate_mla_layers` in `kv_layout.py` mirrors `squeeze_singleton_axis`'s
  strictness -- the detected format has already established what the caller
  holds, so any non-3-D tensor is a layout drift and fails loudly at the
  transfer boundary -- but returns the tensors unchanged.
- **Per-format addressing, shared bookkeeping.**
  `RblnDeviceOps.multi_layer_block_kv_transfer` dispatches on the format
  spec's layout facts (see above) and hands `kv_ops` blocks-first views
  (`[NB, 2, NH, BS, HS]` or `[NB, BS, HS]`). `kv_ops` keeps one table of
  prepared geometries, keyed by layer count, dtype and one block's shape;
  each layout is one class (`HndTransfer`, `MlaTransfer`) that prepares its
  staging (and HND's programs) and moves blocks. A new layout adds a class,
  not a table.

All RBLN layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: `VLLM_RBLN_USE_DEVICE_TENSOR=1`). With the default compile-mode
allocation the per-layer tensors are `meta`, and any transfer -- this
backend's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".


## HND requirements and staging

HND needs torch-rbln 0.12.1a0 or later: its compiled strided copy
(`torch_rbln::copy_strided_view`), placement (`torch.rbln.set_device_alloc`,
`DevicePlacement`), `get_device_properties(device).num_chiplet` and
`torch.rbln.capture_programs()`. LMCache does not probe for them; an older
torch-rbln fails at warm-up. The KV must be bf16 with a head size a
multiple of 64: torch-rbln builds no swap program for other KV, so warm-up
fails before serving, and a transfer of KV that was never prepared fails
before copying. There is no native extension or transfer-path
configuration, and no host-staged HND fallback. Canonical host chunks must
be contiguous.

For each HND geometry and direction, `kv_ops` keeps one head-major
`[2L, NH, BS, HS]` and one token-major `[2L, BS, NH, HS]` staging tensor.
Each is split on its row axis across `n = min(num_chiplet, L)` chiplets,
one `DevicePlacement` per contiguous slice:

```python
torch.rbln.set_device_alloc(head, [
    torch.rbln.DevicePlacement(0, c, ((0, 2 * L * c // n, 2 * L * (c + 1) // n),))
    for c in range(n)
])
```

One swap covers every row, instead of one swap per chiplet.

The staging is allocated when the geometry is prepared and every transfer
reuses it. A split tensor is cheap to allocate, but the first swap and the
first host copy that use a newly allocated one are far slower than later
ones (on Qwen3-1.7B the first swap takes about 38 ms instead of 0.4 ms, and
the first host copy about 8 ms more), so allocating per transfer would pay
that on every transfer. Kept staging also exists before the
engine measures free device memory to size its KV cache, so it is counted.

For element size `E` bytes, HND staging is
`2 directions * 2 tensors * 2L * NH * BS * HS * E` bytes per prepared
geometry (448 MiB for Qwen3-1.7B with bf16 and 1024-token blocks), and MLA
staging `2 directions * L * BS * HS * E`. torch-rbln also keeps device
memory on chiplet 0 for the two swap programs. Measured for Qwen3-1.7B,
preparation adds about 0.9 GiB on chiplet 0 and 0.1 GiB on each other
chiplet. `torch.rbln.empty_cache()`, which vLLM-RBLN calls before it
measures free device memory, then releases the placed slices and leaves
about 0.7 GiB, all on chiplet 0, without slowing transfers. That is what
vLLM-RBLN's KV sizing counts: Qwen3-0.6B, whose KV has the same shape, gets
1142 blocks instead of 1164.

## Preparing transfers

`warm_up_kv_transfer` is the one entry point. The engine calls it once, on
its model-driving thread, after the KV caches it registers are final and
before the first request:

```python
from lmcache.v1.platform.devices.rbln import warm_up_kv_transfer

warm_up_kv_transfer(kv_caches)  # the dict passed to register_kv_caches
```

It groups the caches by shape and dtype, so each group has one layout, and
hands each group to `RblnDeviceOps.prepare_kv_transfer` as one KV cache
group:

- **HND** (`[NB, 2, NH, 1, BS, HS]` or `[2, NB, NH, 1, BS, HS]`): both
  directions' split staging, and each direction's swap run once on it inside
  `torch.rbln.capture_programs()`. The built program's `runtime.run` is
  kept and is all a transfer runs.
- **MLA** (`[NB, BS, HS]`, which also covers single-plane indexer and scale
  caches): one `[L, BS, HS]` staging buffer per direction; nothing to
  build.
- **Any other layout** raises the `ValueError` a transfer of it would.

Why the programs are kept instead of letting torch-rbln build them on first
use:

- **A build on a commit worker breaks the model thread.** Building the
  compiled copy runs `torch.export` decompositions, which set process-wide
  state (`torch.compiler._is_exporting_flag`, `fx_traceback.current_meta`).
  While a store builds on a worker, a `torch._dynamo.disable` frame on the
  model thread can fail with `KeyError: 'custom'`, and the build stalls the
  forward it overlaps.
- **torch-rbln's own lookup can miss later.** `torch._dynamo.reset()` or a
  different cache key makes the next `copy_` build again. A kept program
  runs without a lookup, so a transfer cannot build.

With as many heads as tokens per block (`NH == BS`) both directions' swaps
have one shape, and torch-rbln would serve the restore swap from the
store's program. The restore then runs the store's program, which is valid
because a program is not tied to the buffers it was built on, and shares
the store's lock (below). A swap of either shape that already ran in the
process before preparation has no build to capture, so preparation raises.

A transfer whose geometry was not prepared raises `RuntimeError` before any
copy. The engine-driven store logs the error and drops the store; the
restore reports a miss, so the engine recomputes rather than reading partly
written KV. Caches of one shape that LMCache splits into several groups (a
hybrid model's groups of one layout) are prepared as one group, so their
per-group transfers are not covered.

## Submission and caller contract

Preparation writes the prepared table, staging and programs before
transfers start, and nothing writes them afterwards, so transfers find them
without a lock. Each prepared geometry has one lock per direction, held for
a whole transfer. LMCache runs stores on several commit workers and
serializes only their commits to the server, not their gathers, so the
stores of one geometry take turns on its store staging. torch-rbln's copies
into host memory finish before they return, so the next store may refill
the staging as soon as the lock is free. Restores run serially on the
model-driving thread's current stream with their own staging, so a restore
does not wait for stores.

Taking turns also means no two threads run one swap program at once, which
rebel's runtime requires: it fails with `INIT_INTERNAL` otherwise. When both
directions share one program (`NH == BS`), they share one lock, and a
restore waits for an in-flight store.

Taking turns costs no throughput. On one NPU with Qwen3-1.7B and one block
per chunk, a store takes 6.7 ms on kept split staging against 8.1 ms on
per-chiplet staging allocated per transfer, and four threads storing back to
back finish one store every 8.4-8.5 ms against 9.7-10.0 ms when each
concurrent store allocates its own staging.

Callers own ordering relative to model execution and publish a transfer
only when the transfer context reports completion.

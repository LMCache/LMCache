# LMCache-Driven Transfer: Kernel vs Direct Copy Path

Module: `lmcache/v1/multiprocess/object_group_transfer.py` (eligibility and
planning) driven from `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`
(the STORE/RETRIEVE handlers), native side `csrc/cuda/mp_mem_kernels.cu`.

## Problem

STORE/RETRIEVE moves each chunk between a pinned host memory object
(`[kv, layer, tokens, nh*hs]`) and the engine's paged KV buffers. The
default path stages every chunk through a GPU temp buffer with
`cudaMemcpyAsync` and scatters/gathers it with the
`multi_layer_block_kv_transfer` kernel (grid `(kv, blocks, layers)`, one
thread block per paged block). When blocks are huge and few -- Kimi Linear
with vLLM's 1024-token MLA pages has 7 blocks of 1.1 MB per chunk -- the
kernel runs 7 thread blocks and moves data at ~24 GB/s, about half the
staging copy's line rate, so the kernel dominates the transfer.

## Two copy paths

| | `kernel` (fallback) | `direct` (preferred when usable) |
|---|---|---|
| Mechanism | `cudaMemcpyAsync` to temp buffer + block kernel | one `cudaMemcpyBatchAsync` per chunk, host <-> pages |
| GPU resources | temp buffer (4 slots), SMs | copy engine only |
| CPU cost | ~1 launch + 1 copy per chunk | ~0.6 us per (kv, layer, block) entry |
| Layouts | all `EngineKVFormat`s | token-major, contiguous block: formats 0, 1, 2, 3, 4, 5, 9, 11, 13, 18 |
| Requirements | any CUDA / HIP | CUDA runtime and driver >= 12.8, or ROCm/HIP >= 7.15 (opt-in) |

On ROCm the direct path is **opt-in**: it is built when HIP >= 7.15 but stays
disabled at runtime unless `LMCACHE_ROCM_ENABLE_BATCH_MEMCPY` is set to a
non-zero value. The compile-time `HIP_VERSION` guard cannot prove the op is
functional -- some HIP builds link `hipMemcpyBatchAsync` but return
`hipErrorNotSupported` at runtime -- so `batch_memcpy_supported()` requires the
explicit opt-in before selecting the copy-engine path. Even when opted in, the
probe additionally requires
`hipDeviceAttributeCanUseHostPointerForRegisteredMem == 1`: the batched copy
engine dereferences the `hipHostRegister`'d host operand directly, so on
configs where a registered host VA is not device-addressable (attribute `0`,
observed on some MI300/MI355 ROCm builds) the batched D2H/H2D copy faults with a
GPU memory access fault, and the connector must fall back to the staging kernel.
Some ROCm builds report the attribute as `0` even though the registered host VA
is device-addressable (e.g. MI355X, ROCm 10 / HIP 7.15). On such platforms,
after `tests/v1/multiprocess/test_direct_copy_transfer_gpu.py` and
`test_direct_copy_planner_gpu.py` pass, set `LMCACHE_ROCM_FORCE_BATCH_MEMCPY`
to a non-zero value (together with `LMCACHE_ROCM_ENABLE_BATCH_MEMCPY`) to skip
the attribute check.
CUDA behaviour is unchanged (gated purely on the 12.8 runtime/driver check).

Every eligible format is affine in (kv plane, layer, block):
`address = base(kv, layer) + kv * kv_stride + layer * layer_stride +
block_id * block_stride`. `resolve_block_addressing` in
`mp_mem_kernels.cu` fills this table once per kernel group with strides
mirroring the kernel's `calculate_engine_global_offset` (so the padded
`block_stride_elems` of pool-shared MLA rows is honoured the same way); the
per-entry loop then has no format branches. Eligibility depends on the
geometry as well as the format, so `direct_copy_format_supported` takes the
kernel group's shape desc. An HND block (heads before tokens) with a single
head, `[1, bs, hs]`, holds the same bytes as the NHD block `[bs, 1, hs]`, so
formats 6, 7, 8 and 12 are addressed as their NHD twins 1, 2, 0 and 13 when
`nh == 1` (DeepSeek-V4's latent cache, or any GQA model at one KV head per
rank). Multi-head HND, blocked-scale and per-layer (K, V)-tuple layouts are
rejected because their paged block is not one contiguous `[bs, nh*hs]` run.

Host ranges are split at `LazyMemoryAllocator.PIN_CHUNK_SIZE` boundaries of
the allocator's virtual offset, exactly as `lmcache_memcpy_async` does: each
64 MB pin chunk is a separate `cudaHostRegister` region and a copy spanning
two of them fails with `cudaErrorInvalidValue`.

## Flow (direct path)

```
handler (store/retrieve)
  |- downsample_and_stage_block_ids  -> device tensors (kernel path)
  |                                   + mutated host lists (direct path)
  `- transfer_kv_per_object_group(..., block_ids_host, copy_policy)
       |- _HAS_BATCH_MEMCPY_ASYNC     -> build + driver support (once, at import)
       |- direct_transfer_supported   -> per object group
       `- run_direct_transfer
            |- DirectCopyGroupSpec per kernel group: host layer pointers,
            |  shape desc, format, slots/chunk, byte offset in object,
            |  host block ids
            |- DirectCopyObject per chunk: host ptr, allocator offset,
            |  size, chunk index, per-group skip blocks
            `- device_ops.execute_direct_copy_transfer (one GIL release)
                 for each object: expand entries -> split at pin
                 boundaries -> cudaMemcpyBatchAsync(stream)
```

Window skipping (`num_objects_to_skip`) and `skip_first_n_tokens` follow
the same rules as the staged planner, evaluated per chunk (the direct path
has no batch of 4; the skip is `recalculate_blocks_to_skip` of the chunk's
own skipped tokens).

## Selection

There is no configuration knob: the direct path is taken whenever it can be.
`_HAS_BATCH_MEMCPY_ASYNC` resolves the build- and driver-level support once at
import (`execute_direct_copy_transfer` exported, `batch_memcpy_supported()`
true), and `direct_transfer_supported` then checks the request itself -- host
block ids present, no GDS-backed object, and every kernel group of the object
group on an eligible layout with a per-layer block of at least 32 KB.
Anything unmet falls back to `kernel`; an ineligible layout or a block under
the floor is logged once per layout and geometry, the rest at debug level.

Callers that must stay on the kernel path pass no `block_ids_host`, which is
how the staged path is exercised in tests and how `qstore` keeps its existing
behaviour.

## Cost model

Measured on H200 (CUDA 13.0, driver 580): a batch entry costs ~0.6 us of
CPU and the copy engine starts only after the call returns, so per-chunk
time is `max(entries * 0.6 us, bytes / 55 GB/s)`. Break-even is ~34 KB per
entry; below that the kernel path is faster (see the floor table below).
For Kimi Linear's 1.1 MB blocks the direct path cut a 128k-token
store from 26.6 ms to 19.7 ms per rank (x0.74) and retrieve to x0.91.

`direct_transfer_supported` also applies a size floor: a kernel group whose
paged block is below 32 KB per layer (`bs * nh * hs * element_size`) stays on
the kernel path (the ~34 KB break-even rounded down to the smallest size
that measured faster). Measured through `transfer_kv_per_object_group` on H200
(256-token chunks, 32 per call), store time direct / kernel:

| per-layer block | 1-D (one tensor per layer) | 2-D (layers packed in the block) |
|---|---|---|
| 4 KB | 4.0x | |
| 8 KB | 2.7x | |
| 16 KB | 1.5-1.7x | |
| 18.7 KB (DeepSeek-V4.1 sliding window) | | 1.5x |
| 32-128 KB | 0.87-0.94x | 1.12-1.36x (37-131 KB) |

Common GQA deployments fall under the floor once the KV heads are split
across ranks: Step-3.7-Flash at TP 4 (16 KB bf16, 8 KB fp8) measured 1.7x and
3.0x, MiMo-V2.5 at TP 4 (20 KB) 1.5x, gpt-oss-120b at TP 8 (8 KB) 2.3x.

Packed layouts above the floor still take the 2-D path; the copy engine
moves strided 2-D rows more slowly than 1-D entries of the same size
(42.5 vs 50.6 GB/s at 131 KB rows), so they can lose to the kernel path.

## Testing

- `tests/v1/multiprocess/test_direct_copy_transfer_gpu.py`: bit-exact
  parity with the kernel path for every eligible format, both directions,
  multi-chunk plans with a skipped block prefix, a two-group object, padded
  MLA stride, and pin-boundary splitting; single-head HND layouts (including
  DeepSeek-V4.1's packed sliding-window block); rejection of multi-head HND
  layouts and out-of-range ids.
- `tests/v1/multiprocess/test_direct_transfer_dispatch.py`: eligibility and
  dispatch (CPU only).

# LMCache-Driven Transfer: Kernel vs Direct Copy Path

Module: `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`,
native side `csrc/cuda/mp_mem_kernels.cu`.

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

| | `kernel` (default) | `direct` |
|---|---|---|
| Mechanism | `cudaMemcpyAsync` to temp buffer + block kernel | one `cudaMemcpyBatchAsync` per chunk, host <-> pages |
| GPU resources | temp buffer (4 slots), SMs | copy engine only |
| CPU cost | ~1 launch + 1 copy per chunk | ~0.6 us per (kv, layer, block) entry |
| Layouts | all `EngineKVFormat`s | token-major, contiguous block: formats 0, 1, 2, 3, 4, 5, 9, 11, 13 |
| Requirements | any CUDA / HIP | CUDA runtime and driver >= 12.8 |

Every eligible format is affine in (kv plane, layer, block):
`address = base(kv, layer) + kv * kv_stride + layer * layer_stride +
block_id * block_stride`. `resolve_block_addressing` in
`mp_mem_kernels.cu` fills this table once per kernel group with strides
mirroring the kernel's `calculate_engine_global_offset` (so the padded
`block_stride_elems` of pool-shared MLA rows is honoured the same way); the
per-entry loop then has no format branches. HND (heads before tokens),
blocked-scale and per-layer (K, V)-tuple layouts are rejected by
`direct_copy_format_supported` because their paged block is not one
contiguous `[bs, nh*hs]` run.

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
       |- select_transfer_copy_path   -> KERNEL | DIRECT  (per object group)
       `- _run_direct_copy_plan
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
has no batch of 4; the skip is `_recalculate_blocks_to_skip` of the chunk's
own skipped tokens).

## Configuration

`--transfer-copy-mode {kernel,direct,auto}` and
`--direct-copy-min-block-bytes N` (default 131072) populate
`MPServerConfig.transfer_copy_policy` (`TransferCopyPolicy`), carried on
`MPCacheServerContext`. `auto` chooses `direct` for an object group only
when every kernel group's tight block (`bs * nh * hs * element_size`) is at
least the threshold. Any unmet precondition (no native support, ineligible
layout, GDS objects, no host block ids) falls back to `kernel`; the first
occurrence of each reason is logged as a warning.

## Why the threshold

Measured on H200 (CUDA 13.0, driver 580): a batch entry costs ~0.6 us of
CPU and the copy engine starts only after the call returns, so per-chunk
time is `max(entries * 0.6 us, bytes / 55 GB/s)`. Break-even is ~34 KB per
entry; below that the kernel path is faster (2x at 16 KB blocks, 7-8x at
4 KB). For Kimi Linear's 1.1 MB blocks the direct path cut a 128k-token
store from 26.6 ms to 19.7 ms per rank (x0.74) and retrieve to x0.91. The
default 128 KB keeps small-page layouts on the kernel path with margin.

## Testing

- `tests/v1/multiprocess/test_direct_copy_transfer_gpu.py`: bit-exact
  parity with the kernel path for every eligible format, both directions,
  multi-chunk plans with a skipped block prefix, a two-group object, padded
  MLA stride, and pin-boundary splitting; rejection of HND layouts and
  out-of-range ids.
- `tests/v1/multiprocess/test_transfer_copy_policy.py`: config parsing and
  path selection (CPU only).

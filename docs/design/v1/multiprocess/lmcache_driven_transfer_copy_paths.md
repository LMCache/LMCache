# LMCache-Driven Transfer: Engines and Copy Paths

Package: `lmcache/v1/multiprocess/object_group_transfer/` (eligibility and
planning) driven from `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`
(the STORE/RETRIEVE handlers), native side `csrc/cuda/mp_mem_kernels.cu`.

| Module | Holds |
|---|---|
| `planning.py` | batching, block-id downsampling, skip arithmetic shared by every path |
| `direct_copy.py` | the `direct` path: eligibility and planner |
| `staged.py` | the `kernel` path: as one native plan, or one batch at a time (`run_staged_loop`) |
| `engines.py` | which transfer engine moves an object, the split of a group into same-engine runs, the `DMAEngine` protocol and `stage`, which calls the engine; `run_step`, which runs a retrieve step on or off the stream |
| `transfer_step.py` | `TransferStep` (one retrieve step), the `Staging` it lands in, `retrieve_transfer` / `RetrieveTransfer` (the device work of one retrieve), and the `OffStreamEngine` protocol |
| `gpu_dma.py` | `GpuDmaEngine`: host/GPU copies queued on the stream |
| `nvme_dma.py` | `NvmeDmaEngine`: GDS slab reads and writes queued on the stream |
| `nic_dma.py` | `NicDmaEngine`, a placeholder: its `stage` raises `NotImplementedError` |
| `dispatch.py` | `transfer_kv_per_object_group`: picks each run's copy path |
| `unfull_chunk.py` | the request's trailing unfull chunk |

Callers import the public names from the package itself.

## Transfer engines

The engine is chosen per object, from its L1 type, so one storage-backed
chunk does not move its whole object group off the fast path.

An engine does one thing: `stage(objects, staging regions, direction)` moves
bytes between objects and staging regions. It
does not know whether a region holds a full chunk or the unfull chunk, nor
which blocks it maps to; the caller owns the kernels. So the full-chunk loop
and `transfer_unfull_chunk` use the same engines:

```
retrieve:  stage(engine, objects -> regions)   then the caller's kernels
store:     the caller's kernels                then stage(engine, regions -> objects)
```

Every engine implements the `DMAEngine` protocol, which has that one method,
so the three keep the same interface. Each is a class in its own module:
`GpuDmaEngine` queues host/GPU copies and `NvmeDmaEngine` queues GDS slab
reads and writes, both on the stream. `engines.stage(engine, ...)` looks the
engine up and calls it. They are separate because their tuning and failure handling differ,
even where the code is short today. `nic_dma.py` is a placeholder: objects a
peer owns select `NicDmaEngine`, whose `stage` raises
`NotImplementedError` until the engine is added.

### Retrieve steps: on-stream and off-stream

A retrieve moves a group in steps: one kernel batch of full chunks, or the
unfull chunk. `run_staged_loop` and `transfer_unfull_chunk` describe each as
a `TransferStep` and give it to `engines.run_step(engine, context, step)`:

```python
TransferStep(
    memory_objs,   # the objects to stage, all of one engine
    block_ids,     # the block ids the kernels use
    targets,       # staging -> one region per object
    launch,        # (staging, block_ids) -> queues the kernels
)
```

`targets` and `launch` take the `Staging` the step lands in, so the same
step works for the context's own buffer and for a buffer an engine owns.
Which of the two depends on how the engine's bytes arrive:

| Engine kind | When its bytes are in the staging | `run_step` |
|---|---|---|
| On-stream (GPU DMA, NVMe DMA) | The copy is queued on the context's stream, which orders the kernels after it. | `stage(engine, objs, step.targets(context))`, then `step.launch(context, block_ids)`: staged and launched now, as before. |
| Off-stream (NIC DMA) | They arrive by a path the stream does not wait for (a NIC writing into GPU memory), so the kernels cannot be queued yet. | Hands the step, with its own copy of the block ids, to the retrieve's `RetrieveTransfer`. Nothing is staged or launched yet. |

A caller does not tell the two apart. It runs the transfers of one retrieve
inside `retrieve_transfer()` and asks to be called back once all of the
retrieve's device work is queued on the stream:

```python
with retrieve_transfer() as device_work:
    ...  # the retrieve's transfers
    device_work.when_queued(cache_context, succeeded, on_done)
```

| The retrieve has | `when_queued` |
|---|---|
| only on-stream steps | calls `on_done(succeeded)` now |
| off-stream steps | gives each engine its steps (`engine.submit(cache_context, steps, engine_done)`) and calls `on_done` when the last engine is done: True if all of them queued their kernels |
| an off-stream step without an engine | calls `on_done(False)` now |
| a failed transfer (`succeeded` False) | drops the off-stream steps and calls `on_done(False)` now |

An off-stream engine implements the `OffStreamEngine` protocol and is listed
in `engines.OFF_STREAM_ENGINES` under its `TransferEngine` kind:

```python
engine.submit(cache_context, steps, on_done)  # returns at once
# later, on the submitting thread: on_done(True) after the last step's
# kernels are queued, on_done(False) if the steps could not all be staged
engine.drain(cache_context)                   # before closing the context
```

It stages each step into slots of its own buffer (`step.targets(view)`),
and when the bytes landed calls `step.launch(view, step.block_ids)`. It
takes a completion callback and knows nothing about requests or responses.

The steps reach the `RetrieveTransfer` through a context variable and not as
a return value: a transfer runs as the cache context's
`transfer_object_group` callback, which returns nothing on every platform. A
transfer outside `retrieve_transfer()` gets `NotImplementedError` for an
off-stream step.

Stores are not steps: a store queues the kernels first and stages out
after them, on the stream, with an on-stream engine.

**The RETRIEVE handler** (`lmcache_driven_transfer.py`) has no branch for
either kind. Its transfers run inside `retrieve_transfer()`, and the code
that completes a retrieve (record the completion event, queue the read-lock
release, publish `MP_RETRIEVE_END`) is one function, `finish(succeeded)`,
which it passes to `when_queued`.

**Status.** `OFF_STREAM_ENGINES` is empty, so a retrieve of a peer-owned
object fails (`finish(False)`: no tokens reported, read locks released, the
handler returns False). It failed before too, with `NotImplementedError`.
Every `on_done` therefore runs before `when_queued` returns today; a handler
that answers after it returned is a follow-up.

| L1 object | Engine | `stage` | Full-chunk copy path |
|---|---|---|---|
| Host memory (DRAM, DevDAX) | GPU DMA | host/GPU copy on the stream | `direct` when usable, else `kernel` (native plan, or per batch without it) |
| Storage-backed (`GDSMemoryObject`) | NVMe DMA | file read or write on the stream | `kernel`, per batch |
| Owned by a peer (`BorrowedMemoryObj`) | NIC DMA | not implemented; raises `NotImplementedError` | none |

`split_into_runs` cuts the group's chunks into consecutive runs of one engine
and each copy path gets its run as a `[start, end)` chunk range over the full
object list, so block-id positions stay absolute. Runs are consecutive because
a kernel launch takes its batch's block ids as one contiguous slice. A missing
entry (a chunk outside the sliding window, or a failed allocation on store)
joins the next object's run.

```
objects:  host host | storage storage | host host
runs:     GPU DMA [0,2) -> NVMe DMA [2,4) -> GPU DMA [4,6)   (in order, one stream)
```

A group whose objects are all of one type is a single run, which is what
every single-L1 deployment sees: the result is the same as before the split.

### Dispatch workflow

`transfer_kv_per_object_group` is called once per object group. It splits
the chunks into runs and picks each run's copy path. The direct copy and the
native plan bypass `stage`, so only host-memory runs may take them.

```mermaid
flowchart TD
    H["STORE / RETRIEVE handler<br/>downsample_and_stage_block_ids"]
    D["transfer_kv_per_object_group<br/>one call per object group"]
    S["split_into_runs<br/>select_engine on each chunk's L1 object"]
    R{"next run<br/>engine, start, end"}
    G1{"host memory, batch memcpy available<br/>and direct_transfer_supported?"}
    G2{"host memory and native<br/>plan executor available?"}
    GD["run_direct_transfer<br/>one batched copy per chunk,<br/>host memory to or from the pages"]
    GP["run_object_group_transfer_plan<br/>staging copies and kernels<br/>in one native call"]
    L["run_staged_loop, per batch"]
    ST["stage: the run's engine<br/>GPU DMA: host copy<br/>NVMe DMA: file read or write<br/>NIC DMA: not implemented"]
    K["block transfer kernel<br/>staging slots to or from the pages"]
    E{"more runs?"}
    Q["work is queued on the context's stream"]
    U["transfer_unfull_chunk<br/>select_engine on its object"]
    UK["unfull-chunk kernels"]

    H --> D --> S --> R
    R --> G1
    G1 -->|yes| GD
    G1 -->|no| G2
    G2 -->|yes| GP
    G2 -->|no| L
    L --> ST
    ST --> K
    GD --> E
    GP --> E
    K --> E
    E -->|yes| R
    E -->|no| Q
    H --> U
    U --> ST
    U --> UK
```

- **Order.** Runs are issued in chunk order on the context's one stream, so
  the staging buffer is never used by two runs at once.
- **Host memory** is decided per run, not per group: a run of host-memory
  chunks next to a storage-backed run still takes the direct path.
- **NVMe DMA** always uses the per-batch loop: its staging step is a file
  read or write, which the native plan cannot issue.
- **Skipped chunks.** On retrieve the loop does not stage the chunks of a
  batch that `skip_first_n_tokens` covers entirely; the kernel skips them.
- **Store.** The loop queues the kernels, then stages out.
- **NIC DMA** is not implemented: a retrieve step of objects a peer owns is
  off-stream and has no engine yet, which fails the retrieve; a store
  reaches the placeholder `stage`, which raises `NotImplementedError`.
- **Errors.** A retrieve raises on a missing object; a store skips the batch
  that contains one.
- **Partial work.** When a later run raises, the earlier runs are already
  queued. The handler marks the retrieve failed, so the blocks are not used.
- **One thread per context.** The handlers' client affinity makes one thread
  drive a context's stream and staging slots; the engines rely on it.

### NIC DMA

A peer's object (`BorrowedMemoryObj`) has no local bytes. The planned engine
is off-stream (see "Retrieve steps"): it reads the objects of a step into
slots of a staging buffer it owns, over the transfer channel, and launches
the step's kernels once the reads landed:

```
peer DRAM --transfer channel--> engine's staging slot --kernel--> paged KV
```

That engine is not part of this package yet. `select_engine` already routes
peer-owned objects to `NIC_DMA`, which `run_step` treats as off-stream, so
the copy paths do not change when it arrives (RFC #6399).

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
formats 6, 7, 8 and 12 share the addressing of their NHD twins 1, 2, 0 and 13 when
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
  `- transfer_kv_per_object_group(..., block_ids_host)
       `- for each run of host-memory chunks
            |- batch_memcpy_available      -> build + driver support (once, at import)
            |- direct_transfer_supported   -> per run
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
`batch_memcpy_available` reports the build- and driver-level support, resolved
once at import (`execute_direct_copy_transfer` exported, `batch_memcpy_supported()`
true), and `direct_transfer_supported` then checks the run of host-memory
chunks itself -- host block ids present, no GDS-backed object, and every
kernel group of the object group on an eligible layout with a per-layer block
of at least 32 KB. Anything unmet falls back to `kernel`; an ineligible layout
or a block under the floor is logged once per layout and geometry, the rest at
debug level.

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
- `tests/v1/multiprocess/test_transfer_engines.py`: object type to engine,
  the split into runs (including missing entries), the routing of each
  run with its chunk range, the unimplemented NIC engine, and `run_step`:
  an on-stream step is staged then launched, an off-stream step goes to
  its engine with its own copy of the block ids, and `when_queued` reports
  each case (CPU only).
- `tests/v1/multiprocess/test_lmcache_driven_off_stream_steps.py`: the
  handler completes a retrieve once, with the right result, whether its
  steps ran on the stream, off it, or had no engine (CPU only).
- `tests/v1/multiprocess/test_staged_loop.py`: the per-batch loop's chunk
  range, block-id slices, skipped chunks, missing objects and the order of
  staging and kernels (CPU only, fakes).
- `tests/v1/multiprocess/test_direct_copy_planner_gpu.py`: a group split
  across engines gives the same bytes as one engine moving it whole, with
  the skipped prefix in a direct or a staged run and host runs on the
  direct copy or the native plan.

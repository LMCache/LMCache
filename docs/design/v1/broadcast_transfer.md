# First-rank broadcast retrieval

Module: `lmcache/v1/broadcast_transfer.py`, driven by
`LMCacheEngine._retrieve_broadcast` in `lmcache/v1/cache_engine.py`.
User-facing configuration is documented in
`docs/source/kv_cache/bounded_retrieval.rst`.

## Problem

With `save_only_first_rank` (MLA models), only the first tensor-parallel rank
stores KV chunks. At retrieval time every rank needs the same bytes in its own
KV pages. The previous implementation copied each chunk to the device on the
first rank, broadcast it, and kept every device copy alive until one final
`batched_to_gpu` call. Device memory therefore grew with the size of the cache
hit and could exhaust the device on any rank.

## Design

One fixed staging buffer per engine (`retrieve_buffer_size`, default 64 MiB),
allocated at engine creation and validated to hold at least one KV chunk. A
retrieval is one admission agreement followed by a sequence of *windows*: runs
of consecutive chunks that fit in the buffer. Each window costs one host object
broadcast (the header), one tensor broadcast (the packed bytes), one connector
write, and one fence. The buffer is reused for the next window and never grows.

The module is deliberately pure:

| Piece | Responsibility |
|---|---|
| `plan_windows(chunks, capacity)` | Split token-ordered chunks into runs whose packed layout fits `capacity`. A chunk larger than `capacity` forms its own window, which the engine rejects. |
| `Window.from_chunks(group)` | Lay the chunks out back to back on `CHUNK_ALIGNMENT` boundaries and record their token ranges and metadata as plain data. |
| `Window.views(buffer)` | Rebuild identical `TensorMemoryObj` views over any rank's buffer. Views own nothing. |
| `pack(group, views)` | Enqueue the host-to-device copies of the sources on the current stream. |

Everything that involves ranks, streams, ownership, or failure policy lives in
the engine, next to the single-rank retrieval it mirrors.

## Contracts

- `Window` is plain data and must stay pickle-safe: it crosses ranks through
  the integration's object broadcast.
- Views never take or release references. The source rank keeps owning its
  chunks until the engine releases them after the final fence. That release
  follows the single-rank rule from #5098: blocking gets drop their reference
  only, and lookup pins stay with the lookup owner until `lookup_unpin`; async
  prefetch results own their pins and are released completely.
- `pack` does not block the host. The caller fences the stream before the
  buffer is reused or the sources are released.
- Every chunk starts on a `CHUNK_ALIGNMENT` boundary so transfer kernels see the
  same alignment as a standalone device allocation and `Tensor.view(dtype)`
  stays valid for every dtype.
- The staging capacity is fixed for the life of the engine. Admission checks
  that every rank's capacity is at least the first rank's, so any window the
  first rank can send fits everywhere.
- The staging buffer lives on the device the serving engine made current for
  the worker, never on a device derived from `worker_id` or `local_worker_id`:
  under vLLM data parallelism every DP engine's TP world starts at rank 0, so
  rank ids name another engine's GPU on DP rank >= 1.

## Protocol

```
first rank                                    other ranks
----------                                    -----------
broadcast_object((capacity, n, req))  ----->  compare with own buffer/health/req
agree(ready)                          <---->  agree(ready)        [MIN, once]
acquire chunks (same code as single rank)
for each window:
    pack(group) into staging                  header = broadcast_object(None)
    broadcast_object(window)  ------------->  window.views(staging)
    broadcast(staging[:n])    ------------->  broadcast(staging[:n])
    batched_to_gpu(views); fence              batched_to_gpu(views); fence
    commit ret_mask[window]                   commit ret_mask[window]
broadcast_object(None)        ------------->  return
fence; release chunks
```

The first rank packs a window *before* announcing it, so any failure while
preparing a window happens while peers are still waiting for a header and can
be answered with the terminating `None`.

## Failure semantics

Two rules. **Readiness is agreed once, before any tensor collective**, because
whether a rank can take part in the next broadcast is not a per-rank outcome:
a rank without a buffer, or with a smaller one, would stall its peers. **After
admission every rank performs the same collective sequence regardless of local
outcome.** Local failures shrink that rank's committed prefix; they never add
or remove collectives.

| Failure | First rank | Other ranks |
|---|---|---|
| Buffer reservation failed at init, budgets differ, health monitor disabled a rank, or ranks serve different requests | Not admitted: all-false, no transfer | Same |
| Chunk missing in storage | Prefix truncated by `_process_tokens_internal`, as in single-rank retrieval | Same prefix (they only see the windows that were sent) |
| Storage raises while acquiring, or a chunk exceeds the buffer | Sends the terminator, then re-raises | All-false mask |
| `batched_to_gpu` raises on a rank | That rank fences the work it queued, stops writing, keeps receiving, returns the prefix it committed, and logs the error | Unaffected |
| Device or collective error | Propagates; the engine marks itself failed and refuses later retrievals at admission | Same |

The serving engine already handles per-rank partial loads: the vLLM adapter
reports the tokens missing from the returned mask and vLLM recomputes those
blocks. That is why per-window agreement is unnecessary; admission is not,
because it protects participation in the collectives themselves.

The "peers never wait" guarantee covers recoverable local failures only. A
device or collective failure propagates out of `retrieve`; vLLM's executor
reports the failed worker rather than exiting at the raise site, so the engine
marks itself failed (`mark_init_failed`) and every later retrieval on that rank
votes not-ready at admission. Peers of the request in flight may block until
their communicator times out. Whether the worker group is torn down is the
serving engine's decision, not this module's.

## Streams and fences

Both retrieval paths release their sources only after
`GPUConnectorInterface.synchronize_load()`, including when `batched_to_gpu`
raised: a connector may have queued reads before failing, and a released L1
block can be reused by a concurrent store. Connectors without a `load_stream`
pay a device-wide synchronization for this; giving such a connector its own
load stream is the way to make the fence cheaper.

All device work for a window runs on `broadcast_stream`, which is the
connector's `load_stream` when it has one. In that case stream order alone
guarantees that the next window's `pack` copies wait for the previous window's
page writes. Connectors without a `load_stream` get an explicit
`broadcast_stream.synchronize()` before each write. Every write, successful or
not, is followed by `GPUConnectorInterface.synchronize_load()` before the mask
is committed and the buffer reused: a write that raises may already have
queued reads of the staging buffer, and a connector that writes from its own
streams would otherwise read bytes the next window has overwritten. The final
`finally` fences `broadcast_stream` once more before releasing the sources,
because `pack` copies may still be in flight when an exception unwinds the
loop.

## Memory accounting with vLLM

vLLM creates the worker's KV connector after `determine_available_memory` and
before `initialize_kv_cache`. The buffer is therefore invisible to vLLM's
sizing unless vLLM asks the connector for its device memory requirement first
(`get_required_device_memory`, provided by `lmcache.integration.vllm.utils`
and served by a vLLM with the matching hook). Without the hook and without an
explicit `kv_cache_memory_bytes`, the buffer comes out of the utilization
headroom; the worker logs a warning rather than refusing to start, so the fix
does not force a vLLM upgrade. The warning is explicit that no whole-device
budget guarantee exists in that mode. Note the trade-off against the previous
implementation: it allocated per hit, so it had no resident cost with small or
no hits but an unbounded worst case; the fixed buffer bounds the worst case and
adds `retrieve_buffer_size` of resident memory.

## Deliberately not done

- **Cross-rank agreement per window.** Costs two host all-reduces per window,
  only to turn a per-rank partial load into an all-rank miss. vLLM's
  load-failure recovery already covers partial loads.
- **Slicing chunks along the token dimension.** Needed only when the budget is
  smaller than one chunk. Rejecting such budgets at init is simpler and keeps
  chunks as the unit of transfer everywhere else in the engine.
- **Growing the buffer at runtime.** It would turn the configured budget into a
  floor rather than a ceiling, and a rank that fails to grow after receiving a
  header cannot join the tensor broadcast its peers are already in.
- **A process-wide buffer registry.** vLLM creates one engine per worker
  process. Extra engines on one device pay one more buffer each.
- **Retaining engines after a failed device fence.** A failed device fence is
  not recoverable in-process; the worker restarts.
- **Streaming the storage reads themselves.** The device buffer is what was
  unbounded. Chunk acquisition keeps the existing, well-tested single-rank code
  path so both retrieval modes share prefix semantics and ownership rules.

## Extending

- A connector that writes from several streams should override
  `synchronize_load()` to fence all of them.
- A new integration needs the two broadcast callbacks and, for more than one
  rank, `all_ranks_agree_fn`; without the latter the engine gathers votes with
  one object broadcast per rank. It must call `retrieve` in the same order on
  every rank of the group, which the collectives require anyway.

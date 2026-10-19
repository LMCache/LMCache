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

One staging buffer per engine, sized from `retrieve_window_bytes` (default
64 MiB) and the KV layout: as many whole chunks as fit the target, never fewer
than one, captured as a plain `StagingPlan`. It is reserved at engine
creation from the layout the serving engine derives from its model
configuration. The layout chunks really have can be registered later (vLLM's
V3 connector builds its KV layer groups at the first transfer, after
`post_init`, and only on ranks that transfer), so at every admission the first
rank re-plans from the layout its metadata describes then, reserves that plan,
and announces it; every other rank reserves the announced plan, because the
chunks it will receive are the first rank's. A plan that differs from the
buffer replaces it before the vote, with a warning that memory planning saw
the earlier figure; a matching plan, the common case, costs one integer
comparison. The target is a byte
figure because that is how memory is planned, but a chunk is the unit the
exchange moves, so the floor is one chunk and a smaller target is raised to it
with a warning. Rounding down keeps the target a ceiling in every realizable
case, and a byte target gives every model a similar amortization of per-window
overhead. The effective size is logged at start-up and reported to vLLM's
memory planning through the same formula. A
retrieval is one admission agreement followed by a sequence of *windows*: runs
of consecutive chunks that fit in the buffer. Each window costs one host object
broadcast (the header), one tensor broadcast (the packed bytes), one connector
write, and one fence. The buffer is reused for the next window and never grows.

A single rank has no peers, so none of this applies: it takes the ordinary
retrieval path, writing from the acquired chunks directly, and reserves no
buffer. Staging there would only add a device copy of bytes that cross PCIe
either way. This also keeps single-rank integrations that supply placeholder
collectives (SGLang, TensorRT-LLM) working.

The module is deliberately pure:

| Piece | Responsibility |
|---|---|
| `plan_staging(metadata, chunk_size, target)` | Size the window for the layout `metadata` describes: as many whole chunks as fit `target`, never fewer than one, as plain `StagingPlan` data that the first rank announces at admission. |
| `plan_windows(chunks, capacity)` | Split token-ordered chunks into runs whose packed layout fits `capacity`. Rejects a chunk it cannot send (larger than `capacity`, or without tensor storage) before returning any window. |
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
- The staging buffer only changes at admission, before the vote, and every
  rank then holds the first rank's announced plan, so any window the first
  rank can send fits everywhere. `StagingPlan` must stay pickle-safe: it
  crosses ranks through the integration's object broadcast.
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

Everything that can fail for local reasons on the first rank (storage reads,
planning) happens before its first header, while peers are still waiting for
one, so it can be answered with the terminating `None`. The first rank packs a
window before announcing it, so a peer joining the tensor broadcast finds the
payload already queued.

## Failure semantics

Two rules. **Readiness is agreed once, before any tensor collective**, because
whether a rank can take part in the next broadcast is not a per-rank outcome:
a rank without a buffer, or with a smaller one, would stall its peers. **After
admission every rank performs the same collective sequence regardless of local
outcome.** Local failures shrink that rank's committed prefix; they never add
or remove collectives.

| Failure | First rank | Other ranks |
|---|---|---|
| Buffer reservation failed at init, window sizes differ, health monitor disabled a rank, or ranks serve different requests | Not admitted: all-false, no transfer | Same |
| Chunk missing in storage | Prefix truncated by `_process_tokens_internal`, as in single-rank retrieval | Same prefix (they only see the windows that were sent) |
| Storage raises while acquiring, or a chunk cannot be sent (exceeds the buffer, no tensor storage) | Sends the terminator, then re-raises; the engine stays usable | All-false mask |
| `batched_to_gpu` raises on a rank | That rank fences the work it queued, stops writing, keeps receiving, returns the prefix it committed, and logs the error | Unaffected |
| Device or collective error, including a failed `pack` or fence | Propagates; the engine marks itself failed and refuses later retrievals at admission | Same |

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
budget guarantee exists in that mode. The hook can only measure the layout
derived from the model configuration, since vLLM asks before it builds its KV
caches; a model whose registered layout plans a different window is logged at
the first retrieval after registration, and that difference is outside the
plan. Note the trade-off against the previous
implementation: it allocated per hit, so it had no resident cost with small or
no hits but an unbounded worst case; the fixed buffer bounds the worst case and
adds one staging window of resident memory.

## Deliberately not done

- **Cross-rank agreement per window.** Costs two host all-reduces per window,
  only to turn a per-rank partial load into an all-rank miss. vLLM's
  load-failure recovery already covers partial loads.
- **Slicing chunks along the token dimension.** It only matters when the
  staging window is smaller than one chunk. The window is floored at one chunk
  instead, so that state cannot occur, and chunks stay the unit of transfer
  everywhere else in the engine.
- **Resizing the buffer during an exchange.** A rank that fails to grow after
  receiving a header cannot join the tensor broadcast its peers are already
  in. The admission-time reservation is not that: it precedes the vote, every
  rank reserves the one announced plan, and a rank it fails on votes not
  ready.
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
- A new integration with more than one rank needs the two broadcast callbacks
  and `all_ranks_agree_fn` (the engine refuses to start without it). The
  object-broadcast callback cannot stand in: it addresses the integration's
  group by that group's own rank space, not by `worker_id`. The integration
  must call `retrieve` in the same order on every rank of the group, which
  the collectives require anyway. A single rank calls none of the callbacks.

# Chunk-event stores

## Summary

`STORE_WITH_CHUNK_EVENTS` is a store that additionally reports one device
completion event per token chunk, so the caller can release each chunk's source
buffers before the store as a whole finishes.

The ordinary `STORE` path is unchanged. This is an additive request type; a
caller that does not need early release keeps using `store()`.

See [sparse_null_policy.md](sparse_null_policy.md) for the per-group null block
policy that this builds on for its staging.

## Motivation

A connector that offloads KV asynchronously has to keep the source GPU blocks
pinned until it knows the device-to-host copy has landed. With only a terminal
completion signal, that means holding the entire token range until the last
chunk copies — the first chunk's blocks stay pinned long after their data is
safely in host memory.

## Contract

```
STORE_WITH_CHUNK_EVENTS
  payload : (key, instance_id, block_ids, producer_event_handle)
  response: (terminal_event_handle,
             [(chunk_event_handle, start, end), ...],
             store_succeeded)
```

`start`/`end` are absolute token offsets; `end` is clamped to `key.end`. Ranges
are in ascending token order. The list is empty when the store was rejected
before any device work was submitted.

Client side, `ChunkEventDeviceMessagingFuture.take_completed_ranges()` drains
the ranges whose event has completed. It is non-blocking (`query_event`, never
`synchronize_event`), returns each range at most once, and returns `()` while
the server response is still in flight.

**The base future contract is unchanged.** `query()`, `wait()`, and `result()`
still pend on terminal completion only. Chunk events are a strictly earlier,
advisory signal — a caller must still treat terminal completion as the only
guaranteed one, since a transport without chunk-event support falls back to an
ordinary future that has no `take_completed_ranges`.

## Why not one RPC per chunk

The caller's actual goal is early *source release*, not granular storing.
Splitting the RPC would give the same release granularity but would:

- multiply the per-store fixed cost (registration lookup, index-buffer H2D,
  allocator reservation, event export, round trip) by the chunk count;
- interleave index staging with the copies it describes, breaking one
  contiguous stream segment into N stop-start segments;
- fragment atomicity. A store is one reservation and one commit. Split across N
  RPCs, a mid-sequence failure leaves a committed dense prefix with no
  corresponding sparse-group object, which no retrieve can ever use.

Chunk events keep the transfer as one logical operation and report progress
within it.

## Cost

`PER_CHUNK_EVENTS` issues `num_chunks` native transfer calls instead of one
batched call, plus one event record/export per chunk. Callers that do not need
early release must use plain `store`. The two entry points are separate methods,
and the internal selector is `StoreCompletionDetail` rather than a boolean, so
the cost is visible at the call site.

## Geometry sharing

The per-chunk slice stride into the staged buffer must equal what
`downsample_and_stage_block_ids` wrote. Both derive it from
`kept_blocks_per_chunk()`; computing it independently in either place lets the
slice silently drift from the staged layout.

## Compatibility

| Direction | Behavior |
|---|---|
| Old client → new server | Plain `STORE` unchanged. |
| New client → new server, no chunk events | `store()` unchanged; no per-chunk work. |
| New client → **old server** | **Not supported.** See below. |

`RequestType.STORE_WITH_CHUNK_EVENTS` is appended to the enum for wire
compatibility, but an old server has no handler for it. The MQ server decodes
the request type before dispatch and has no negotiated fallback: an unknown
type either fails to decode in the receive loop or falls through to a branch
that logs and sends no response, so the client's future never resolves. A
client must therefore not send this request type to a server that does not
implement it — deployments must upgrade the MP server together with the
connector.

## Testing

- `tests/v1/multiprocess/test_futures.py` — drain semantics, at-most-once,
  terminal contract unchanged.
- `tests/v1/multiprocess/test_lmcache_driven_transfer_skip.py` — chunk-by-chunk
  transfer enqueue and reported ranges.
- `tests/v1/multiprocess/test_protocols.py` — request/response shape.
- `tests/v1/test_atom_mp_adapter.py` — adapter forwarding and event retention.

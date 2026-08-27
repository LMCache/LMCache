# Cache-event emission (MP server → key directory)

Module: `lmcache/v1/mp_coordinator/cache_events.py`
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
Consumer: `lmcache/v1/mp_coordinator/views/key_directory.py` (see
[key_directory.md](key_directory.md))

This is the emission half of the key directory (M1 of the control-plane
RFC, [issue #4226](https://github.com/LMCache/LMCache/issues/4226)): MP
servers turn storage-listener callbacks into `CacheEventBatch` streams
and deliver them to the coordinator's directory.

## The transport seam

Production deployments may replace direct HTTP push with Kafka or
another message queue. The design isolates that choice behind one
interface so nothing else changes:

```
storage layer ──► EventBus ──► CacheEventSubscriber ──► CacheEventSink ──► directory
 (publishes)      (drain      (event → vocabulary,        (transport)
                  thread)      seq, batching)
```

- **`CacheEventSink`** — `publish(batches)` with **at-least-once**
  delivery, preserving list order within and across calls. That is the
  entire transport contract, and it is deliberately weak: the directory
  already absorbs everything a real transport does wrong. Redelivery is
  deduplicated by the per-instance `seq` cursor, loss surfaces as a
  `seq` gap that marks the instance's slice stale until the stream is
  replayed, and restarts are fenced by `incarnation`. A sink never needs exactly-once or global ordering.
- **`HttpCacheEventSink`** — the first sink: one
  `POST /events` per flush, batches in list order. Failures
  raise `CacheEventPublishError`; the caller decides retry vs drop
  (both are safe, see above).
- A future **Kafka sink** produces to a topic with the message key set
  to `instance_id`, so one partition carries one instance's stream —
  partition FIFO is exactly the per-instance FIFO the directory needs.
  The coordinator side gains a consumer that feeds
  the coordinator's `EventGate`; the subscriber and producers are
  untouched.

## Batching and sequencing (inside the subscriber)

One `CacheEventSubscriber` per MP-server process owns the buffer, the
`seq` counter, and the sink:

- **Order-preserving batching.** The buffer is a list of *pending
  batches*: consecutive records with the same `(event_type, tier,
  backend, shared)` identity append to the last pending batch; an identity
  change starts a new one (never merged backwards). Flushing emits one
  `CacheEventBatch` per pending batch, so the batch sequence preserves
  the total order of recorded events — a store followed by a delete of
  the same key can never be reordered into "delete, then store".
  Alternating identities therefore produce multiple pending batches of
  the same identity; that is intentional (extra batch headers, never
  reordering).
- **`seq` is consumed even when publish fails.** A failed flush drops
  the drained list (bounding memory while the coordinator is down) but
  keeps the `seq` numbers it assigned. The directory sees a gap and
  sets `gap_detected` for the instance — the honest signal that events
  were lost and the slice needs an event-stream replay to reconcile.
  Reusing the seqs instead would hide partial-delivery ambiguity (an
  HTTP timeout after the coordinator applied the batch).
- **`incarnation` = server start time** (`int(time.time())` at
  lifespan startup). A restarted server's first batch fences out the
  **L1** placements its previous incarnation reported, matching the
  fact that its memory restarted empty; L2 placements survive because
  the bytes persist on disk (see `key_directory.md`).

## Event flow (the observability bus)

The storage layer already publishes key-level events to the
observability `EventBus` (`mp_observability/event_bus.py`);
cache-event emission rides the same bus instead of adding parallel
listener plumbing or a dedicated flush task:

- **Producers.** `L1Manager` publishes `l1.write.finished`,
  `l1.write_finished_and_read_reserved`, `l1.keys.evicted` (all delete
  paths), and the new `l1.keys.accessed` (`touch_keys` — the MP request
  end's unified touch of a request's retrieved and stored keys; the
  subscriber deliberately does **not** consume `l1.read.finished`,
  which would duplicate those accesses). The placement-bearing events
  (stores and evictions) carry `meta: list[L1ObjectMeta]` — each
  object's `size_bytes` (`MemoryObj.get_size()`) and its
  `L1BackendType` medium from `L1ManagerProtocol.get_backend_type()` (the
  Device-DAX tier resolves it per object via
  `DevDaxMemoryAllocator.is_devdax_obj`, i.e. `MemoryObj.parent()`) —
  so a hybrid DRAM+DAX L1 reports exactly where each object landed,
  and deletes target the same placement identity `(instance, tier,
  backend)` their store reported. The L2 base adapter's
  listener-notify funnel publishes the new `l2.keys.stored`
  (`keys`+`sizes`+`backend`), `l2.keys.accessed`, and `l2.keys.deleted`
  events; the backend name is the registered adapter type and the
  optional `shared` flag comes from the adapter config, both stamped by
  the storage manager via `set_backend_identity` at build time — so
  **runtime-added adapters emit automatically**. Batches carry the
  `shared` flag; for shared batches the backend type name identifies
  the pool fleet-wide (one pool per backend type), so the directory
  deduplicates shared-storage placements across emitters (see
  `key_directory.md` — Shared pools).
  The LMCache-driven store path additionally publishes
  `mp.tokens` (parallel `chunk_hashes` + `token_chunks` +
  `token_offsets`) at
  store submission — ordered ahead of the store's write-finished
  events, built only when the event has a subscriber, so the cost is
  zero with event reporting off (and no hashing anywhere: the directory
  indexes tokens by the chunk hash already in every key). Only
  worker 0 reports: bindings depend on token content alone, so one
  report covers every rank's keys. Other store paths (engine-driven
  transfer, blend pre-computed docs, experimental qstore) do not emit
  bindings yet — the engine-driven path can publish the same event from
  its ``commit_store`` when it needs directory tokens.
- **`CacheEventSubscriber`** maps those events onto the directory
  vocabulary (writes → `STORE`, evictions/deletes → `DELETE`, split per
  actual L1 medium from the event metadata; touches → `ACCESS`). The
  token-binding events produce no batches of their own: the subscriber
  remembers their chunk-hash → (token ids, offset) pairs (LRU cache
  bounded at
  65536; passing the bound evicts the oldest half in one batch, so
  eviction — and its warning — stays rare) and stamps `token_ids` and
  `token_offset` onto
  every L1/L2 `STORE` entry,
  so token bindings ride the store events themselves. Tokens are
  therefore repeated per rank/group/tier placement — an accepted wire
  trade for a self-contained protocol (see
  [key_directory.md](key_directory.md) — Token index).
  `ACCESS` batches carry an **empty backend**: the directory only
  refreshes key-level recency on access, so there is no placement
  identity to name — the vocabulary requires a non-empty backend for
  `store`/`delete` only. The subscriber is single-threaded by design —
  everything runs on the bus's drain thread, so it needs no locking.
- **Threading.** The bus dispatches on one drain thread, which is
  exactly the per-instance FIFO the directory needs. The subscriber
  self-paces delivery: recording flushes when `flush_interval` has
  elapsed since the last flush, bounding the sink-publish rate under
  load. There is no timer of its own — the subscriber additionally
  subscribes to `l1.eviction.loop_tick` (published continuously by the
  L1 eviction loop) as a flush pump, so a burst-ending tail (e.g. L2
  store completions) is delivered within one tick of the interval
  elapsing instead of waiting for the next request. The sink posts
  synchronously with a short timeout (a slow coordinator briefly
  stalls the drain, bounded by the timeout; overflow beyond the bus's
  bounded queue is dropped and surfaces as a `seq` gap → replay).
- **Coupling.** The stream requires the bus: enabling
  `--coordinator-event-reporting` together with
  `--disable-observability` is rejected at startup. Bus-level drops
  under overload are acceptable by the same argument as transport loss —
  the directory is eventually consistent soft state.

## L1 media

L1 media are a closed set, hence the `L1BackendType` enum
(`distributed/api.py`: `DRAM`, `DEVDAX`, `GDS`); L2 backends stay
strings because adapter types are an open registry (plugins register
new type names).

The `shared` flag is tier-agnostic, and L1 already contains the
shared-capable medium: `DEVDAX` (e.g. CXL-attached memory exposed as a
`/dev/dax` device) can be mapped by several instances, while `DRAM` and
`GDS` are inherently instance-private. Today each instance uses its
DevDAX region privately (its own allocator, its own lifetime), so L1
events emit `shared=False`. When pooled DevDAX lands (allocation
governed by the pool's own controller, reporting still per instance),
only the producer side changes: the subscriber already splits L1
records per `L1BackendType`, so it stamps `shared` on the pooled
backend's runs — the vocabulary, batching identity, and directory
semantics need no change.

## Engine-format KV event sink

Module: `lmcache/v1/mp_coordinator/kv_event_sink.py`

KV-cache-aware routers (llm-d's EPP; see the RFC linked from
[issue #4352](https://github.com/LMCache/LMCache/issues/4352)) learn
placements only from engine KV events: vLLM `KVEventBatch` msgpack over
a ZMQ PUB socket, topic `kv@<emitter_id>@<model>`. In MP mode the MP
connectors emit nothing there, so LMCache tiers are invisible to such a
router. `ZmqKVEventSink` is a second `CacheEventSink` that re-emits the
same `CacheEventBatch` stream in that wire format, so an unmodified vLLM
adapter indexes LMCache's tiers next to the engines' GPU tier.
`CompositeCacheEventSink` fans one subscriber out to both sinks when the
coordinator sink is also configured.

Translation (vLLM positional layout, tag first; HMA fields never set):

| vLLM wire | Source |
|---|---|
| topic `kv@<emitter_id>@<model>` | one message per configured emitter id (`--kv-events-emitter-ids`, default `node:<node name>`); `model` from the entry's `ObjectKey` |
| `BlockStored.block_hashes` | `[chunk_hash]` (raw bytes; routers truncate to the last 8) |
| `BlockStored.parent_block_hash` | `CacheEventEntry.parent_hash_hex`: the previous chunk in the same `mp.tokens` binding event, `nil` for the first (sequence start or a predecessor stored by an earlier request) |
| `BlockStored.token_ids` / `block_size` | the entry's `token_ids` / their count (the chunk size) |
| `medium` | `lmcache-l1` for L1, `lmcache-l2-<backend>` for L2 — identical on store and delete (routers refcount per medium) |
| `BlockRemoved` | `delete` batches (L1 evictions, L2 deletes), one event per batch |
| `access`, `config` | not forwarded |

A tokenless `store` entry (token-binding cache miss) is skipped: llm-d
recomputes its own keys from the tokens and cannot index an unknown hash
without them. A `delete` is never skipped. The `seq` frame is one counter
across all topics, numbered from 0 like vLLM; with
`--kv-events-replay-port` a ROUTER socket answers vLLM-style replay
requests (`[b"", start_seq:8]` → the buffered `[topic, seq, payload]`
frames from `start_seq` on, then `[b"", -1:8, b""]`) from a bounded
ring of recent messages. Sends are non-blocking (PUB drops at its
high-water mark), so a slow router never stalls the bus drain thread.

Deliberately out of scope here, tracked in the RFC: engine identity from
`REGISTER_KV_CACHE` (fan-out currently uses configured ids only), and
mounting the same sink on the coordinator's `POST /events` so shared L2
is credited fleet-wide once.

## Wiring and configuration

Enabled in the MP HTTP server lifespan when a coordinator URL is set
and `--coordinator-event-reporting` (or
`LMCACHE_COORDINATOR_EVENT_REPORTING`) is on, and/or when
`--kv-events-endpoint` (or `LMCACHE_KV_EVENTS_ENDPOINT`) is set;
`--coordinator-event-flush-interval` paces the subscriber's
event-driven flushes (default 1s). Both ride the observability bus, so
both are rejected with `--disable-observability`.

## Known limitations (follow-ups)

- **Bus overflow drops events silently** (bounded queue, rate-limited
  warning); the resulting `seq` gap marks the instance's slice stale.
  Reconstruction is by replaying the event stream (durable-transport
  retention) — wiring that replay up is future work.
- **The flush pump is coupled to the eviction loop's tick** — decouple
  it (e.g. a bus-owned periodic hook) so tail freshness does not depend
  on that loop's cadence.

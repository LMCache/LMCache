# KV event channel (`lmcache/v1/multiprocess/modules/kv_events.py`)

Module: `lmcache/v1/multiprocess/modules/kv_events.py`
Wire contract: `lmcache/v1/multiprocess/protocols/kv_events.py`,
`lmcache/v1/multiprocess/custom_types.py` (`KVEventRecord`, `KVEventPollResult`)
Consumer: `LMCacheMPWorkerAdapter` and `LMCacheMPConnector` in
`lmcache/integration/vllm/` (vLLM `BlockStored` / `BlockRemoved`)
User docs: `docs/source/production/dynamo_coordination.rst`

## Problem

A KV-aware router (NVIDIA Dynamo, or any consumer of vLLM's KV event stream)
decides where to send a request from the blocks each worker holds. With
LMCache MP, a worker's host-cache placements change inside the MP server:
its own stores complete there, other engines sharing the server store
chunks it can also retrieve, and the L1 eviction loop and L2 controllers
add and remove chunks without any engine noticing. PR #5076 taught the
vLLM connector to publish a worker's *own completed stores*; nothing told
the router when those chunks left the host cache, so its view went stale
after the first eviction.

## Design

```
storage layer ──► EventBus ──► KVEventSubscriber ──► KVEventLog ──► POLL_KV_EVENTS ──► worker adapter ──► connector ──► vLLM publisher
 (L1/L2 key      (drain        (keys → records,     (bounded,      (SYNC handler,     (non-blocking     (BlockStored /     (ZMQ → router)
  events)         thread)       token bindings)      sequenced)     cursor + model)    poll per step)     BlockRemoved)
```

The MP server keeps a **bounded, sequenced log** of key-level cache events
and engine workers **poll** it. This keeps the server unaware of routers
and engines (it serves any consumer that can poll), reuses the
observability bus the storage layer already publishes on, and lets the
engine's own KV event publisher carry the result, so the router needs no
new transport.

### Server side

- `KVEventSubscriber` consumes `L1_WRITE_FINISHED`,
  `L1_WRITE_FINISHED_AND_READ_RESERVED`, `L1_KEYS_EVICTED`,
  `L2_KEYS_STORED`, `L2_KEYS_DELETED`, and `MP_TOKENS`. It runs on the bus's
  single drain thread, so its own state (the token-binding cache) needs no
  lock.
- One **record per chunk** for stores (`kind="stored"`, one hash, the chunk's
  tokens, its predecessor's hash, the chunk size) and one **record per
  model** for removals (`kind="removed"`, every distinct hash in the event).
  KV ranks and object groups collapse into one record because a router
  tracks chunks, not shards. Media are vLLM's: `CPU` for L1, `STORAGE` for L2.
- **Token bindings.** A router keys its radix index by the block's tokens and
  parent hash, so a stored record must carry them. The store path publishes
  `MP_TOKENS` (chunk hashes, tokens, offsets, and now `parent_hashes`) ahead
  of the write-finished events; the subscriber caches them (LRU, 65536
  entries). A store whose binding is unknown (binding evicted, or an L2
  prefetch of a chunk stored long ago) is **counted and skipped**
  (`unbound_stores` in `report_status`), never reported without tokens.
  The first chunk of a mid-sequence store gets its parent from the session's
  hash chain (`_publish_token_bindings`).
- `KVEventLog` assigns consecutive sequence numbers from 1 and discards the
  oldest records past `capacity` (`--kv-event-log-size`, default 32768; 0
  disables the channel). `read_after(cursor, model_name, max_events)`
  returns the records after `cursor` for one model, the next cursor, and
  `lost`: the cursor predates the oldest retained record, the cursor was
  never issued by this log, or a **loss marker** sits in the scanned range.
  The marker is appended whenever the bus's dropped-event count grows
  (checked on every subscriber callback and on every poll, so a dropped
  eviction with no later bus traffic still surfaces); records before the
  last marker in range are withheld, because their world may be
  incomplete.
- `KVEventModule` owns the log and the subscriber, serves
  `POLL_KV_EVENTS` (SYNC: it only copies records out of memory), and stamps
  every answer with an **incarnation** (`time.time_ns()` at module
  construction). It is disabled, answering `enabled=False`, when the log
  size is 0 or the observability bus is off (`--disable-observability`),
  since no event could reach the log.

### Worker side (`LMCacheMPWorkerAdapter`)

- **One poller per server.** Every rank attached to a server reads the same
  records, and a repeated `BlockRemoved` makes a KV-aware router return
  `BlockNotFound` and abort the rest of the batch, unlike a repeated
  `BlockStored`. So exactly one rank per server polls
  (`ParallelStrategy.is_kv_event_poller`, the first rank of each server's
  contiguous block), while every rank attached to an advertising server
  stops announcing its own stores: one publisher per engine.
- `get_kv_events()`, called every model-runner step, advances a
  **non-blocking** poll. It buffers a completed poll's records, then issues
  the next once `lmcache.mp.kv_event_poll_interval` elapsed, or at once
  after a full page. The step never waits on the server.
- **Server records are the only store source while polling.** A store
  result only says the request finished without a fatal error: chunks the
  storage manager could not reserve are skipped silently and the result is
  still `True`, so #5076's own-store events can announce chunks that were
  never written (seen in the e2e: four filler stores skipped, all
  announced). `L1_WRITE_FINISHED` records name exactly the chunks written.
  With polling off, or once it is disabled, own stores are announced as
  before.
- **Announced set.** Per medium, the chunk hashes this worker announced. A
  removal is reported only for announced chunks, and a store only for
  chunks not yet announced, so duplicate records collapse into one event.
  The set mirrors the server's live key set for this model, so the server's
  cache capacity bounds it.
- **Resync.** On an incarnation change (server restart) or a `lost` answer,
  the adapter withdraws every announced chunk, one `BlockRemoved` per
  medium, and clears the set; placements return as chunks are stored again.
  `AllBlocksCleared` is never used, as it would also wipe the router's
  GPU-tier view. `lost` on first contact is ignored, since nothing was
  announced from that log yet.
- **Version skew.** A worker polls only a server advertising `kv_events`
  through `GET_EXPERIMENTAL`, which the adapter already queries at
  construction. This is not an optimization: a server that predates
  `POLL_KV_EVENTS` decodes the request-type frame outside any `try` in its
  request loop, so one poll would abort that loop for every engine attached
  to it. The capability is advertised on the ZMQ transport only, because the
  gRPC client builds its methods from the generated service descriptors,
  which do not carry this request yet.

### Connector side (`LMCacheMPConnector`)

- `get_kv_connector_kv_cache_events()` converts `CacheStoreEvent` to
  `BlockStored` and `CacheRemoveEvent` to `BlockRemoved`, keeping the medium.
- `LMCacheMPKVEvents.aggregate()` merges the ranks' batches as an
  **order-preserving, deduplicated union** (`kv_event_merge.py`) rather than
  vLLM's `KVEventAggregator` intersection. That aggregator counts only the
  ranks that reported in a step and keeps what all of them reported, so
  store-completion skew drops events, and with several MP servers per engine
  the disjoint pollers would intersect to nothing. A repeated `BlockStored`
  is idempotent for a router, and removals never arrive twice because one
  rank per server polls.
- The scheduler keeps one aggregated batch per step, and `take_events()`
  returns them in order.

## Guarantees and limits

- Exactly the chunks a worker can serve from LMCache are announced, once,
  in the order they became (un)available, within one poll interval plus the
  bus drain lag.
- Every loss of fidelity is explicit: `lost` and incarnation changes
  resync (metric `vllm:lmcache_mp_kv_event_resyncs_total{reason}`),
  unknown bindings are counted, and a disabled channel is logged once.
- Not covered: access (LRU touch) events, which vLLM's event vocabulary
  lacks; salted requests, whose vLLM-side block hashes include the salt
  while LMCache chunk hashes do not (removals stay conservative); shared L2
  backends are reported as this worker's `STORAGE` tier, not as a shared
  pool.

## Configuration

Operator-facing settings are documented in
`docs/source/mp/configuration.rst`: the server's `--kv-event-log-size` and
the connector's `lmcache.mp.kv_event_poll_interval`. Two further conditions
are not settings of this module: the channel needs the observability event
bus, which `--disable-observability` turns off, and the connector's
`lmcache.mp.hash_algorithm` must match the server's `--hash-algorithm`, or
the chunk hashes on both sides disagree.

## Observability

- Server: `report_status()["kv_events"]` and the gauge
  `lmcache_mp.kv_events.log_depth`.
- Worker, on the vLLM Prometheus registry:
  `vllm:lmcache_mp_kv_events_{generated,drained}_total`, `..._buffered`,
  `vllm:lmcache_mp_kv_event_polls_total` and
  `vllm:lmcache_mp_kv_event_resyncs_total{reason}`.

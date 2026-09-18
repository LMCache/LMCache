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
  records, and a repeated `BlockRemoved` is an error for a KV-aware router
  (`lower_tier.rs` returns `BlockNotFound` and aborts the rest of the batch),
  unlike a repeated `BlockStored`. So exactly one rank per server polls
  (`ParallelStrategy.is_kv_event_poller`, the first rank of each server's
  contiguous rank block), while **every** rank attached to an advertising
  server stops announcing its own stores. The engine therefore has a single
  publisher.
- `get_kv_events()` (called by the connector every model-runner step)
  advances a **non-blocking** poll: it consumes a completed poll's records
  into the event buffer, then issues the next poll once
  `lmcache.mp.kv_event_poll_interval` (default 0.1 s) elapsed, or at once
  after a full page of 1024 records.
  The step never waits on the server.
- **Server records are the only store source while polling.** A store
  result only says the request completed without a fatal error: chunks the
  storage manager cannot reserve (allocation failure while the eviction loop
  lags) are skipped silently and the result is still `True`, so the own
  completed-store events of #5076 can announce chunks that were never
  written (observed in the e2e: four filler stores skipped, all announced).
  With polling on, the adapter builds no own-store events and announces
  stores from `L1_WRITE_FINISHED` records, which name exactly the chunks
  written, one poll interval later. With polling off (or once it is
  disabled), own completed stores are announced as before.
- **Announced set.** The adapter tracks, per medium, the chunk hashes it has
  announced as stored. A removal record is reported only for announced
  chunks (a router counts a removal of an unknown block as an error), and a
  stored record only for chunks not yet announced, so duplicate records
  (one per KV rank, or a store repeated) collapse into one `BlockStored`.
  The set mirrors the server's live key set for this model, so the server's
  cache capacity bounds it.
- **Resync.** When the incarnation changes (server restart) or the answer
  says `lost`, the adapter withdraws *every* announced chunk with one
  `BlockRemoved` per medium and clears the set; placements return as chunks
  are stored again. `AllBlocksCleared` is not used: it would also wipe the
  router's GPU-tier view of the worker. `lost` on first contact is ignored
  (nothing was announced from that log yet).
- **Version skew.** A worker polls only a server that advertises
  `kv_events` through `GET_EXPERIMENTAL`, which the adapter already queries
  at construction. This is not an optimization: a server that predates
  `POLL_KV_EVENTS` decodes the request-type frame outside any `try` in its
  request loop, so one poll would abort that loop and take the cache server
  down for every engine attached to it. The `mq_timeout` branch then covers
  only a server that advertises the channel and stops answering. When the
  capability is absent, or
  polling is disabled at runtime, the worker falls back to announcing its
  own completed stores exactly as #5076 does.

### Connector side (`LMCacheMPConnector`)

- `get_kv_connector_kv_cache_events()` converts `CacheStoreEvent` to
  `BlockStored` and `CacheRemoveEvent` to `BlockRemoved`, keeping the
  medium.
- `LMCacheMPKVEvents.aggregate()` merges the tensor-parallel workers'
  batches as an **order-preserving, deduplicated union**
  (`kv_event_merge.py`), not vLLM's `KVEventAggregator` intersection. That
  aggregator counts only the ranks that reported something in a step and
  keeps what all of them reported, so an event one rank reports while
  another reports a different batch in the same step is dropped for good.
  Store completions skew across ranks, which makes that the normal case.
  The union is safe for stores because a router applies a repeated
  `BlockStored` idempotently; removals never arrive twice because only one
  rank per server polls.
- The scheduler keeps one aggregated batch per step and `take_events()`
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

| Where | Setting | Default | Meaning |
|---|---|---|---|
| MP server | `--kv-event-log-size` | 32768 | Records retained; 0 disables the channel |
| MP server | observability bus | on | `--disable-observability` disables the channel |
| vLLM connector | `lmcache.mp.kv_event_poll_interval` | 0.1 s | Seconds between polls; 0 disables polling |
| vLLM connector | `lmcache.mp.hash_algorithm` | `blake3` | Must match the server's `--hash-algorithm` |

## Observability

- Server: `report_status()["kv_events"]` (`enabled`, `incarnation`,
  `log_depth`, `log_capacity`, `next_seq`, `lost_markers`,
  `unbound_stores`), gauge `lmcache_mp.kv_events.log_depth`.
- Worker (vLLM Prometheus registry): `vllm:lmcache_mp_kv_events_generated_total`,
  `..._drained_total`, `..._buffered`, `vllm:lmcache_mp_kv_event_polls_total`,
  `vllm:lmcache_mp_kv_event_resyncs_total{reason=server_restart|events_lost}`.

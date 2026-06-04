# Valkey L2 Adapter Design

This document describes the built-in `valkey` L2 adapter for LMCache
multiprocess mode. The adapter stores KV cache chunks in a Valkey (or
Redis) instance, supporting both standalone and Valkey-Cluster
topologies. It uses the official `valkey-glide` Python sync client.

## Goals

- Provide a first-class Valkey-Cluster backend for LMCache MP mode.
- Preserve **zero-copy** SET and GET for multi-megabyte KV chunks.
- Reuse the existing wire-key format so `cache_salt`-based per-tenant
  isolation works without adapter-specific code.
- Stay within the standard `L2AdapterInterface` contract — submit,
  event-fd, query-result, no controller changes.

## Why glide sync (not async)

`glide-async` cannot do zero-copy for either SET or GET. The async
client serializes commands as Protobuf messages across an in-process
Rust-runtime boundary, which forces a `bytes` copy of every value.
glide upstream documents this explicitly: *"zero-copy is sync-only"*
(see [valkey-glide#5492][pr-5492] and [#5493][pr-5493]).

KV cache chunks are multi-megabyte (∼10 MB per chunk for a 70B model
with TP=8) and every request touches hundreds of chunks. Paying a copy
on every transfer would dominate the wire time. Sync glide is the only
configuration that preserves zero-copy via CFFI (`ffi.from_buffer`), so
this adapter standardizes on it.

A sync API blocks the calling Python thread per call, so the adapter
runs a `ThreadPoolExecutor` of independent worker threads to drive
concurrent in-flight requests. Each worker holds its own glide client
via `threading.local()` — glide's internal multiplexing makes one
client capable of high concurrency, but the sync Python API only lets a
single thread submit one command at a time, so N threads × N clients
is what unlocks batch-level parallelism.

[pr-5492]: https://github.com/valkey-io/valkey-glide/pull/5492
[pr-5493]: https://github.com/valkey-io/valkey-glide/pull/5493

## Dependencies

- `valkey-glide-sync >= 2.3.0` — the **sync** GLIDE client package
  (provides the `glide_sync` module), the first stable release with both
  zero-copy SET (`#5492`) and buffer GET (`#5493`). It bundles
  `glide_shared` and `glide_sync`. 
  
  It's lazy imported, won't get used unless triggered 
  , if host is missing dependency, following error will be raised

  If valkey is installed but version is outdated, 
  valkey adaptor will fall back to use copying through a `bytes` 

  ```
  Valkey support requires the glide_sync module.
  Install: pip install 'valkey-glide-sync>=2.3.0'
  ```

## Wire key layout

key be written to Valkey has the form:

```
[<key_prefix>@]<model_name>@<kv_rank_hex>@<chunk_hash_hex>[@<cache_salt>]
```

## Threading model

```
                ┌──────────────── L2 controller ─────────────────┐
                │ submit_store_task / lookup / load              │
                │   ▼                                             │
ValkeyL2Adapter │ submit one Future per key to ThreadPoolExecutor│
                │   │                                             │
                └───┼─────────────────────────────────────────────┘
                    │
        ┌───────────┴──────── worker pool ─────────────┐
        │ N threads, each with thread-local            │
        │ GlideClient / GlideClusterClient             │
        │                                              │
        │ ┌────────── Future.add_done_callback ──────┐ │
        │ │  Atomically decrement batch.remaining,   │ │
        │ │  record per-key ok/fail.                 │ │
        │ │  Last finisher publishes the result and  │ │
        │ │  signals the matching event fd.          │ │
        │ └──────────────────────────────────────────┘ │
        └──────────────────────────────────────────────┘
```

### Per-batch state (`_SubmitTaskBatchState`)

Each `submit_*_task` allocates a `_SubmitTaskBatchState` shared by N
per-key futures:

| Field         | Purpose                                                       |
|---------------|---------------------------------------------------------------|
| `task_id`     | Public id returned to the caller.                             |
| `remaining`   | Decremented under `lock`; zero ⇒ this is the finishing callback.|
| `per_key_ok`  | Indexed by submit-order; written by each callback.            |
| `keys`        | Original keys (for listener notifies + per-salt accounting).  |
| `sizes`       | Bytes per key (store batches only — others leave empty).      |


## Size validation on load

`_do_get_into` returns `False` (cache miss) whenever the GET result is
inconsistent with the expected buffer size:

1. **Buffer-GET path** — glide returns the number of bytes written;
   if `bytes_written != len(buf)`, the stored value is stale or
   truncated and treated as a miss.
2. **Fallback path** — `len(data) != len(buf)` yields the same miss
   semantics.

In both cases the load bitmap reports a `0` bit for that key, and the
chunk is recomputed by the engine — no stale data ever flows out of
the adapter.

## Partial-failure accounting

A store task is reported as failed if **any** key in it fails to write.
This is the safe choice: the store controller only drops a task's keys
from L1 when the task succeeds, so reporting success on a partial failure
would evict the un-stored keys from L1 and lose them.

Byte accounting is finer-grained. `_notify_keys_stored` is fired with
**only** the keys that actually wrote, and the base class folds those
into per-`cache_salt` and aggregate totals — so `get_usage()` stays
accurate even when the task as a whole is marked failed.

## Cluster vs standalone

| Mode         | glide class            | `database_id` | startup nodes used |
|--------------|------------------------|---------------|---------------------|
| standalone   | `GlideClient`          | honored       | first only (warned if >1) |
| cluster      | `GlideClusterClient`   | ignored (warned) | full list — seeds for discovery |

### Slot routing & redirects (handled by glide)

Each key belongs to one of 16384 slots (`CRC16(key) % 16384`), and each
slot is owned by one primary. The client caches a slot→node map and
sends each request straight to the owning node. That map can go stale
(after a reshard or a failover), so the request lands on the wrong node
and the server replies with a **redirect** — the cluster's self-healing
mechanism, not an error:

- **MOVED** — the slot's owner has **permanently changed** (reshard
  finished, or a replica was promoted). The client retries on the new
  node and updates its cached map.
- **ASK** — the slot is **mid-migration** and *this specific key* has
  already moved to the target while the slot still officially belongs to
  the source. The client sends `ASKING` + the command to the target for
  this one request and does **not** update its map (the migration isn't
  done).

All of this — initial discovery (`CLUSTER SHARDS`), MOVED/ASK handling,
`ASKING`, and topology refresh — is owned by `GlideClusterClient`. The
adapter never parses redirects: a `client.get`/`set` call returns only
the final, post-redirect result.

### Cluster routing & load balancing

The adapter performs **no explicit load balancing** — distribution
happens at two independent layers:

1. **Across nodes** — glide hashes each key to one of 16384 slots
   (`CRC16(key) % 16384`) and routes to the slot's owning primary. Our
   wire key embeds a uniform content hash (`chunk_hash`) and contains
   **no hashtag** (`{...}`), so keys spread evenly across slots and
   therefore across nodes with no extra work.
2. **Across worker threads** — a batch fans out one future per key onto
   the `ValkeyWorkerPool`'s `num_workers` threads, each holding a full
   cluster client. A large batch thus parallelizes over both threads and
   nodes simultaneously; keys landing on different nodes is normal and
   handled transparently by glide.

Known imbalance sources / tuning levers:

- **Reads hit primaries only.** Replica reads are **not enabled**
  (no read-from-replica preference is set), so replicas provide
  redundancy but do not share read load. Read-heavy workloads put all
  GET traffic on primaries.
- **Hot keys.** A single very popular chunk (e.g. a shared system
  prompt) maps to one slot → one node, creating a read hotspot that
  uniform hashing cannot spread. Replica reads would help here.
- **Pool size vs node count.** Concurrency is capped at `num_workers`
  (default 8). On a large cluster, the thread pool — not the cluster —
  can become the bottleneck; raise `num_workers` to saturate more nodes.
- **No hashtag routing.** Co-locating a `cache_salt`'s keys on one node
  (via a `{cache_salt}` hashtag) is intentionally **not** done; it would
  aid pipelining but risk hot shards. Uniform spread is preferred.

`report_status()` exposes two different views of usage:

- `current_size_bytes` — LMCache's **aggregate logical** byte count. In
  cluster mode this aggregate can mask a hot shard (one node full while
  others are nearly empty).
- `node_memory` (cluster mode only) — the server's **real per-node
  physical** memory, collected by routing `INFO memory` to all nodes
  (`used_memory` / `maxmemory` per node). Use this to spot shard
  imbalance. It is best-effort: an empty dict means the `INFO` query
  failed and never blocks status reporting.

**Resharding** (adding/removing nodes) is transparent to the adapter:
glide handles `MOVED`/`ASK` redirects and topology refresh, and seed
nodes need not be updated. A graceful reshard migrates keys and keeps the
cache warm; dropping a node without migrating its slots loses those keys,
which simply surface as cache misses and LMCache's byte accounting may drift afterward

## Authentication and TLS

`ServerCredentials(username, password)` is passed to glide when either
field is non-empty. `tls_enable=True` sets glide's `use_tls`. Credentials
come from the config only; the resolved values never appear in
`report_status()` output or logs.

### TLS scope and limitations

`tls_enable` is an on/off switch (glide's `use_tls`) with no certificate
options. It works only when the server's certificate is verifiable
against the client's existing OS trust store — i.e. publicly-signed
certs (ElastiCache Serverless, Let's Encrypt).

Custom-certificate setups are a planned follow-up (exposing glide's
`TlsAdvancedConfiguration`):

| Deployment | `tls_enable` alone |
|------------|--------------------|
| Public-CA cert (ElastiCache Serverless) | ✅ works |
| Self-signed cert | ⏳ to be supported |
| Private / internal CA | ⏳ to be supported |
| Mutual TLS (mTLS) | ⏳ to be supported |

## Capacity and eviction

- `max_capacity_gb > 0` enables aggregate (global) eviction signals
  via `get_usage().usage_fraction`. Set this when LMCache should track
  total bytes used and trigger eviction at a watermark.
- Per-`cache_salt` quotas operate regardless of `max_capacity_gb` —
  the base class tracks per-salt totals from `_notify_keys_stored` /
  `_notify_keys_deleted` for any quota policy.

`delete(keys)` is synchronous: it submits one DEL per key to the pool,
waits for each completion (up to `request_timeout`), and fires
`_notify_keys_deleted` with the per-key sizes captured at store time
(via `_key_sizes`). Keys whose DEL returned `0` (already absent) are
silently skipped.

### Two eviction layers — configure only one

There are **two independent eviction mechanisms** and they can conflict:

1. **LMCache** (this controller) — tracks bytes via `get_usage()` and
   issues `delete()` calls. It understands LRU ordering, prefix-chain
   ordering, per-`cache_salt` quotas, and locked keys.
2. **The Valkey server** — its own `maxmemory` + `maxmemory-policy`
   evicts keys when server memory fills, **without notifying LMCache**.

Pick exactly one as the authority. **The recommended setup is
LMCache-driven: leave Valkey server-side eviction off (`maxmemory 0` /
`noeviction`) and let LMCache evict via `max_capacity_gb > 0`.** LMCache's
policy is strictly smarter than the server's — it honors prefix-chain
ordering, per-`cache_salt` quotas, and read-locked keys, none of which a
plain `allkeys-lru` understands.

| Setup | Valkey config | LMCache config |
|-------|---------------|----------------|
| **LMCache-driven** (recommended) | `maxmemory 0` (or large) + `noeviction` | `max_capacity_gb > 0` |
| **Server-driven** (only if LMCache eviction is intentionally disabled) | `maxmemory <N>` + `allkeys-lru` | `max_capacity_gb = 0` |

> **Note** — this guidance is specific to the MP-mode `valkey` L2
> adapter. The non-MP `ValkeyConnector` has no eviction layer of its own
> and *must* rely on Valkey server-side eviction, so the recommendation
> there is the opposite.

Running both (a server `maxmemory` cap *and* `max_capacity_gb > 0`) lets
the server silently drop keys that LMCache still counts, so LMCache's
byte accounting and LRU order drift from reality. This is a
**performance / accounting** issue, not a correctness one: a server-side
drop surfaces as a lookup/load miss (handled by the size-validation and
per-key miss paths above), so no stale data is ever returned — but the
LMCache-side LRU bookkeeping will hold ghost keys.

This adapter does not set a TTL on SET, so keys never expire on the
server by time; lifetime is governed entirely by whichever eviction
authority is configured above.

## Lock semantics

Locking is **client-side**: glide / Valkey have no notion of LMCache's
eviction-pinning concept. The adapter maintains a per-key refcount in
`_locked_keys`; a successful `lookup` increments it, `submit_unlock`
decrements it, and refcounts at zero are removed. The adapter never
prevents eviction on the server side — pinning is enforced only at the
L2 controller layer above this adapter.

## Error model

| Failure                              | Per-key bit | Logged          | Notes                                |
|--------------------------------------|-------------|-----------------|--------------------------------------|
| Single SET raises                    | `False`     | `WARNING`       | Other keys in batch unaffected.      |
| Single GET raises / times out        | `False`     | `WARNING`       | Cache miss; chunk recomputed.        |
| GET size mismatch                    | `False`     | `DEBUG`         | Stale/truncated value rejected.      |
| Single EXISTS raises                 | `False`     | `WARNING`       | Looks like a miss.                   |
| Single DEL raises                    | (skipped)   | `WARNING`       | Listener not notified for that key.  |
| Cluster MOVED/ASK                    | (handled by glide) | —        | Transparent to the adapter.          |
| Connection drop                      | (handled by glide) | —        | Glide reconnects; calls retry.       |
| Missing `valkey-glide-sync`       | construction fails | (raises) | Clear actionable error.              |

## Configuration

JSON schema (CLI `--l2-adapter`):

```json
{
  "type": "valkey",
  "startup_nodes": "host1:6379,host2:6379",
  "cluster_mode": true,
  "username": "...",
  "password": "...",
  "key_prefix": "prod",
  "num_workers": 8,
  "tls_enable": true,
  "request_timeout": 5.0,
  "connection_timeout": 10.0,
  "max_capacity_gb": 0
}
```

Shortcut for a single node: `{"host": "host1", "port": 6379, ...}`.

## Non-goals

- **Custom MOVED/ASK parsing**: delegated to glide.
- **Custom slot map**: delegated to glide.
- **C++ extension**: glide's Rust runtime already runs FFI-released
  GIL; a hand-rolled C++ connector would duplicate glide.
- **Async-glide path**: cannot do zero-copy (see *Why glide sync*).
- **EXISTS-time size check**: RESP `EXISTS` doesn't return size; the
  subsequent GET enforces it instead.
- **Advanced TLS** (self-signed / private CA / mTLS): only the basic
  `use_tls` flag is wired now — to be supported (see *TLS scope and
  limitations*).

## Testing

- `tests/v1/distributed/test_valkey_l2_adapter.py` — unit tests
  against an in-process fake `glide_sync` module. 
- `tests/v1/distributed/test_valkey_l2_adapter_integration.py` - integration tests running with real local valkey instance

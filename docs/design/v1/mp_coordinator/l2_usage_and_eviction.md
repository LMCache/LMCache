# Fleet-Wide L2 Usage Tracking and Eviction

A coordinator-level capability that gives fleet-wide visibility into per-tenant
L2 cache usage and enforces per-``cache_salt`` byte quotas via LRU eviction.
MP servers **report store/lookup events** to the coordinator; the coordinator
aggregates usage, manages quotas, and periodically selects LRU keys to evict
when a tenant exceeds its quota. It is **opt-in** (gated by
``l2_event_reporting`` in ``CoordinatorConfig``) and **additive** (the existing
per-server eviction is unchanged).

Code: `lmcache/v1/mp_coordinator/l2/` (coordinator side),
`lmcache/v1/mp_coordinator/http_apis/l2_api.py` (REST endpoints),
`lmcache/v1/mp_coordinator/schemas.py` (wire types),
`lmcache/v1/multiprocess/http_server.py` (MP-server wiring).

## Why

L2 eviction today is **local to each MP server**: the ``IsolatedLRUEvictionPolicy``
tracks only what that server stored and enforces quotas within that scope. With
a shared L2 backend (e.g. S3), multiple servers write to the same storage, but
no single server has a fleet-wide view of total per-tenant usage. The coordinator
centralizes usage accounting and quota enforcement so limits apply to the
aggregate, not per-replica.

## Architecture

```
MP server (store/lookup)
  L2 adapter fires on_l2_keys_stored / on_l2_keys_accessed
        │
        ▼
  L2EventListener (L2AdapterListener)
    converts ObjectKey → CacheKey, buffers UsageEvents
        │  flush every l2_event_flush_interval (default 1s)
        │
        ▼
  POST /l2/events ──▶ Coordinator
                        ├─ L2UsageManager: per-salt byte accounting
                        ├─ L2EvictionManager: per-salt LRU
                        └─ QuotaManager: per-salt byte limits

  Coordinator background loop (every eviction_check_interval, default 5s)
        │
        ▼
  execute_evictions():
    for each tracked salt:
      limit = quota (default 0 → evict all)
      if usage > limit → select LRU keys, log eviction plan
```

## Wire types (`schemas.py`)

- **``CacheKey``** — frozen dataclass: ``chunk_hash_hex``, ``model_name``,
  ``kv_rank``, ``cache_salt``. Torch-free equivalent of ``ObjectKey``;
  ``chunk_hash`` is hex-encoded instead of raw bytes.
- **``EventType``** — ``str`` enum: ``STORE``, ``LOOKUP``.
- **``UsageEvent``** — ``type: EventType``, ``key: CacheKey``, ``bytes: int``.
- **``ReportUsageRequest``** — batch of ``UsageEvent``s.

The ``ObjectKey`` → ``CacheKey`` conversion happens at the MP-server boundary
(``_object_key_to_cache_key`` in ``event_listener.py``), so the coordinator
never imports ``torch``.

## Coordinator components (`l2/`)

### L2UsageManager (`usage_manager.py`)

Thread-safe per-salt byte counter. Two operations:

- ``record_stored(cache_salt, n_bytes)`` — increment.
- ``record_evicted(cache_salt, n_bytes)`` — decrement (clamped at zero).

Exposes ``get(salt)``, ``get_all()``, ``get_total()`` for the status endpoints.

### QuotaManager (reused from ``lmcache.v1.distributed.quota_manager``)

Thread-safe in-memory quota registry (``dict[str, int]`` + lock). CRUD via
``set_quota``, ``get_limit_bytes``, ``delete_quota``, ``list_quotas``.
Quotas are set in GiB at the API and stored as bytes internally.
Unregistered salts default to a 0-byte limit (allowlist semantics).

### L2EvictionManager (`eviction_manager.py`)

Per-``cache_salt`` LRU, mirroring ``IsolatedLRUEvictionPolicy`` but using
``CacheKey`` and running in the coordinator process.

Data structures:

```
_per_salt_order : dict[str, OrderedDict[CacheKey, None]]   # LRU per salt
_key_sizes      : dict[CacheKey, int]                       # byte size per key
```

- ``on_store(key, size_bytes)`` — insert/refresh in LRU, record size.
- ``on_lookup(key)`` — touch (move to MRU end).
- ``on_remove(keys)`` — remove from LRU tracking after confirmed deletion.
- ``execute_evictions()`` — for each tracked salt, compare usage (from
  ``L2UsageManager``) against quota (from ``QuotaManager``, default 0). If over
  quota, select LRU keys targeting ``eviction_ratio`` of the overage. No quota
  or zero quota means evict all keys for that salt.

Eviction is currently **log-only**: ``execute_evictions`` returns the plan but
does not issue deletes. Once wired end-to-end, ``on_remove`` will be called
after the MP server confirms deletion.

## REST endpoints (`l2_api.py`)

| Method | Path | Description |
| --- | --- | --- |
| ``PUT`` | ``/l2/quota/{cache_salt}`` | Set quota (GiB) |
| ``DELETE`` | ``/l2/quota/{cache_salt}`` | Remove quota |
| ``POST`` | ``/l2/events`` | Ingest batch of store/lookup events |
| ``GET`` | ``/l2/status/{cache_salt}`` | Quota + usage for one salt |
| ``GET`` | ``/l2/status`` | Quota + usage for all salts |

Status responses report usage in GiB only (no raw bytes in the API).

## MP-server event listener (`event_listener.py`)

``L2EventListener`` implements ``L2AdapterListener`` and is registered
on all L2 adapters via ``StorageManager.register_l2_listener()``. It:

1. Receives ``on_l2_keys_stored(keys, sizes)`` and ``on_l2_keys_accessed(keys)``
   callbacks from the L2 adapter (any thread).
2. Converts each ``ObjectKey`` to ``CacheKey`` (hex-encodes ``chunk_hash``).
3. Buffers ``UsageEvent``s under a lock.
4. Flushes the buffer to ``POST /l2/events`` on a timer
   (``l2_event_flush_interval``, default 1s). Failures are logged and the
   batch is dropped to prevent unbounded growth.

``on_l2_keys_deleted`` is a no-op — the coordinator handles deletion via its
own eviction loop.

## Configuration

### Coordinator side (`MPCoordinatorConfig`)

| Field | Default | Description |
| --- | --- | --- |
| ``eviction_check_interval`` | ``5.0`` | Seconds between eviction cycles (0 disables) |
| ``eviction_ratio`` | ``0.5`` | Fraction of over-quota bytes to target per cycle |

### MP-server side (`CoordinatorConfig`)

| Field | Default | Env var | Description |
| --- | --- | --- | --- |
| ``l2_event_reporting`` | ``False`` | ``LMCACHE_COORDINATOR_L2_EVENT_REPORTING`` | Enable event reporting |
| ``l2_event_flush_interval`` | ``1.0`` | ``LMCACHE_COORDINATOR_L2_EVENT_FLUSH_INTERVAL`` | Seconds between flushes |

Both also accept CLI flags (``--coordinator-l2-event-reporting``,
``--coordinator-l2-event-flush-interval``).

## Failure modes

| Event | Effect | Handling |
| --- | --- | --- |
| Coordinator down | Events not delivered | Flush fails → batch dropped, logged; MP server unaffected |
| Coordinator restart | Usage/LRU state lost | Rebuilt from incoming events; stale until servers report |
| Flush timeout | One batch delayed | Next flush sends new batch; no retry of old batch |
| Usage accounting drift | Quota enforcement imprecise | Self-correcting as new events arrive |

## Scope

Additive: no change to per-server eviction, L2 adapter store/lookup paths, or
the coordinator's membership/health loop. Composes via the ``L2AdapterListener``
interface and the ``http_apis`` auto-discovery — a new router reading
``app.state``, plus the opt-in event listener — with no edits to existing
controllers or adapters beyond passing ``sizes`` through to
``on_l2_keys_stored``.

# Fleet-Wide L2 Usage Tracking and Eviction

A coordinator-level capability that gives fleet-wide visibility into per-tenant
L2 cache usage and enforces per-``cache_salt`` byte quotas via LRU eviction.
MP servers **report store/lookup events** to the coordinator; the coordinator
aggregates usage, manages quotas, and periodically selects LRU keys to evict
when a tenant exceeds its quota. It is **opt-in** (gated by
``event_reporting`` in ``CoordinatorConfig``, shared with the key-directory stream) and **additive** (the existing
per-server eviction is unchanged).

Code: `lmcache/v1/mp_coordinator/controllers/` (coordinator side),
`lmcache/v1/mp_coordinator/http_apis/cache_api.py` (REST endpoints),
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
  L2 adapter publishes l2.keys.stored / accessed / deleted on the bus
        │
        ▼
  CacheEventSubscriber (see cache_events.md)
    converts ObjectKey → EncodedObjectKey, buffers CacheEventBatches
        │  flush paced by event_flush_interval (default 1s)
        │
        ▼
  POST /events ──▶ Coordinator
                        EventGate: fencing / seq dedup / gap detection
                          │  (see ingest.md)
                          └─ CacheEventBroadcaster (admitted batches → consumers):
                             ├─ KeyDirectory.consume: placements
                             └─ FleetEvictionController.consume:
                                  ├─ L2UsageManager: per-salt byte view
                                  ├─ QuotaManager:   per-salt byte limits
                                  └─ IsolatedLRU:    eviction order

  Coordinator background loop (every eviction_check_interval, default 5s)
        │
        ▼
  execute_evictions():
    for each tracked salt:
      limit = quota (default 0 → evict all)
      if usage ≥ watermark·limit → select LRU keys,
        fire-and-forget DELETE /cache/objects to a holder
```

## Wire types (`schemas.py`)

- **``EncodedObjectKey``** — torch-free wire shape of ``ObjectKey`` (owned by
  ``lmcache.v1.distributed.api``); ``chunk_hash`` is hex-encoded instead of raw
  bytes. The coordinator rebuilds the canonical ``ObjectKey`` via
  ``key.to_object_key()``.
- **``CacheEventBatch`` / ``CacheEventEntry`` / ``CacheEventType``**
  (``mp_coordinator/api.py``) — the fleet cache-event vocabulary shared
  with the key directory; ``STORE`` carries ``size_bytes``, L2 lookups
  arrive as ``ACCESS``.

The ``ObjectKey`` → ``EncodedObjectKey`` conversion happens at the MP-server
boundary (``obj.to_encoded_object_key()`` in ``cache_events.py``).

## Coordinator components (`controllers/`)

The eviction controller owns the whole quota → usage → evict loop: the
budgets (target), what is spent against them (observation), and the LRU
that picks victims when a salt is over (action). Holding all three is
what makes it a controller rather than a bag of state — so they sit
behind one collaborator (`ctx.eviction_controller`, with `.quota` and
`.usage` for the read-only `/quota` endpoints) rather than as three
peers the app has to wire together in the right order. Its node-local
counterpart one scope down is
`distributed/storage_controllers/eviction_controller.py`.

### L2UsageManager (`usage_manager.py`)

The per-salt L2 byte totals are a **derived view of the global key
directory**, constructed and maintained in this manager rather than
inside the directory itself: the owning eviction controller feeds it
the same admitted batches the directory sees. Accounting mirrors the directory's placement
identity — re-storing a placement delta-adjusts, two private copies of
one key count twice, N reporters of one shared pool count once — and
only ``DELETE`` events remove bytes, so reporter restarts never zero
the view (L2 bytes persist on disk across restarts).

Exposes ``get(salt)``, ``get_all()``, ``get_total()`` for the status
endpoints and ``get_key_size(key)`` for eviction sizing.

### QuotaManager (reused from ``lmcache.v1.distributed.quota_manager``)

Thread-safe in-memory quota registry (``dict[str, int]`` + lock). CRUD via
``set_quota``, ``get_limit_bytes``, ``delete_quota``, ``list_quotas``.
Quotas are set in GiB at the API and stored as bytes internally.

Unregistered salts resolve through a configurable **default limit**
(``set_default_limit_bytes`` / ``effective_limit_bytes``), which starts as ``None``:

- ``None`` (boot default) — unregistered salts are **exempt** from coordinator
  eviction. Quotas live in memory, so a restarted coordinator has an empty
  table until the external quota controller re-syncs it; the exempt default
  makes that window safe (an empty table cannot mass-evict unknown tenants).
- ``0`` — strict allowlist: all bytes under unregistered salts become
  evictable next cycle. The external controller sets this via
  ``PUT /quota/config`` **after** re-registering every per-salt quota — it is
  the explicit signal that arms fleet-wide eviction of unquota'd data.
- ``> 0`` — every unregistered salt gets that byte budget.

The MP server's local eviction controller keeps the legacy
``get_limit_bytes`` path (unregistered ⇒ 0) and is unaffected by the default.
Because the server-local and coordinator quota registries are independent and
never synchronized, a deployment must register quotas through **one** of the
two ``/quota`` APIs, never both — the server-side enforcer fully evicts any
salt missing from its own table, so it would fight quotas registered only on
the coordinator (see the warning in ``docs/source/mp/coordinator.rst``).

### FleetEvictionController (`eviction_controller.py`)

Per-``cache_salt`` LRU for the fleet, plus ownership of the quota
registry and usage view above. It delegates the ordering to
a coordinator-side ``IsolatedLRUEvictionPolicy`` instance, keyed by the canonical
``ObjectKey`` (rebuilt from the wire ``EncodedObjectKey``). Per-salt byte
accounting lives in ``L2UsageManager``; the LRU only tracks order.

- ``consume(batch)`` — the `CacheEventConsumer` hook. Feeds the usage
  view **first**, then maps L2 entries onto the LRU (`STORE` registers,
  `ACCESS` touches, `DELETE` drops). A delete drops the key from the LRU
  only once its **last** L2 placement is gone: usage is per placement,
  so while another copy still holds bytes the key must stay evictable —
  otherwise those bytes could exceed quota with nothing for the planner
  to select. Usage consuming first is what makes that read of the
  post-batch size correct; it is now internal to this manager rather
  than a consumer-registration-order constraint.
- ``fence_instance(id)`` — no-op. A restart voids L1 state only; the L2
  bytes this manager accounts outlive the reporting process.
- ``on_store(key)`` — register the key in the LRU
  (``policy.on_keys_created``). The paired byte increment happens in the
  usage view consuming the same event.
- ``on_lookup(key)`` — touch (``policy.on_keys_touched``, move to MRU end).
- ``on_remove(key)`` — drop from the LRU (``policy.on_keys_removed``); the paired
  byte decrement happens in the usage view consuming the same event.
- ``compute_eviction_plan() -> dict[str, list[ObjectKey]]`` — **pure**: for each
  tracked salt, fire when ``usage ≥ trigger_watermark · quota`` (quota 0 ⇒ evict
  all), selecting ``eviction_ratio`` of the salt's LRU keys via
  ``policy.get_eviction_actions``. Salts without an explicit quota use the
  registry's default limit; while it is unset (``None``) they are skipped
  entirely — see QuotaManager above. No network, no mutation.
- ``run(registry, http_client, check_interval)`` — the control loop
  itself, started as an asyncio task by the app lifespan and cancelled on
  shutdown. ``EVICTION_CHECK_INTERVAL = 0`` disables it (the task is
  never created).
- ``execute_evictions(registry, http_client)`` — one pass: computes the plan and
  **fire-and-forget** ``DELETE /cache/objects`` to a holder MP server for each
  salt's victims; on confirmed deletion ``on_remove`` drops them from tracking.
  The plan's keys are split into chunks of at most ``MAX_DELETE_BATCH`` (imported
  from the MP server's ``object_service``) and each chunk is dispatched as its
  own request, because the endpoint rejects any single delete over that cap with
  HTTP 400 — a full-salt eviction (quota dropped to 0) routinely exceeds it.

## REST endpoints (`quota_api.py`)

| Method | Path | Description |
| --- | --- | --- |
| ``PUT`` | ``/quota/config`` | Set the default limit for unquota'd salts (``null`` ⇒ exempt; ``0`` arms allowlist eviction) |
| ``GET`` | ``/quota/config`` | Read the default limit |
| ``PUT`` | ``/quota/{cache_salt}`` | Set quota (GiB) |
| ``DELETE`` | ``/quota/{cache_salt}`` | Remove quota |
| ``GET`` | ``/quota/{cache_salt}`` | Quota + usage for one salt |
| ``GET`` | ``/quota`` | Quota + usage for all salts |

Event ingestion happens on ``POST /events`` (the fleet
cache-event stream): the ``EventGate`` admits each batch and the
``CacheEventBroadcaster`` fans it to the registered consumers — the key
directory and this manager, whose ``consume`` filters to ``l2``. The
gate's seq dedup is what stops replayed batches double-counting; see
[ingest.md](ingest.md).

The ``/quota/config`` routes are declared before ``/quota/{cache_salt}`` so the
literal ``config`` segment is not captured as a salt. Expected controller flow
after a coordinator (re)start: ``PUT /quota/{salt}`` for every tenant, then
``PUT /quota/config {"default_limit_gb": 0}`` to arm eviction of everything
else.

These quota/usage-accounting endpoints live in the ``/quota`` group (mirroring
the MP server's node-local ``/quota``); warm-prefetch dispatch is the only thing
left on the coordinator's ``/cache/*`` surface (``cache_api.py``). Paths are
tier-neutral; the tier is request data (`tier`, default `l2`). Status responses
report usage in GiB only (no raw bytes in the API).

## MP-server event emission (`cache_events.py`)

The L2 adapters publish key events on the observability bus; one
``CacheEventSubscriber`` per MP server turns them into ordered batches
delivered to ``POST /events`` (flushes paced by
``event_flush_interval``, default 1s); failures drop the batch (the
resulting seq gap flags replay). See
[cache_events.md](cache_events.md) for the subscriber/transport design.
``l2.keys.deleted`` becomes a ``DELETE`` event so the coordinator can
drop the key from its usage accounting and LRU tracking.

## Configuration

### Coordinator side (`MPCoordinatorConfig`)

| Field | Default | Description |
| --- | --- | --- |
| ``eviction_check_interval`` | ``5.0`` | Seconds between eviction cycles (0 disables) |
| ``eviction_ratio`` | ``0.2`` | Fraction of a salt's LRU keys (by count) to evict per cycle |
| ``trigger_watermark`` | ``1.0`` | Eviction fires when usage reaches this fraction of the quota |

### MP-server side (`CoordinatorConfig`)

| Field | Default | Env var | Description |
| --- | --- | --- | --- |
| ``event_reporting`` | ``False`` | ``LMCACHE_COORDINATOR_EVENT_REPORTING`` | Enable event reporting (also gates the key-directory stream) |
| ``event_flush_interval`` | ``1.0`` | ``LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL`` | Seconds between flushes |

Both also accept CLI flags (``--coordinator-event-reporting``,
``--coordinator-event-flush-interval``).

## Failure modes

| Event | Effect | Handling |
| --- | --- | --- |
| Coordinator down | Events not delivered | Flush fails → batch dropped, logged; MP server unaffected |
| Coordinator restart | Usage/LRU state lost | Rebuilt from the cache-event stream as events arrive; usage under-reports until it catches up |
| Flush timeout | One batch delayed | Next flush sends new batch; no retry of old batch |
| Usage accounting drift | Quota enforcement imprecise | Self-correcting as new events arrive |

## Scope

Additive: no change to per-server eviction, L2 adapter store/lookup paths, or
the coordinator's membership/health loop. Composes via the ``L2AdapterListener``
interface and the ``http_apis`` auto-discovery — a new router reading
``app.state``, plus the opt-in event listener — with no edits to existing
controllers or adapters beyond passing ``sizes`` through to
``on_l2_keys_stored``.

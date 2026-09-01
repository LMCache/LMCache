# Fleet-Wide Usage Tracking and Eviction

A coordinator-level capability that gives fleet-wide visibility into cache
usage across **both tiers**, and enforces per-``cache_salt`` byte quotas on
**each tier** via LRU eviction. MP servers **report store/lookup events** to
the coordinator; the coordinator aggregates usage, manages quotas, and
periodically selects LRU keys to evict when a tenant exceeds its quota. It is
**opt-in** (gated by ``event_reporting`` in ``CoordinatorConfig``, shared with
the key-directory stream) and **additive** (the existing per-server eviction is
unchanged).

Usage is tracked on two axes, because the tiers are asked different questions:

| Axis | Read as | Answers |
| --- | --- | --- |
| ``cache_salt`` | per tier | How much does this tenant hold? (what quota enforcement compares against, on either tier) |
| ``(instance_id, backend)`` | per tier | How full is this node's DRAM/GDS, or this shared pool? (what a placement or prefetch decision needs, on L1) |

Both axes come from the same event stream and the same delta, so tracking L1
costs no new plumbing on the MP-server side — the L1 events were already being
reported for the key directory and simply discarded by the usage view.

The tenant axis is served over REST (`/quota`, either tier). The capacity axis
is tracked but **not yet exposed**: it is read in-process off
`ctx.usage_manager`, and gets an endpoint when something needs one.

Code: `lmcache/v1/mp_coordinator/controllers/` (coordinator side),
`lmcache/v1/mp_coordinator/http_apis/quota_api.py` (REST endpoints),
`lmcache/v1/mp_coordinator/schemas.py` (wire types),
`lmcache/v1/multiprocess/http_server.py` (MP-server wiring).

## Why

L2 eviction today is **local to each MP server**: the ``IsolatedLRUEvictionPolicy``
tracks only what that server stored and enforces quotas within that scope. With
a shared L2 backend (e.g. S3), multiple servers write to the same storage, but
no single server has a fleet-wide view of total per-tenant usage. The coordinator
centralizes usage accounting and quota enforcement so limits apply to the
aggregate, not per-replica.

L1 needs the same centralization for a different reason. L1 is a node's own
memory, so "how full is L1" is a per-node question no tenant-level total can
answer — and the node that knows the answer is not the one deciding where to
place or prefetch a chunk. The coordinator already receives every L1 store,
eviction, and access for the key directory, so the occupancy view is a rollup
of a stream it was already applying.

L1 quotas are the tenant question asked of the same stream, and no node can
answer it either. Node-local L1 eviction bounds each node's *capacity*: it
keeps one server's memory from filling, and knows nothing about how much of
the fleet's memory one tenant is holding. A tenant spread thinly over ten
nodes can exceed any fleet-wide share while sitting comfortably under every
node's local watermark. The coordinator is the only place that total exists,
so it is the only place that budget can be enforced — and once a victim is
chosen, the delete has to go back to the node that holds it, because L1 bytes
are that process's memory and nobody else can free them.

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
                             ├─ KeyDirectory.consume:       placements
                             ├─ CacheUsageManager.consume:  per-tier byte view
                             │    (by salt and by instance/backend)
                             ├─ FleetEvictionController.consume:    (l2 entries)
                             │    ├─ QuotaManager: per-salt L2 byte limits
                             │    └─ IsolatedLRU:  L2 eviction order
                             └─ FleetEvictionController.consume:   (both tiers)
                                  ├─ QuotaManager: per-salt L1 byte limits
                                  └─ IsolatedLRU:  L1 eviction order

  Coordinator health loop (every health_check_interval)
        │
        ▼
  evict_stale() reaps a timed-out instance
    └─ EventGate.drop_instance → broadcaster.fence_instance
         ├─ KeyDirectory:             its L1 placements
         ├─ CacheUsageManager:        its L1 bytes
         └─ FleetEvictionController:  its L1 LRU entries

  Coordinator background loop (every eviction_check_interval, default 5s)
    one task per eviction controller, created by its own Controller.run
        │
        ▼
  execute_evictions():
    for each tracked salt on this controller's tier:
      limit = quota (default 0 → evict all)
      if usage ≥ watermark·limit → select LRU keys,
        fire-and-forget DELETE /cache/objects
          L2 → any registered member (they share the storage)
          L1 → every node the key directory says holds the key
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

An eviction controller owns the whole quota → usage → evict loop for
**one tier**: the budgets (target), what is spent against them
(observation), and the LRU that picks victims when a salt is over
(action). Holding all three is what makes it a controller rather than a
bag of state — so they sit behind one collaborator (with `.quota` for
the `/quota` endpoints) rather than as peers the app has to wire
together in the right order. The **usage view is the exception**: it is
a consumer in its own right (`ctx.views.get(CacheUsageManager)`),
because it spans both tiers while each controller enforces against one
half — owning it would mean one controller owned bytes it never reads.
Both take it as a dependency instead. The node-local counterpart one
scope down is `distributed/storage_controllers/eviction_controller.py`.

There is **one controller for both tiers**, because the identical half
is so much bigger than the differing half. The tier is a parameter, not a
subclass:

| | `l2` | `l1` |
| --- | --- | --- |
| Selection | shared | shared |
| Quota section | `quotas` | `l1_quotas` |
| LRU section | `lru_order` | `l1_lru_order` |
| Delete goes to | any registered member | every node the directory says holds the key |
| `fence_instance` | no-op — L2 outlives the reporter | drops the keys only that node held |
| Pins | `POST /cache/pins` holds keys back | none; the node skips locked objects instead |

The two registries are **independent**: a salt can be budgeted on one
tier, both, or neither, and the numbers never mix — they govern
different bytes. A key resident in both tiers spends against both.

### CacheUsageManager (`views/usage_manager.py`)

The byte totals are a **derived view of the global key directory**,
constructed and maintained in this manager rather than inside the
directory itself: it consumes the same admitted batches the directory
sees. Accounting mirrors the directory's placement identity,
``(tier, key, owner, backend)`` — re-storing a placement delta-adjusts,
two private copies of one key count twice, N reporters of one shared
pool count once.

It is registered on the broadcaster **before** the eviction controller,
because the controller's delete handling reads a key's remaining size
for the batch it is currently consuming.

**Every read is tier-explicit.** A key resident in both tiers holds
bytes in both, so a tier-blind total would double-count it; there is no
cross-tier read, and the ``all`` tier is rejected at the API.

| Read | Axis |
| --- | --- |
| ``get_salt_bytes(tier, salt)`` / ``get_bytes_by_salt(tier)`` | tenant |
| ``get_total_bytes(tier)`` | tenant (aggregate) |
| ``get_key_bytes(tier, key)`` | per key — eviction sizing |
| ``get_bytes_by_instance(tier)`` | capacity — no REST surface yet |

#### What removes bytes, per tier

This is the one place the tiers genuinely differ, and getting it wrong
leaks a tier's bytes forever:

| Tier | Bytes live in | Removed by |
| --- | --- | --- |
| L2 | storage the fleet shares | ``DELETE`` events only — a reporter restart must **not** zero them |
| L1 | the reporter's own process memory | ``DELETE`` events, **plus** the reporter restarting or leaving |

Both L1 cases arrive as ``fence_instance``, which the ingest gate raises
on an incarnation bump (restart) and on ``drop_instance`` (departure) —
see [ingest.md](ingest.md). The view keeps an instance → L1-placement
index so a fence costs that instance's placements rather than a full
scan; ``FleetEvictionController`` keeps its own copy of that index for
the same reason, and consults the view — already fenced, since views run
first — to tell a key's last copy from one another node still holds.
``FleetEvictionController`` no-ops the same hook: its LRU is L2-only,
and L2 outlives the reporter.

Shared pools are owned by ``""``, not by any reporter, so a fence never
touches them — the pool outlives any one member. That holds per placement,
not per tier: only L1 placements with a real owner are indexed for fencing,
so a shared L1 pool is spared the same way a shared L2 one is.

### QuotaManager (reused from ``lmcache.v1.distributed.quota_manager``)

Thread-safe in-memory quota registry (``dict[str, int]`` + lock). CRUD via
``set_quota``, ``get_limit_bytes``, ``delete_quota``, ``list_quotas``.
Quotas are set in GiB at the API and stored as bytes internally.

The coordinator holds **one registry per enforced tier**, each owned by
that tier's controller, rather than one registry with a tier dimension.
A tier's budget is only ever read by the controller enforcing it, so a
shared table would add a key nothing joins on — and the same class is
reused by the MP server's node-local enforcement, which knows nothing of
tiers. The registries differ only in the durable section they name
(``section_name``, ``quotas`` vs ``l1_quotas``), since two sections of
one document cannot share a name.

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

The quota → usage → evict machine, once, for both tiers. It owns a quota
registry and a coordinator-side ``IsolatedLRUEvictionPolicy`` **per
tier**, keyed by the canonical ``ObjectKey`` (rebuilt from the wire
``EncodedObjectKey``), so a key resident in both is budgeted twice and
ordered separately. Per-salt byte accounting lives in
``CacheUsageManager``; the LRUs only track order.

Reads that name a tier take it as an argument — ``quota(tier)``,
``policy(tier)``, ``compute_eviction_plan(tier)``,
``plan_dispatches(tier, plan)`` — and ``execute_evictions`` sweeps every
tier in ``ENFORCED_TIERS``, returning a plan per tier.

- ``consume(batch)`` — the `CacheEventConsumer` hook. Maps this tier's
  entries onto the LRU (`STORE` registers, `ACCESS` touches, `DELETE`
  drops); other tiers are ignored. A delete drops the key from the LRU
  only once its **last** placement on this tier is gone: usage is per
  placement, so while another copy still holds bytes the key must stay
  evictable — otherwise those bytes could exceed quota with nothing for
  the planner to select. The usage view consuming the same batch first
  is what makes that read of the post-batch size correct, which
  registration order in ``create_app`` guarantees.
- ``on_store(key)`` — register the key in the LRU
  (``policy.on_keys_created``). The paired byte increment happens in the
  usage view consuming the same event.
- ``on_lookup(key)`` — touch (``policy.on_keys_touched``, move to MRU end).
- ``on_remove(key)`` — drop from the LRU (``policy.on_keys_removed``); the paired
  byte decrement happens in the usage view consuming the same event.
- ``is_evictable(key)`` — whether a key may be selected. ``True`` for
  everything tracked unless a tier overrides it.
- ``compute_eviction_plan() -> dict[str, list[ObjectKey]]`` — **pure**: for each
  tracked salt, fire when ``usage ≥ trigger_watermark · quota`` (quota 0 ⇒ evict
  all), selecting ``eviction_ratio`` of the salt's LRU keys via
  ``policy.get_eviction_actions``. Salts without an explicit quota use the
  registry's default limit; while it is unset (``None``) they are skipped
  entirely — see QuotaManager above. No network, no mutation.
- ``plan_dispatches(plan)`` — **abstract**: routes the plan's victims to
  the servers that can delete them. The one thing the tiers genuinely
  disagree on.
- ``fence_instance(id)`` — **abstract**: what a reporter's restart or
  departure voids. No sensible default, so each tier states its position.
- ``run(runtime)`` — the ``Controller`` lifetime hook, an async context
  manager. Entering creates the loop task; ``EVICTION_CHECK_INTERVAL = 0``
  creates none. Exiting cancels the loop, then drains the dispatches
  already sent so a ``DELETE`` the last sweep launched still arrives. The
  app enters every controller's ``run`` and names none of them, so a new
  tier's loop starts by existing.
- ``execute_evictions(http_client)`` — one pass: computes the plan,
  routes it through ``plan_dispatches``, and **fire-and-forgets**
  ``DELETE /cache/objects`` to each target; on confirmed deletion ``on_remove``
  drops the keys from tracking. Each target's keys are split into chunks of at
  most ``MAX_DELETE_BATCH`` (imported from the MP server's ``object_service``)
  and each chunk is dispatched as its own request, because the endpoint rejects
  any single delete over that cap with HTTP 400 — a full-salt eviction (quota
  dropped to 0) routinely exceeds it.

### What differs by tier

Only two answers, and both follow from where the bytes live.

**Dispatch** (``plan_dispatches``). L2 bytes sit on storage the fleet
shares, so one uniformly chosen registered member evicts for everyone.
L1 bytes are one node's memory, so each victim is resolved through
``KeyDirectory.lookup`` and grouped by holding ``instance_id`` — one
request per node, and a key with copies on several nodes is deleted on
all of them, because every copy spends against the tenant's budget.
Victims the directory has no L1 placement for, and holders no longer
registered, are skipped and logged; the next delete event repairs the
disagreement.

**Fencing** (``fence_instance``). A restart voids L1 only, so the
controller drops the keys that node was the **last** holder of and leaves
L2 alone. It keeps its own reporter → ``(key, backend)`` index for this
(``_L1PlacementOwners``); the usage view has already subtracted the
fenced placements by the time the controller is fenced, so a remaining
byte count means a remaining copy elsewhere. Without it, a restarted
node's keys would linger in the LRU with no bytes behind them, and every
later plan would dispatch deletes that free nothing — a loop that never
converges. The index is **checkpointed** for the same reason: a restored
ordering without it would hit exactly that for any node restarting after
the coordinator did.

Two smaller asymmetries fall out of the same fact. **Pins** (``POST
/cache/pins``) hold L2 keys back from selection; L1 has no pin table
because the node refuses a locked object at delete time, which no
coordinator-side table could anticipate. And deletes go out
**non-force**, so a node skips objects under a read or write lock and
reports them as ``skipped`` — those stay in the LRU and are retried next
cycle, where force-dropping one would corrupt an in-flight transfer.


## REST endpoints (`quota_api.py`)

Every endpoint takes a ``tier`` (``l1`` or ``l2``, default ``l2``) — a query
parameter on reads and deletes, a body field on writes — and is **wholly
scoped to it**, quota fields included.

| Method | Path | Description |
| --- | --- | --- |
| ``PUT`` | ``/quota/config`` | Set the tier's default limit for unquota'd salts (``null`` ⇒ exempt; ``0`` arms allowlist eviction) |
| ``GET`` | ``/quota/config`` | Read the tier's default limit |
| ``DELETE`` | ``/quota/{cache_salt}`` | Remove the salt's quota on that tier |
| ``PUT`` | ``/quota/{cache_salt}`` | Set the salt's quota on that tier (GiB) |
| ``GET`` | ``/quota/{cache_salt}`` | Quota + usage for one salt |
| ``GET`` | ``/quota`` | Quota + usage for all salts |

Each tier has its own registry, so a salt budgeted on L2 alone reads as
``quota_exists=False`` on L1 rather than borrowing the L2 budget: the two
govern different bytes, and a client dividing one by the other would get a
meaningless ratio. Setting both is the normal case for a tenant you want
bounded in memory *and* in shared storage; the numbers are unrelated and
neither implies the other.

``all`` is rejected everywhere. On a read it would double-count a key resident
in both tiers; on a write it would have to pick one of two budgets. Responses
echo the ``tier`` they describe, so a reply is unambiguous even when the
request relied on the default.

## MP-server event emission (`cache_events.py`)

The L2 adapters publish key events on the observability bus; one
``CacheEventSubscriber`` per MP server turns them into ordered batches
delivered to ``POST /events`` (flushes paced by
``event_flush_interval``, default 1s); failures drop the batch (the
resulting seq gap flags replay). See
[cache_events.md](cache_events.md) for the subscriber/transport design.
``l2.keys.deleted`` becomes a ``DELETE`` event so the coordinator can
drop the key from its usage accounting and LRU tracking. The L1 side needs
no new emission: ``l1.write_finished`` / ``l1.keys.evicted`` /
``l1.keys.accessed`` already become ``STORE`` / ``DELETE`` / ``ACCESS``
batches (one per ``L1BackendType``, carrying per-object sizes) for the key
directory.

## Configuration

### Coordinator side (`MPCoordinatorConfig`)

| Field | Default | Description |
| --- | --- | --- |
| ``eviction_check_interval`` | ``5.0`` | Seconds between eviction cycles (0 disables) |
| ``eviction_ratio`` | ``0.2`` | Fraction of a salt's LRU keys (by count) to evict per cycle |
| ``trigger_watermark`` | ``1.0`` | Eviction fires when usage reaches this fraction of the quota |

All three apply to **both** tiers' loops. There is nothing yet that wants L1
paced differently from L2, and per-tier knobs are cheap to add once something
does; what is *not* shared is the part that actually differs per deployment —
the quotas themselves, which are set per tier over REST.

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
| L1 victim locked on the node | Those bytes stay over quota one cycle | Node reports it as ``skipped``; the key stays in the LRU and is retried |
| L1 victim's holder gone | Nothing to delete | Skipped and logged; the fence that removed it also drops it from the LRU |

## Scope

Additive: no change to per-server eviction, L2 adapter store/lookup paths, or
the coordinator's membership/health loop. Composes via the ``L2AdapterListener``
interface and the ``http_apis`` auto-discovery — a new router reading
``app.state``, plus the opt-in event listener — with no edits to existing
controllers or adapters beyond passing ``sizes`` through to
``on_l2_keys_stored``.

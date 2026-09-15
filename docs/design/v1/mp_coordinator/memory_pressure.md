# Fleet Memory Usage

How full is each MP server? The coordinator already knows how many bytes a
tier holds; it does not know how many that tier *can* hold. This adds the
denominator and joins the two.

`used / capacity`, per compartment, read-only:

    GET /instances/usage                  the whole fleet
    GET /instances/{instance_id}/usage    one server

Code: `server_config.py` (capacity registry),
`http_apis/instances_usage_api.py` (endpoints),
`controllers/usage_manager.py` (usage, pre-existing),
`cache_events.py` + `distributed/storage_manager.py` (MP-server side).

## Workflow

A **compartment** is one thing that owns bytes: the L1 pool of one backing
medium, or one L2 adapter. Identity is `(tier, backend)` — the same axis
cache events already tag placements with, so the two halves join without
translation.

**Declaring capacity** (MP server → coordinator):

1. Something changes: the server registers, or an L2 adapter is added,
   removed, or reconfigured.
2. `StorageManager.publish_capacity()` builds the topology —
   `get_configured_capacity_bytes()` for L1 per medium, each adapter's
   `get_usage().total_capacity_bytes` for L2 — and publishes
   `SM_CAPACITY_CHANGED` on the observability bus.
3. `CacheEventSubscriber` turns it into one `config` batch per compartment,
   all stamped with one `capacity_revision`, and flushes them with whatever
   placement batches are pending.
4. `EventGate` admits them like any other batch: incarnation fencing, seq
   dedup, gap detection.
5. `ServerConfigRegistry.consume()` groups them by
   `(incarnation, capacity_revision)`.

**Reporting usage** (unchanged, pre-existing): `l1.*` / `l2.*` events →
`EventGate` → `CacheUsageManager`, rolled up per `(instance, tier, backend)`.

**Reading**: the endpoint calls `get_bytes_by_instance()` for the numerator
and `server_config.get_all()` for the denominator, and divides.

```
MP server                                Coordinator
─────────                                ───────────
publish_capacity()
  L1: per backing medium
  L2: per adapter + shared flag
       │
       │ subscriber: one config batch per compartment
       ▼
  POST /events ──────▶ EventGate ──▶ ServerConfigRegistry   capacity
                                                    │           │
l1.* / l2.* on the bus                              │           │
       │                                            │           │
       ▼                                            │           │
  POST /events ──────▶ EventGate ──▶ CacheUsageManager     usage │
                                                    │           │
                                                    ▼           ▼
                                          GET /instances/usage
```

## Why capacity rides the same stream

One path. Registration used to carry it too, and two sources for one fact
could disagree — worse, registration and event reporting are separately
configurable, so a server with reporting off used to declare capacity whose
usage never arrived, making every compartment read a confident `0.0`.

Registration still *triggers* a declaration, through `keep_registered`'s
`on_registered` hook. The coordinator holds declarations in memory only, so a
restarted one has forgotten them, and re-registration (via the heartbeat
`404`) is the only signal a server gets that its declaration may be gone.

`config` is its own event type because capacity is configuration: it changes
at boot and at reconfiguration, not per operation. It carries no `ObjectKey`,
so `entries` is required empty — which is exactly what makes it a no-op for
the other consumers, all of which are entries-driven and need no change.

## Declarations, not deltas

A declaration is always the whole topology. That is what makes a lossy
channel acceptable: a dropped batch is repaired by the next declaration,
where a dropped byte delta would be permanent. The emitter also holds an
unsent declaration across a publish failure.

`(incarnation, capacity_revision)` groups batches: a newer stamp starts a
fresh set, an equal one extends it, an older one is dropped. **This is what
retires a compartment** — a declaration that omits it never re-adds it, so a
deleted L2 adapter stops being reported. Incarnation fencing and seq dedup
come from the gate; nothing here duplicates them.

The revision is assigned by the subscriber, beside `seq` and for the same
reason: the bus drains on one thread, so neither counter needs a lock and a
number cannot come apart from the topology it labels.

Two gotchas:

- `config` batches share the seq space with placements, so one counter covers
  both and a reused seq is dropped as a duplicate.
- The registry never sees a whole list, so it cannot reject a compartment
  declared twice in one declaration. It upserts; the last batch wins.
  `_build_capacities` emits one entry per medium and adapter, so a correct
  producer cannot do it.

## Capacity is the configured size, not the live heap

| Manager | `get_memory_usage()[1]` | `get_configured_capacity_bytes()` |
| --- | --- | --- |
| `L1MemoryManager` (default, lazy) | grown heap | `{dram: size_in_bytes}` |
| `GDSL1MemoryManager` | configured slab | `{gds: size_in_bytes}` |
| `DevDaxL1MemoryManager` | live arenas | `{devdax: …}` + `{dram: …}` if hybrid |

The lazy allocator's total is the heap it has grown so far, so a freshly
booted server would report itself nearly full and appear to drain as the pool
warms. The configured size is stable from boot.

L1 is declared **per medium** because one tier can span two: a hybrid
Device-DAX tier backs objects with both `devdax` and `dram`, and each
placement is tagged accordingly. One total would leave two compartments of
usage sharing one denominator.

`report_status()` exposes the same number as `memory_configured_bytes`, so
the status dict and this API cannot drift.

## Unknown is a value

`capacity_bytes == 0` means no declared limit, and this is the **common**
case: `fs`, `mooncake`, `p2p`, and `sagemaker` have no capacity knob at all.

So `usage_ratio` is `null` — not `0.0`, not `-1.0`. A number there reads as
real occupancy, and a fleet view that treats capacity-less backends as empty
reports "healthy" regardless of what is happening.

Ratios above `1.0` are **not clamped**: a tier holding more than its declared
cap means the declaration is wrong, and hiding that hides a misconfiguration.

## Shared pools are counted once

An adapter with `shared=True` is storage several instances mount — one S3
bucket, one CXL region. Summing across the N servers that report it would
overstate by N, and the result looks plausible, which is worse than an
obvious error.

Shared placements are keyed under an empty owner, the same convention the key
directory uses, and surface under `shared_modules` rather than inside any
instance. Capacity is resolved across every declaring server; when they
disagree the pool reads as undeclared, since preferring one value would make
the answer depend on registration order.

## Lifecycle

- **Registration** records membership. Capacity follows on the stream, so a
  just-registered server reads `declared_capacity: false` until its first
  declaration lands.
- **Fencing** discards the instance's **L1** bytes — L1 lives in the
  reporting process and dies with it, while L2 outlives its reporter and
  leaves only through `DELETE`. Three paths reach it: a higher incarnation on
  the stream (restart), the stale-eviction loop (heartbeat timeout), and
  `DELETE /instances/{id}` (clean shutdown).
- **Deregistration** also drops the declaration, which would otherwise grow
  without bound across a churning fleet. Surviving L2 bytes are still
  reported, without a ratio.

An instance appears in the fleet view when it is registered, holds bytes, *or*
has declared capacity — so a departed server whose L2 placements survive is
not silently dropped.

The registry is a `DurableComponent` (section `server_config`, type
`CHECKPOINT`) like every other event consumer. A capture carries each
declaration's stamp with its modules; without it a restored registry would
accept a straggler from before the capture and regress what it just loaded.

## Scope

Read-only: never evicts, throttles, or pushes. No derived pressure level, no
smoothing, no ranking — while most L2 backends declare no capacity, a
normalized `LOW`/`HIGH`/`CRITICAL` score would be confidently wrong on the
majority of deployments. Bytes, capacity where known, and an explicit unknown
are what the data supports.

Follow-ups: `lmcache describe` should prefer `memory_configured_bytes` over
`memory_total_bytes` (it prints the grown heap as "L1 capacity" today);
forwarding `L1_ALLOCATION_FAILED` would give a measured pressure signal
rather than an inferred ratio; a derived level becomes reasonable once
capacity declaration is widespread.

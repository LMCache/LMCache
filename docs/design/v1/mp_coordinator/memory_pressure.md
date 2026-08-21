# Fleet Memory Pressure

A coordinator-level read-only view of how full each MP server's memory
compartments are. It joins two things the coordinator holds separately: byte
usage derived from the admitted cache-event stream, and capacity declared by
each server on that same stream. Neither is a pressure reading on its own.

Code: `lmcache/v1/mp_coordinator/controllers/usage_manager.py` (usage view),
`lmcache/v1/mp_coordinator/server_config.py` (capacity registry),
`lmcache/v1/mp_coordinator/http_apis/instances_usage_api.py` (REST endpoints),
`lmcache/v1/distributed/storage_manager.py` (MP-server capacity source),
`lmcache/v1/mp_coordinator/cache_events.py` (MP-server declaration).

## Why

The coordinator can already say how many bytes a tier holds. It cannot say
whether that is a lot, because nothing tells it how many bytes the tier *can*
hold. Pressure is `used / capacity`, and the cache-event stream carries only
the numerator.

The usage half needs no new tracking. `CacheUsageManager` already rolls the
admitted cache-event stream up per `(instance_id, backend)` for both tiers —
`get_bytes_by_instance(tier)` — which is exactly the axis a pressure reading
needs. This capability adds only the denominator.

## How capacity reaches the coordinator

Capacity is a **`config` cache event**, so it travels the one ingest path
every other event does and the registry is a `CacheEventConsumer` like the
key directory and the usage manager. One declaration is one `config` batch
per compartment, all sharing a `capacity_revision`.

`StorageManager` publishes `SM_CAPACITY_CHANGED` on the observability bus and
the cache-event subscriber ships it on its next flush. It fires:

- **After every successful registration**, via the `on_registered` hook
  `keep_registered` calls, wired to `publish_capacity()`. This covers the
  first registration — without which a server that never reconfigures would
  never declare at all — and, critically, every *re*-registration. The
  coordinator holds declarations in memory only, so a restarted one has
  forgotten them; re-registration is the single event that tells a server its
  declaration may be gone. Nothing else would resend it, and the fleet view
  would sit at `declared_capacity: false` indefinitely.
- **On every topology change** — adapter added, removed, or reconfigured.

Registration deliberately carries no capacity in its *body*. One path means
the coordinator cannot hold a declaration that disagrees with the event
stream, and it closes a real hole: registration and event reporting are
separately configurable, so a server registering without event reporting used
to declare capacity whose usage never arrived, making every compartment read
as a confident `0.0` instead of unknown.

A declaration is always the **whole topology**, never a delta. That is what
makes the lossy channel acceptable: a dropped batch is repaired by the next
declaration, where a dropped byte delta would be permanent. The emitter also
keeps an unsent declaration across a publish failure.

`(incarnation, capacity_revision)` groups batches into declarations. A newer
stamp starts a fresh set, an equal one extends it, an older one is dropped.

The revision is assigned by the **cache-event subscriber**, not by
`StorageManager`. That is deliberate and it is why no lock is involved: the
bus drains on one thread, so the subscriber's revision counter needs no more
protection than its `seq` counter already has, and the number cannot come
apart from the topology it labels. `StorageManager`'s publishers are
concurrent -- registration publishes from the event loop while a worker may
be adding an adapter -- so a counter there would need a lock and could still
tag an older topology with a newer number. Coalescing falls out for free: two
publishes merged into one flush share one revision instead of burning two.
**That is what retires a compartment**: a declaration omitting it simply
never re-adds it, which is how a deleted L2 adapter stops being reported.
Incarnation fencing and seq dedup come from the gate, so nothing here
duplicates them.

Two consequences worth knowing:

- `config` batches share the seq space with placements, so the emitter's one
  counter covers both and a reused seq is dropped as a duplicate.
- The registry cannot reject a compartment declared twice inside one
  declaration — it never sees the whole list — so it upserts and the last
  batch wins. `_build_capacities` derives one entry per medium and adapter,
  so a correct producer cannot do it.

The batch envelope carries a declaration on `tier`, `backend`, `shared`,
plus `capacity_bytes` and `capacity_revision`; `entries` is required empty.
That empty-entries invariant is what makes the other consumers safe: the key
directory, usage manager, and eviction controller are all entries-driven, so
a `config` batch is a no-op for each without any of them knowing the type
exists.

## Why capacity is its own event type

Capacity is configuration: it changes when a server boots and when it is
reconfigured, not once per cache operation. Reusing `store` would mean
inventing a placement for something that has no key; emitting it per
operation would republish an unchanging number thousands of times a second.
A distinct type emitted on change keeps the volume proportional to what
actually varies and leaves the placement types untouched.

The cost is two fields on `CacheEventBatch` that only `config` reads, so
every placement batch carries them as zeros on the wire.

## Architecture

```
MP server                                   Coordinator
─────────                                   ───────────
StorageManager.publish_capacity()
  L1: get_configured_capacity_bytes(l1_config)
      → one entry per backing medium
  L2: per adapter, total_capacity_bytes
      + shared flag from its config
        │
        │  subscriber: ModuleMemoryCapacity → config CacheEventBatch
        ▼
  POST /events ─────────▶ EventGate ─▶  ServerConfigRegistry.consume()
   (config batches)                       one set per (incarnation, revision)
                                                    │
L2 adapter / L1 manager publish                     │  capacity
  l1.* / l2.* on the event bus                      │
        │                                           │
        ▼                                           │
  CacheEventSubscriber (cache_events.md)            │
        │                                           │
        ▼                                           │
  POST /events ─────────────────────▶  EventGate (ingest.md)                │
                                          └─ CacheEventBroadcaster          │
                                             ├─ KeyDirectory                │
                                             ├─ FleetEvictionController     │
                                             └─ CacheUsageManager  ─────────┤
                                                  (instance, tier, backend) │
                                                                    usage   │
                                                                            ▼
                                              GET /instances/usage  joins both
                                              GET /instances/{instance_id}/usage
```

## The compartment axis

A "module" is a compartment that owns bytes: the L1 pool of one backing
medium, or one L2 adapter. Identity is `(tier, backend)` — the same axis
cache events tag placements with, so a declaration and a usage total join
without translation.

L1 capacity is reported **per medium** because one tier can span several: a
hybrid Device-DAX tier backs objects with both `devdax` and `dram`, and
`L1ObjectMeta.backend` tags each placement accordingly. Flattening it to one
total would leave two compartments of usage sharing one denominator.

## Capacity is the configured size, not the live heap

`get_configured_capacity_bytes()` is the denominator rather than
`get_memory_usage()[1]`. On the default lazy allocator that total is the
**currently grown heap**: it starts small and grows on demand, so a freshly
booted server would report itself nearly full and then appear to drain as the
pool warms. The configured size is stable from boot and is the only sound
denominator.

| Manager | `get_memory_usage()[1]` | `get_configured_capacity_bytes()` |
| --- | --- | --- |
| `L1MemoryManager` (default, lazy) | grown heap | `{dram: size_in_bytes}` |
| `GDSL1MemoryManager` | configured slab | `{gds: size_in_bytes}` |
| `DevDaxL1MemoryManager` | live active arenas | `{devdax: …}` + `{dram: …}` when hybrid |

`L1Manager.report_status()` exposes the same value as
`memory_configured_bytes` so the status dict and the capacity API cannot
drift. `memory_total_bytes` keeps its existing meaning for existing consumers.

## Shared pools are counted once

An adapter with `shared=True` is storage several instances mount — one S3
bucket, one CXL region. Its bytes and its capacity are fleet-scoped. Summing
them across the N servers that report them would overstate both by N, and the
result looks plausible, which is worse than an obvious error.

The usage tracker follows the same convention the key directory and the
per-salt view use: shared placements are keyed under an empty owner
(`SHARED_OWNER`), so they are counted once and attributed to no instance.
`GET /instances/usage` reports them under `shared_modules`, never inside an
instance.

Capacity for a shared pool is resolved across every server that declares it.
Declarations should agree; when they do not, the pool is reported as
undeclared rather than picking one, since preferring a value would make the
reading depend on registration order.

## Unknown is a value

`capacity_bytes == 0` means the server declares no limit. This is the
**common** case, not an edge case: `fs`, `mooncake`, `p2p`, and `sagemaker`
return `0` unconditionally with no configuration knob at all.

So `usage_ratio` is `null` — not `0.0`, not `-1.0` — whenever there is no
capacity to divide by. A number there would be read as a real occupancy, and
a fleet view that treats capacity-less backends as empty reports "healthy"
regardless of what is happening.

Ratios above `1.0` are **not clamped**. A tier holding more than its declared
cap means the declaration disagrees with what the tier admitted, and hiding
that would hide a misconfiguration.

## Lifecycle

- **Registration** records membership only. Capacity follows on the event
  stream, so a just-registered server reads as `declared_capacity: false`
  until its first report lands. A report replaces the set wholesale: a
  server that dropped an adapter must not keep the old compartment's
  capacity.
- **Fencing** discards the instance's **L1** bytes. L1 lives in the reporting
  process and dies with it; L2 bytes outlive the reporter and leave only
  through `DELETE`. Three paths reach it: a higher incarnation arriving on
  the stream (restart), the stale-eviction loop (heartbeat timeout), and
  `DELETE /instances/{id}` (clean shutdown). That last one used to be
  missing, so a server that left cleanly kept reporting L1 bytes forever —
  the eviction loop could never reach it, having already been removed from
  the registry.
- **Deregistration** also drops the capacity declaration. A departed server's
  caps describe a process that no longer exists, and keeping them would grow
  without bound across a churning fleet. Its surviving L2 bytes are still
  reported, without a ratio.

The registry is also a `DurableComponent`, like every other event consumer.
Its section is `server_config` and its type is `CHECKPOINT`: declarations are
derived from the event stream, and every server redeclares on registration,
so they are rebuildable. A capture carries each declaration's
`(incarnation, revision)` stamp with it -- without that, a restored registry
would start from scratch and accept a straggler from before the capture,
regressing the topology it had just loaded.

An instance appears in `GET /instances/usage` when it is registered, when it
holds bytes, or when it has declared capacity — so a deregistered server whose
L2 placements survive is not silently dropped.

## Scope

Read-only. This never evicts, throttles, or pushes. There is no derived
pressure level, no smoothing, no trend, and no ranking: while most L2
backends have no declared capacity, a normalized `LOW`/`HIGH`/`CRITICAL`
score would be confidently wrong on the majority of deployments. Bytes,
capacity-where-known, and an explicit unknown are what the data supports
today.

Named follow-ups: `lmcache describe` should prefer `memory_configured_bytes`
over `memory_total_bytes` (it currently prints the grown heap as "L1
capacity"); forwarding `L1_ALLOCATION_FAILED` to the coordinator would give a
directly-measured pressure signal rather than an inferred ratio; a derived
level becomes reasonable once capacity declaration is widespread.

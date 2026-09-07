# Cache-event ingest

Modules: `lmcache/v1/mp_coordinator/ingest/`
 - `event_gate.py` — `EventGate`: admission (fencing, dedup, gap detection)
 - `event_broadcaster.py` — `CacheEventBroadcaster` + the `CacheEventConsumer` protocol
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
HTTP surface: `http_apis/events_api.py` (`POST /events`)

Everything the coordinator knows about the fleet's cache contents is
built from one event stream. This layer is where that stream enters:
it decides **what** is admitted (the gate) and **who** sees it (the
broadcaster). Neither holds cache state — the consumers do.

```
source                    ingest layer                     consumers
─────────────────────────────────────────────────────────────────────────
POST /events ──────────▶ EventGate.ingest ──▶ CacheEventBroadcaster
  (live emitter stream)    fence / dedup /      .broadcast(batch) ────▶ KeyDirectory
                           gap detect           .fence_instance(id) ──▶ FleetEvictionController
```

## Why a gate separate from the directory

The two jobs have different lifetimes and different owners. Stream
admission is a property of the **emitter** — one cursor per emitter,
valid whether or not any state is kept. Placements, usage, and the LRU
are properties of the **cache**. Folding admission into the key
directory made the directory the mandatory first consumer of every
event: adding a second consumer meant routing through it, and asking
"was this batch a replay?" meant asking the placement store. With the
gate separate, the directory is just another `CacheEventConsumer`
(registered first, by convention, as the source of truth), and a new
consumer is a `register_consumer` call in `create_app`.

## Admission (`EventGate.ingest`)

One batch = `(instance_id, incarnation, seq, event_type, tier, backend,
entries[], ts)`. The gate enforces, in order:

| mechanism | rule | why |
| --- | --- | --- |
| Incarnation fencing | `incarnation <` current → drop batch (`STALE_INCARNATION`). `incarnation >` current → `fence_instance(id)` on every consumer, then start a fresh cursor. | A restart empties the reporter's *memory* — its L1 placements must not survive. L2 bytes persist on disk across restarts, so L2 is deliberately not fenced (consumers that track L2 only no-op the hook). |
| Seq dedup | `seq <=` last admitted (same incarnation) → drop batch (`DUPLICATE`). | Replays (retry, event-bus redelivery) must be idempotent. |
| Gap detection | `seq >` last admitted `+ 1` → set the emitter's `gap_detected` flag, admit anyway. | Events may be lost; the flag marks the emitter's slice as stale until the stream is replayed (durable-transport retention). Consumer application is idempotent, so admitting past a gap is safe. |

Per-instance FIFO by `seq` is the **only** ordering the design needs:
each instance is the sole writer of its own facts, so there is no
global order and no cross-instance arbitration. `instance_id` is really
the *emitter stream id* — a shared medium's controller sends under its
own stable id and gets an ordinary deduplicated, fenced stream with no
special-casing (see [key_directory.md](key_directory.md), Shared pools).

The gate's lock is held across the fan-out, so an emitter's batches
reach the consumers in admission order. Consumers must therefore not
call back into the gate.

`drop_instance(id)` is the same fence without a batch — for
deregistration and heartbeat-timeout eviction. It also forgets the
cursor, so a reconnect starts fresh at any incarnation. (Wiring it to
the registry is still a follow-up; the method exists and is tested.)

## Sources

`ingest` is the only door, and every source must carry a stream:
`(instance_id, incarnation, seq)`. Today that is `POST /events`, fed by
the MP-server `CacheEventSubscriber` (see
[cache_events.md](cache_events.md)); a durable message-queue consumer
would enter the same way, unchanged.

A source that is a *scan* of current contents rather than a stream —
the startup L2 resync that used to paginate `GET /cache/objects` — has
no stream position, and admitting it needs a second, cursor-free door.
That door is deliberately absent: one entry point means there is
exactly one place where ordering and fencing are decided. Reintroducing
a scan source means reintroducing `reconcile` (and answering what its
batches do to the cursor), so weigh that against event replay from a
durable transport, which needs no new door at all.

## Fan-out (`CacheEventBroadcaster`)

Consumers implement two hooks:

- `consume(batch)` — apply one admitted batch. Called in admission
  order; each event arrives at most once per delivery attempt. Skipping
  irrelevant tiers and event types is the consumer's own job.
- `fence_instance(instance_id)` — discard what that instance held in
  its own memory. **L1 only**: `KeyDirectory` drops the L1 placements
  the instance reported (its per-instance reverse index makes this
  proportional to that instance's keys, not a full scan);
  `FleetEvictionController` no-ops, because the L2 bytes it accounts
  outlive the process and leave only via `DELETE`.

Registration order is invocation order. Today: the key directory
(placements and token bindings, the source of truth), then the eviction
controller (per-salt usage and the LRU). The two are independent — the
controller's own read-after-write ordering is internal to it (see
[l2_usage_and_eviction.md](l2_usage_and_eviction.md)), not a property of
registration order.

## Where the state is

| question | asked of |
| --- | --- |
| Is this batch a replay? What incarnation is this emitter on? Did we lose events? | `EventGate.stats()` |
| Where does this key live? What tokens does this chunk hold? | `KeyDirectory` |
| How many bytes is this salt using? What should be evicted? | `FleetEvictionController` |

`EventGate.stats()` has **no HTTP endpoint yet** — `GET /directory/stats`
deliberately reports directory contents only. So `gap_detected` is
currently invisible to operators; exposing it is part of the replay
follow-up below.

## Deliberately out of scope (follow-ups)

- **Replay integration**: exposing `gap_detected` over HTTP, then acting
  on it by replaying the emitter's stream from a durable transport's
  retention.
- **Registry integration**: calling `EventGate.drop_instance` from
  deregistration / heartbeat-timeout eviction.
- **Allocation generations** for shared pools (deterministic
  cross-reporter conflict resolution).

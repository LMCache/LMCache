# Cache-event ingest

Modules: `lmcache/v1/mp_coordinator/ingest/`
 - `event_source.py` — source lifecycle/status contract
 - `http_event_source.py` — non-durable `POST /events` push source
 - `kafka_event_source.py` — durable Kafka pull source (poll thread)
 - `stream_position.py` — `StreamPosition`: the checkpoint's own record of
   how far into the Kafka stream it reaches
 - `event_gate.py` — `EventGate`: admission (fencing, dedup, gap detection)
 - `event_broadcaster.py` — `CacheEventBroadcaster` + the `CacheEventConsumer` protocol
 - `readiness.py` — `IngestReadiness`: source lag vs. an operator's budget
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
HTTP surface: `http_apis/events_api.py` (`POST /events`)

Everything the coordinator knows about the fleet's cache contents is
built from one event stream. This layer is where that stream enters:
it decides **what** is admitted (the gate) and **who** sees it (the
broadcaster). Neither holds cache state — the consumers do.

```
source adapters                     ingest layer                  consumers
────────────────────────────────────────────────────────────────────────────────
POST /events ──▶ HttpCacheEventSource ──┐
 (HTTP push)      .ingest(batches)       ├─▶ EventGate ──▶ CacheEventBroadcaster
Kafka topic  ──▶ KafkaCacheEventSource ──┘    fence /      .broadcast(batch) ────▶ KeyDirectory
 (durable pull)   poll thread                  dedup /       .fence_instance(id) ──▶ FleetEvictionController
                                               gap detect
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
`(instance_id, incarnation, seq)`. Source adapters feed ordered batch
lists to `EventGate.ingest_batches`, which reports the aggregate admitted /
duplicate / stale counts.

Two adapters exist, and a coordinator runs exactly one of them, selected by
`--event-transport` (default `http`). One path per emitter stream is what
the gate's per-emitter `seq` cursor assumes: two paths carrying the same
stream could interleave, and a late lower `seq` would then be dropped as a
duplicate it never was.

- **`HttpCacheEventSource`**, fed by `POST /events` from the MP-server
  `CacheEventSubscriber` (see [cache_events.md](cache_events.md)). A
  non-durable push source: FastAPI owns its request lifecycle, and it cannot
  seek or replay events that failed before the coordinator accepted them. It
  reports `replay_capability=none` and implements no silent no-op `seek`.
- **`KafkaCacheEventSource`**, selected by `--event-transport kafka` (with
  `--kafka-bootstrap-servers`, `--kafka-topic`, `--kafka-group-id`,
  `--max-ready-lag`);
  `POST /events` then answers 404. One
  poll thread reads the topic the MP servers' `KafkaCacheEventSink`
  produces to -- one `CacheEventsRequest` envelope per record (the
  `POST /events` body), keyed by `instance_id` so a partition is one
  instance's stream in order -- and
  offers each
  record's batches to `ingest_batches`. A record's offset is recorded
  into `StreamPosition` (see below) only after the gate has seen it, and
  stored for the consumer group's own auto-commit at the same point, so
  delivery is at-least-once; the gate's dedup absorbs any redelivery. A
  record that does not decode, or that makes a consumer raise, is logged
  and skipped -- and still recorded, so it cannot stall its partition. It
  reports `replay_capability=seekable` because the topic retains the
  stream: `StreamPosition`, and an operator resetting the group's offsets
  by hand, both replay it.

  The same poll thread also refreshes a cached `lag`: the sum, across its
  assigned partitions, of each partition's high watermark minus the
  consumer's current position (cached watermarks first, a live query only
  for a partition nothing has been fetched from yet). `lag` is
  `UNKNOWN_LAG` (`-1`) until the first refresh resolves, or whenever the
  broker cannot answer -- an unreachable broker must never read as
  "caught up." `HttpCacheEventSource` always reports `lag=0`: nothing is
  retained, so nothing can be behind.

## Resuming correctly (`StreamPosition`)

Two things could answer "where does a restarted consumer resume from,"
and they disagree in exactly the window that matters. Kafka's own
consumer-group commit and the coordinator's own checkpoint save are
independent, on independent timers (offsets commit roughly every few
seconds; a checkpoint every `--checkpoint-interval`, default 60s, or
only at a clean shutdown). On an ungraceful crash between two checkpoint
saves, the committed offset is already ahead of what the last saved
checkpoint reflects -- and resuming from it, as a plain consumer group
would, replays nothing in between: the broker considers those records
delivered, and the checkpoint that gets restored does not.

`StreamPosition` is checkpointed in the *same* artifact as the state it
describes (registered alongside `EventGate` in `app.py`'s
`checkpoint_components`), recording each partition's offset only after
the gate has admitted that record -- so a captured position can only
ever lag the state beside it in the checkpoint, never lead it.
`KafkaCacheEventSource`'s `on_assign` callback seeks every assigned
partition to `StreamPosition.next_offset(...)` rather than trusting the
group's committed offset, so a restart always resumes from what the
coordinator's own restored state proves it has seen. A partition with no
recorded position (a first run, or a genuinely new partition) falls back
to the group's committed offset, or `auto.offset.reset` if the group has
none either -- so a deployment with no checkpoint configured still gets
Kafka's own best-effort resumption, just not the stronger guarantee.

Transport positions (Kafka partition offsets) are a separate coordinate
system from the gate's per-emitter seq cursors -- one counts records
per topic-partition, the other per logical emitter -- and both ride in
the checkpoint, independently.

## Readiness (`IngestReadiness`)

A coordinator resuming from `StreamPosition` is behind for a while
whenever that resumption point is far behind the topic's tip -- a
coordinator down for a while, replaying a real backlog: it restores
quotas in full (metadata) before ingest starts, but the usage those
quotas are enforced against fills back in only as the poll thread
applies the backlog. Planning in that window compares a
real limit against partial usage, which orders evictions the fleet does
not need -- not a smaller correct plan, a wrong one.

Rather than let every controller reason about a partially-caught-up
coordinator, the lifespan (`app.py`) does not let one exist:
`IngestReadiness.wait_until_ready()` blocks startup -- serving nothing,
not even `/healthz` -- until the source's `lag` is within
`--max-ready-lag` (default `1000`, Kafka-only), logging progress on a
timer while it waits. Only once that clears do controllers start. An
`UNKNOWN_LAG` source counts as not ready and holds startup indefinitely
-- the safe direction, since a coordinator that cannot reach its broker
must not come up acting on a view it knows is incomplete. A deployment
with no durable transport clears immediately: HTTP's lag is always `0`.

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
[usage_and_eviction.md](usage_and_eviction.md)), not a property of
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

- **Gap visibility**: `gap_detected` still has no HTTP endpoint --
  `GET /directory/stats` deliberately reports directory contents only.
  Lag (how far behind) now gates startup but is not otherwise exposed;
  a gap (events that will never arrive) is a different condition and
  stays invisible either way.
- **Operator-driven replay from an arbitrary point**: a restart already
  resumes correctly from `StreamPosition` (or, absent one, the group's
  committed offset), and `IngestReadiness` makes it safe to act on once
  caught up. Replaying from further back than that -- reprocessing
  retained history the checkpoint has already moved past -- is still
  manual: an operator clears the `kafka_stream_position` checkpoint
  section (and resets the consumer group's offsets, if a checkpoint isn't
  the only thing pinning them) by hand.
- **Registry integration**: calling `EventGate.drop_instance` from
  deregistration / heartbeat-timeout eviction.
- **Allocation generations** for shared pools (deterministic
  cross-reporter conflict resolution).

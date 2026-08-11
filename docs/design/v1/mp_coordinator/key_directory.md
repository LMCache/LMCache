# Key directory

Module: `lmcache/v1/mp_coordinator/key_directory.py`
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
HTTP surface: `lmcache/v1/mp_coordinator/http_apis/directory_api.py` (`/directory/*`)

This is the first milestone (M1) of the control-plane RFC
([issue #4226](https://github.com/LMCache/LMCache/issues/4226)): a fleet-wide
directory mapping each `ObjectKey` to its known placements — which instance
holds it, on which tier (`l1`/`l2`) and backend (`dram`, `cxl`, `fs`, ...), at
what size.

## Contract

The directory is **eventually consistent, soft state, and never on the
serving hot path**:

- It is built purely from `CacheEvent` batches emitted by MP servers. It
never mutates memory and never grants access to bytes.
- Every answer is a **hint**. Consumers (P2P discovery, cache control, prefetch planning) must validate at the owning MP server before touching bytes (validate-on-use). Stale state costs a wasted probe or a missed reuse.
- Almost all state is reconstructible by replaying the event stream, and
what is not — L2 placements no reporter will re-announce — is carried
across restarts by a checkpoint (see *Snapshot and restore*). Operator
intent is not reconstructible at all and is stored separately (see
*Metadata state*).

## Event application semantics

The directory is a `CacheEventConsumer`: batches reach `consume()` only
after the ingest layer's gate has ordered, deduped, and fenced them, so
nothing here depends on `seq` or `incarnation` — see
[ingest.md](ingest.md) for that half of the contract. What the
directory owns is what a batch *means*.

Application is idempotent, which is what lets the gate admit past a
detected gap: re-storing upserts, deleting an absent placement is a
no-op.

Entry semantics by event type:

- `STORE` — upsert the placement at its identity: fleet-scoped
`(tier, backend)` when the batch is `shared`, else
`(instance, tier, backend)`; re-store replaces the size. A new record
joins its chunk's token binding; an entry carrying `token_ids` fills
the binding's tokens (see below).
- `DELETE` — remove that placement identity (owners report evictions as
deletes too). Removing an absent placement/key is a no-op. A key with no
remaining placements is dropped from the directory (leaving its chunk's
token binding).
- `ACCESS` — refresh the key's `last_access` recency (max of batch `ts`);
never creates records, and carries no placement identity — its
`backend` may be empty (`tier`/`backend` are ignored on apply). `ts` is
emitter wall-clock and is never compared across instances.

Plus one non-batch hook, `fence_instance(instance_id)`: the gate calls
it when an emitter restarts (a higher incarnation) or leaves the fleet,
and the directory drops every **L1** placement that instance reported.
L2 placements survive — their bytes persist on disk across the
reporter's restarts and leave only via `DELETE`.

## Token index (chunk hash → token ids + keys)

The directory can answer "what token ids does this chunk hold?",
"where in its sequence did it sit?" and "which keys share this chunk?"
from the `STORE` entries themselves.
Every store entry carries its chunk's `token_ids` plus its
`token_offset` — a deliberate wire trade: both ride once per
rank/group/tier placement in exchange for a
self-contained protocol (no per-chunk canonical entry, no cross-entry
ordering to reason about). Bindings are keyed by `ObjectKey.chunk_hash`
— the identity every entry and every lookup already carries — so the
coordinator does no hashing at all.

`token_offset` is the position of the chunk's first token in the
sequence it was stored under, or `UNKNOWN_TOKEN_OFFSET` when no emitter
reported one — distinct from 0, which is a real position. It cannot be derived: chunk hashes are
prefix-chained, so the hash *implies* a unique position but does not
reveal it, and position-dependent reuse (blend re-RoPE, which shifts KV
from its stored position to the query's) needs the value. It is a
property of the chunk hash, not of a placement, so it lives on the
binding; two entries disagreeing about it would mean a hash collision.

**Representation.** Tokens are held as a read-only `uint32` numpy array
(`TokenBinding.token_ids`), ~1 KB per 256-token chunk against the ~10 KB
a `tuple[int, ...]` of boxed ints costs — a live-chunk footprint the
directory pays per chunk, not per placement, since every rank shares one
binding. It also keeps content comparison against a query window
vectorized. Token ids outside `uint32` leave the binding unfilled (a
lookup miss) instead of failing the batch.

The lookups this serves:

- **key → tokens**: `key.chunk_hash` → binding
  (`get_token_ids(chunk_hashes)`, served by `POST /directory/lookup`'s
  keys form). Content only — the stored position reaches the blend path
  as a match's `old_st`, never as a standalone lookup.
- **prefix tokens → keys**: stateless — `POST /directory/lookup`'s
  tokens form recomputes keys from the sequence; no index involved.
- **fragment tokens → keys** (blend-style): rolling
  fingerprints over the query discover candidate bindings, the query
  window is verified against `binding.token_ids` (exact), and the
  binding's keys give the placements. Discovery uses blend's cheap
  polynomial hash family; the index itself never needs a
  content-addressed key because every hit is token-verified. Implemented
  as a derived view over these bindings — see
  [blend_index.md](blend_index.md), served by `POST
  /directory/blend-lookup`.

Lifecycle is record lifecycle — bounded structurally, not configured:
every record joins its chunk's binding at creation and leaves when it
is dropped (delete or `fence_instance`); the binding dies with the
chunk's last key. A binding's tokens stay empty until one
of its entries was stamped (the emitter's token cache is bounded) — an
empty binding is a lookup miss, never an error, repaired by the chunk's
next stamped entry. Eventually consistent soft state, like the rest of
the directory.

Non-goal (deliberate): identical content stored under different
prefixes has different chunk hashes and binds per chunk — a
position-independent content-hash index (cross-prefix dedup, lookup by
content identity) is not required by the lookups above and can be
layered on later coordinator-side, since the tokens already arrive.

Emission is the emitter's side of the contract — see
[cache_events.md](cache_events.md).

## Shared pools

Storage media shared by several instances — one S3 bucket, a shared
filesystem, and soon shared L1 (e.g. a CXL pool) — carry the
operator-configured `shared` flag, and the **backend type name
identifies the pool** fleet-wide — deployments must keep one pool per
backend type (mounting two distinct pools under one type name would
merge them in the directory). Reporting is per **emitter stream**: in a
controllerless pool the mounting instances report the operations they
perform; a pool with its own allocation controller has the controller
report all pool placements on its stream (stores at reservation,
deletes at reclaim — the writing instance's id can ride the event as
payload the directory does not act on), making it the pool's sole
reporter:

- **Identity**: placement identity is `(tier, backend)` fleet-scoped
  when shared (vs `(instance, tier, backend)` private) — stores of one
  key into one shared pool by N instances upsert a single placement (no
  double counting), and any mounting instance's reported delete removes
  it.
- **Lifecycle**: uniform with private placements — fencing is by the
  *stream that reported it*, scoped to **L1**. A shared-L1 pool (e.g.
  CXL) whose controller is its sole reporter is therefore cleared by
  the controller's restart, exactly matching a pool reset; shared-L2
  pools (S3, NFS) survive any reporter's restart because the bytes do.
  A shared placement re-reported by a later emitter survives the
  original reporter's restart (the fence matches each placement's
  recorded reporter).
- **Ordering caveat**: shared pools have multiple writers with no
  cross-stream order, so a late-arriving report can briefly resurrect or
  miss a placement — within the validate-on-use contract, and repaired
  by event replay. When shared-medium controllers hand out a monotone
  allocation generation with each reservation, attaching it to entries
  would make cross-reporter conflicts deterministic (future work).
- `Placement.instance_id` is the **last reporter** for shared
  placements, not an owner; any instance mounting the pool can serve
  the bytes.
- **Controller-emitted batches**: the pool's own controller (which
  drives its allocations and evictions) reports to the coordinator
  directly. `instance_id` is really the *emitter stream id* — the
  controller sends under its own stable id and gets an ordinary
  seq-deduplicated, incarnation-fenced stream, with no special-casing
  anywhere in the directory.

## Structures

```
ObjectKey → _KeyRecord {
    placements: list[Placement],   # ≤1 per placement identity (see STORE)
    last_access
}
chunk_hash → _TokenBinding { token_ids: uint32[], token_offset, keys }
instance_id → set[ObjectKey]       # L1 reverse index
```

The L1 reverse index (`_l1_keys_by_instance`) is what makes
`fence_instance` proportional to the instance's own keys instead of a
full directory scan. The emitter's stream cursor is **not** here — it
belongs to the gate ([ingest.md](ingest.md)).

The Python-phase directory is keyed by `ObjectKey` directly (hashable
frozen dataclass). The RFC's 16-byte
`key_hash` with interned `model_id`/`salt_id` is a memory/native-port
optimization (M6), not a semantic change.

## Snapshot and restore

The event stream cannot rebuild the directory on its own: an L2 object
stored hours ago is never re-announced. `KeyDirectory.snapshot()` captures
the state and `restore()` loads it back, across three layers that know
nothing of each other — `persistence/snapshot_codec.py` owns the format
(msgpack via `msgspec`, as the trace subsystem does, with the token
arrays written raw beside it so the encoder never copies them),
[`store.py`](../../../../lmcache/v1/mp_coordinator/persistence/store.py)
owns where the bytes live (`LocalArtifactStore` today, an object store one
class later), and `persistence/checkpoint.py` owns when.

- **`snapshot()` copies nothing large.** It returns an immutable capture
sharing no mutable state with the directory, taken under the lock, so the
caller can encode it off-thread. Token arrays are aliased — bindings
replace them wholesale and never write in place.
- **What is not captured.** The blend index is derived from the token
bindings and rebuilt on restore, so `enable_blend_lookup` must run first.
Content-free bindings and each binding's key set are likewise re-derived
from the key table.
- **The ingest gate's cursors ride along.** They are not the directory's
to capture — the gate owns them (see [ingest.md](ingest.md)) — but they
must be restored *with* the placements they gate, because fencing only
fires when a prior incarnation exists to compare against. Restore the
directory without them and an emitter that restarted during the outage
looks brand new, so the L1 slice its previous incarnation reported is
never dropped. `write_snapshot` therefore takes both, and
`EventGate.stats()` doubles as the capture half of
`EventGate.restore_cursors`.
- **Cursors restore as captured**, so an instance that kept emitting
while the coordinator was away trips `gap_detected` on its first batch
back. That is a true report; producer-side spill is what would close it.
- **The derived views are rebuilt too.** `restore` fills the directory
alone, so the load replays restored placements through the
`CacheEventBroadcaster` as synthesized `STORE` batches — otherwise the
coordinator returns holding a full directory and zero accounted bytes,
with quota enforcement blind to everything stored before the restart.
Replay order does not matter: what it rebuilds is per-salt byte totals.
- **Anything the replay cannot rebuild is a `DurableComponent`** whose
`persistence_type` is `CHECKPOINT`; it lands in the checkpoint's trailing
sections, restored after the replay so it replaces whatever that left
behind.
Today that is the eviction policy: an ordering-based one (the per-salt
LRU) *is* its recency, and `last_access` cannot stand in for it — that
is one flush-quantized timestamp per key rather than a sequence, and
comparable across emitters only as far as their clocks agree. Durability
is optional on `EvictionPolicy`, defaulting to capturing nothing, so a
policy that derives everything from its keys needs no section and a
deployment that swaps policies simply finds none under the new name. The
sections are JSON inside the binary artifact, so a component owns its
shape without knowing the key table.
- **Atomicity belongs to the store**, not the caller: the local backend
writes beside the target and renames, an object store would simply PUT.
- **Every failure is survivable.** A missing, corrupt, or
future-version checkpoint logs and starts cold; a failed write retries
next tick. The coordinator never refuses to boot over one.

Enabled by `--snapshot-path`; `--snapshot-interval` (default 60s) paces the
writes, `0` writes only on shutdown, and the lifespan writes once more on a
clean stop.

## Metadata state

Pins and per-`cache_salt` quotas, in
[`metadata_persister.py`](../../../../lmcache/v1/mp_coordinator/persistence/metadata_persister.py)
behind `--metadata-path`. A separate artifact because it agrees with the
snapshot on nothing that matters: small, JSON, unreconstructible, and
consequential to lose.

- **Components serialize themselves.** Each implements `name` /
`persistence_type` / `capture` / `restore` — the shape
`CacheEventBroadcaster.register_consumer` uses. The persister knows
storage and nothing else, and takes only `METADATA` components,
rejecting derived state outright.
- **Controllers are discovered, not listed.** `build_durable_owners`
scans `controllers/` for classes implementing `DurableOwner` and builds
each one — via its `from_config` when it has one, so configuration stays
next to the code that reads it. `collect_durable_components` then asks
each owner what it needs persisted (`FleetEvictionController` returns the
pin table, the quota registry, and the eviction policy) and groups them
by `persistence_type`. Dropping a controller into that directory is
therefore the whole of adding it: nothing in `create_app` enumerates
controllers or knows what any of them stores. Each metadata section validates itself with a Pydantic
model, so nothing outside a component parses the document.
- **Written on change, not on a tick.** Intent moves rarely, so
`MetadataPersister.save()` runs synchronously with the request that changed something —
a `200` from `POST /cache/pins` means the pin survives a restart. Every
mutating handler must `await` it; that is the one contract a new pin or
quota endpoint has to honour.
- **Restore order matters.** Metadata loads before the checkpoint's
replay: quotas *arm* eviction, pins are the only thing holding it off,
and the replay is what puts keys in the LRU.
- **Restored intent may be stale**, and staleness here is not harmless —
a quota raised while the coordinator was down comes back at its old lower
value and over-evicts. The load logs the document's age; the external
controller stays the authority, as it already is for quotas.
- **Failures degrade to no state**, which leaves eviction disarmed
(`effective_limit_bytes` returns `None`) rather than enforcing something
wrong.

## HTTP surface

- `POST /events` — offer `CacheEventBatch` batches to the
ingest gate (list order; per-instance emission order required).
Duplicates and stale batches are counted in the response, not errors.
See [ingest.md](ingest.md).
- `POST /directory/lookup` — resolve content to placements **and** token
ids, in either direction (POST because the payload rides in the body).
Supply exactly one of: `keys` (resolve keys directly) or `token_ids`
(prefix-exact resolution via the fleet `TokenHasher` + per-rank fan-out,
as the pin APIs do; requires `model_name` / `world_size` / `cache_salt`
since key identity includes them — and the sequence must be the
request's whole prefix, since chunk hashes are prefix-chained). One
result per resolved key, request order, both fields empty for unknown
keys. Position-independent token matching arrives with the content
index (M2).
- `GET /directory/keys` — paginated listing (`offset`/`limit`) with
`tier`/`instance_id`/`backend` filters; each row carries the key, its
matching placements, recency, and `num_tokens` — a cheap indicator of
whether the chunk's tokens are known. Full token ids are deliberately
not inlined (a page repeats each chunk across its ranks/groups; fetch
content via `/directory/lookup` for exactly the keys that need it).
- `GET /directory/stats` — key/placement counts, per-instance L1 key
counts (the fencing index), and the blend-index counts; per-key L2
detail lives on the keys listing endpoint. Directory contents only —
per-emitter stream state lives on the ingest gate and has no endpoint
yet (see [ingest.md](ingest.md)).

Type placement:

- **`api.py`** — the cache-event vocabulary (`CacheEventType`,
`CacheEventEntry`, `CacheEventBatch`): the contract between the
MP-server emitter and the directory. Plain dataclasses with intrinsic
invariants in `__post_init__` (the `ObjectKey` pattern: `seq >= 1`,
concrete tier, non-empty ids are unconstructible anywhere).
- **`key_directory.py`** — the engine plus everything the directory
itself produces (`Placement`, `DirectoryStats`) and its private
records. Admission outcomes (`IngestResult`) and stream cursors
(`InstanceStreamStats`) belong to `ingest/event_gate.py`.
- **`schemas.py`** — HTTP models only. 

MP-server emission of the `CacheEvent` stream (L1 + L2, `incarnation` =
server start time) is implemented — see
[cache_events.md](cache_events.md). It is the fleet's single event
stream: `/events` offers it to the gate, which fans admitted
batches out to every consumer ([ingest.md](ingest.md)) — this
directory, and the eviction manager's per-salt usage view and LRU (see
[l2_usage_and_eviction.md](l2_usage_and_eviction.md)).

## Deliberately out of scope (follow-ups)
- **Stream-level follow-ups** (replay on `gap_detected`, registry-driven
`drop_instance`, shared-pool allocation generations) now live with the
gate — see [ingest.md](ingest.md).
- **Blend rewiring**: pointing the mp-server blend lookup at
`/directory/blend-lookup` and retiring `blend_directory.py` plus the
per-store fingerprint publish. The coordinator side is done — see
[blend_index.md](blend_index.md).
- **Producer-side spill**: a checkpoint restores what was captured, not
what arrived while the coordinator was down — those events are dropped
by the emitter and show up as a `seq` gap on the first batch back.
Buffering them at the MP server closes that window (see
[cache_events.md](cache_events.md)).
- The `DELETE_PENDING`/pin placement states used by tier-aware
cache-control directives (M3–M4 of the RFC).
- **Token store (I3)**: the opt-in `content_hash → token_ids` store for
`key → tokens` introspection, fed by `TOKENS` events and refcounted from
key records via the `content_hash` back-pointer. Nothing
correctness-bearing reads it, so it ships with its first real consumer.


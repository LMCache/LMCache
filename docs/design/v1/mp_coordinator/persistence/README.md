# Durable coordinator state

What the coordinator can rebuild after a restart, and what it cannot. This
covers the *contract* only: which state is durable, who owns it, and how a
copy is taken that is worth storing. Where the bytes go is a separate
concern (see the checkpoint and metadata artifacts).

## Everything persisted is a component

There is no rebuild step on load. Each piece of state implements the
contract below and restores itself from its own section:

| Component | Section | Why it cannot be recomputed |
|---|---|---|
| `KeyDirectory` | `key_directory` | A restarted MP server re-announces its L1, but nothing re-announces bytes already resting in L2 |
| `CacheUsageManager` | `cache_usage` | Byte accounting for those same placements; quota enforcement is blind without it |
| `EventGate` | `stream_cursors` | Fencing compares an arriving batch against a prior incarnation; with no cursor there is nothing to compare, and a restarted server's stale L1 slice is advertised forever |
| `IsolatedLRUEvictionPolicy` | `lru_order` | Recency is position, not a timestamp |
| `FleetEvictionController` | `pins` | Operator intent |
| `QuotaManager` | `quotas` | Operator intent |

Nothing in `create_app` enumerates them. Controllers are discovered by
scanning `controllers/`, each advertises what it needs persisted, and each
component says which artifact it belongs in -- so a new controller with
durable state is one file, and the wiring follows. The directory and the
ingest gate are named explicitly because they are durable but are not
controllers.

The alternative -- restore the directory, then re-deliver its placements
as synthesized `STORE` batches to rebuild the views -- couples every
component to a fake event stream and to an ordering: views must be
rebuilt before the components that overwrite them, and a component that
forgets to register is silently left empty. Loading each section directly
has neither problem, and a component that gains state gains it by
implementing four methods.

Restores are order-independent, and each component refuses a second load
rather than double-counting.

## The contract

A `DurableComponent`
([`durable.py`](../../../../../lmcache/v1/mp_coordinator/persistence/durable.py))
owns one section end to end:

| Member | Meaning |
|---|---|
| `name` | The section's name in its artifact |
| `persistence_type` | `CHECKPOINT` (derived from the event stream, disposable) or `METADATA` (operator intent, irreplaceable) |
| `capture()` | The current state, in the form the artifact holds |
| `restore(state)` | Replace the current state with a captured one |

Nothing outside a component understands its shape. That is what lets one
writer serve every section, and what lets a component change its own
encoding without touching the code that stores it -- which only holds if
a capture is **plain data**: nested dicts, lists and tuples of scalars,
strings and bytes. Hand back an `ObjectKey` or a numpy array and every
artifact writer has to learn that type, so components flatten their own
domain objects on the way out (`utils/encoding.py` for keys, since
several sections hold them). A test walks every capture and fails on
anything else.

`persistence_type` exists because the two kinds of state want opposite
things. Derived state is large, changes constantly, and costs only hit
rate when lost, so it suits a periodic write. Operator intent is small,
changes rarely, and cannot be reconstructed by anything, so it must be
written the moment it changes. One flag keeps that decision with the
component instead of with the caller.

## Why a capture needs a quiesce

Durable state is spread across consumers of the cache-event stream, each
with its own lock, and one batch is applied by more than one of them.
`create_app` registers the usage view before the eviction controller
precisely because the controller reads that view for the batch the view
just consumed.

Read them one after another while the event stream runs and the result is
a state that never existed -- bytes accounted against a key the policy has
no record of, or a key ordered for eviction whose bytes are not counted.
Restoring that is worse than restoring nothing, because it is plausible:
the numbers look fine and the quota arithmetic is quietly wrong.

`QuiesceLock`
([`quiesce.py`](../../../../../lmcache/v1/mp_coordinator/persistence/quiesce.py))
closes the window with a condition variable shared by two roles, and
`CacheEventBroadcaster` owns the one instance:

- **The gate** holds `applying()` around each mutating call -- `ingest`
  and `drop_instance` -- so the unit of atomicity is one batch (or one
  fence) applied *everywhere*, plus the cursor update that accompanies
  it. Arriving while a quiesce is pending parks the call before it starts,
  so a quiesce never interrupts work half-done.
- **A capture** takes the same lock, reads every component, and
  releases. Nothing in this PR does so yet; the checkpoint module will,
  and it keeps the quiesce and the reads together so a caller cannot take
  the torn version by forgetting to hold the lock. One capturer at a
  time: two overlapping captures would each clear the request on the way
  out, letting ingest resume while the second was still reading.

The gate holds the lock rather than the broadcaster, for three reasons. It is where
every mutation enters -- `fence_instance` reaches consumers without going
through `broadcast`, so a broadcaster-level hold missed it entirely. The
cursors are mutated there too, alongside the fan-out, and they are a
durable section themselves. And it is the only correct place: the quiesce
must be acquired **outside** the gate's own lock. Held inside it, an
ingest arriving during a capture takes the gate lock and then parks on
the quiesce, while the capture -- holding the quiesce -- waits for that
same lock to read the cursors. That deadlock is permanent; the entry
timeout is long spent by then.

Captures are serialized against each other, so two cannot interleave over
the same state.

Two consequences worth keeping in mind:

- **A quiesce stalls ingest**, so it must cover reads of in-memory state
  and nothing else. Encoding or writing under one pauses the fleet's
  event stream for the duration of a disk write. Even pure reads are not
  free: capturing a fleet with 100k L2 keys parks ingest for roughly
  150 ms (102 ms of it the directory), and that grows linearly with the
  key count. Bringing it down means capturing cheap references and
  flattening outside the quiesce, which needs replace-on-write internals
  the components do not have today.
- **A quiesce cannot be taken from inside a batch** -- it would wait for
  a batch that is waiting for it. Captures run on their own thread, never
  from a consumer.
- **A wedged batch times out** rather than stalling ingest indefinitely.
  A capture is best effort; the event stream is not.

### Why the directory is a component like any other

It is the largest section by far, and an earlier design kept it outside
the contract: the load path restored it first and then replayed its
placements as synthesized `STORE` batches to rebuild the other views, so
the directory was a substrate rather than a peer. With the replay gone
that argument goes with it -- nothing derives from the directory at load,
so it restores itself from its own section like everything else.

The one real consequence is for whoever writes the artifact: the
directory's capture holds numpy token arrays, which must be written
outside the msgpack document rather than inlined, or peak write memory
tracks the whole token corpus instead of the metadata.

## The two artifacts

| | Checkpoint | Metadata document |
|---|---|---|
| Holds | `CHECKPOINT` sections | `METADATA` sections |
| Format | msgpack, 12-byte header | JSON, indented |
| Written | every `--checkpoint-interval` and on a clean stop | synchronously, by whatever changed a pin or a quota |
| Losing it costs | hit rate until the fleet re-stores | an operator noticing and re-applying |
| Enabled by | `--checkpoint-path` | `--metadata-path` |

Both go through `ArtifactStore`: one whole object, replaced atomically,
at one location. The local backend writes beside the target and renames,
so a reader sees the previous artifact or the new one and never a torn
one; an unconfigured path gets a store that discards writes, so
persistence being off is not a case every caller tests for.

**The codec knows nothing about any section.** That is the payoff from
the plain-data contract: `write_checkpoint` is one `msgspec.msgpack`
encode of every section together, and adding a component changes no code
here. The cost is that a checkpoint is encoded whole before it is
written -- peak memory runs about 1.3x the file, so roughly 1.6 GB for a
fleet holding a million chunks, of which 84% is token content. Encoding
section by section is the escape hatch if that ever bites.

**Every failure is survivable.** A missing, corrupt, or future-version
artifact logs and leaves the coordinator cold rather than refusing to
boot; one unreadable section does not cost the others; a failed write is
logged and retried on the next tick. A checkpoint is an optimization, and
a coordinator that dies because it could not write one is strictly worse
than one that keeps serving.

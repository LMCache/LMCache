# Memory orchestrator

## Scope

`lmcache memory` is one process per shared memory region. It owns the
region's keys, extents, epoch, write tokens and read leases behind nine gRPC
calls, the five on the data path batched. It never touches KV bytes and never
learns what backs the region: a Device-DAX device several hosts attach, or a
file several servers on one host map. Its clients are L1 bindings inside MP
servers that map the region on their own host.

Limits of this milestone: one region per orchestrator, monotonic allocation
(no extent reuse), state in memory. Not here: reclamation or eviction, read
lease TTL and client fencing, HA or durability.

## Why a separate owner

- The pool outlives every server, so its allocator cannot live in one.
- CXL 2.0 multi-host memory has no coherent cross-host atomics, so locks or
  a free list inside the shared bytes cannot arbitrate.
- "This extent is committed" needs one global order, and a writer whose RPC
  timed out cannot fence itself.
- Reclaiming later needs every host's outstanding readers in one place.

| Alternative | Why not |
| --- | --- |
| Index, allocator and locks inside the region | Needs cache-coherent mappings; across CXL 2.0 hosts a crashed host can leave locks nobody can safely break |
| One MP server leads | Leader lifetime is not pool lifetime; election is the orchestrator with extra steps |
| Allocation inside the MP coordinator | Its directory is hint-only and eventual by contract; an authority on the per-request path needs its own lifetime and failure domain |

The cost is one process per region and two metadata round trips per batch.
Private L1s pay nothing; the MP coordinator stays a hint directory.

## Protocol

`protos/memory_orchestrator.proto`, package `lmcache.memory_orchestrator.v1`.
Every request carries an envelope (`region_id`, `expected_region_epoch`,
`client_id`, `client_incarnation`, `request_id`); a stale region, epoch or
incarnation is refused without a state change, and a repeated `request_id`
from the same incarnation gets the recorded reply while it is among that
incarnation's newest 1024 (a `CloseClient` reply is not recorded). Batches are
capped by `max_batch_entries`; the client splits larger ones, and a split
`ReserveWrite` or `ReserveRead` that fails part-way aborts or releases what its
earlier parts were granted.

`ReserveWrite` rejects a batch with `INVALID_ARGUMENT` if a live key's stored
shapes, dtypes or payload size differ from the request, before changing any
state. Reads still return the stored layout for the client to verify; this
write-side check does not replace precision-aware cache keys.

`RegionContract` describes the region before a client registers:

| Field | Meaning |
| --- | --- |
| `layout_fingerprint` | Operator-configured compatibility label, default `lmcache/kv_2ltd/v1`. Registration checks equality; it is not a computed hash or layout negotiation. The orchestrator does not inspect KV bytes. Per-object shapes and dtypes travel with each write. |
| `visibility_mode` | `coherent` requires a coherent mapping; `software_fenced` requires the client to perform the platform's visibility operations, for example on non-coherent shared CXL memory. Both require transfer completion before publishing. The service checks agreement but does not flush caches or synchronize DMA. |
| `max_batch_entries` | Maximum entries or tokens per RPC (default 4096). The client splits larger calls to keep each request bounded. |
| `reset_required` | The startup marker indicates potentially live users of lost metadata, such as after a crash or shutdown with registered clients. Only `DescribeRegion` is served until an offline reset. |

Object keys, tensor shapes and layouts reuse `ObjectKey`, `TensorShape` and
`MemoryLayoutDesc` from the MP transport's `protos/p2p_service.proto`.
`WireObjectKey` aliases `distributed.api.ObjectKey`, sharing its identity fields
and validation. Invalid wire keys reject the entire reservation RPC with
`INVALID_ARGUMENT` before changing region state. A codec test checks the shared
class and schema and verifies round trips; protobuf field numbers remain an
explicit wire contract rather than being inferred from Python field order.

| RPC | Does |
| --- | --- |
| `DescribeRegion` | region id, epoch, capacity, alignment, visibility mode, layout fingerprint, batch limit, `reset_required` |
| `RegisterClient` | registers `client_id` with a new incarnation; checks layout fingerprint, mapped bytes and visibility mode; retires the previous incarnation's writes and leases |
| `ReserveWrite` | per key: `WRITE_GRANTED` (handle + token) for a new key, `EXISTS_VALID` or `BUSY_WRITING` for a known one; if the batch's new keys do not all fit, every one of them gets `OUT_OF_SPACE` and nothing is allocated |
| `FinishWrite` / `AbortWrite` | tokens: `WRITING` to `VALID` / to `CONSUMED` (the extent is never reused) |
| `ReserveRead` / `FinishRead` | per key: `READ_GRANTED` (handle, layout, leases), `MISS`, `BUSY_WRITING`; release leases |
| `Usage`, `CloseClient` | counters; retire this incarnation |

These RPCs sit below `L1ManagerInterface`. The shared L1 binding will translate
its existing key-based calls: write keys and `MemoryLayoutDesc` become entries
with payload sizes, while read keys and `read_locks` become entries with reader
counts. Local tags and temporary staging remain binding concerns. The service
returns region-relative handles instead of process-local `MemoryObj` pointers.

Completion identifies a reservation, not just a key. If a write expires and
the same key is reserved again, its old token must not publish the replacement.
Each read lease likewise identifies one reader: retrying a release must not
decrement another reader's count. `TokenBatch` carries these tokens plus the
client/region envelope; the binding retains them behind the local L1 API.

```text
              ReserveWrite               FinishWrite
(no object) ---------------> WRITING -----------------> VALID (terminal here:
                                |                       no eviction or reuse)
                                |
                                |  Write expiry, AbortWrite, CloseClient, or
                                |  the writer registering again (new incarnation)
                                v
                            CONSUMED
            record dropped, extent never reused; a later store
            of the key gets a new extent
```

A client sends `FinishWrite` only once its copies into the extent are complete
and visible to other hosts; reads are granted on `VALID` only, and `VALID` is
terminal here. A write reservation expires after `--write-ttl-seconds`
(default 600). Key and write-token access lazily consume expired reservations
under the state lock. Their tokens become stale and a later store gets a fresh
extent. Usage counters reflect expiry once processed; idle records are not
swept. Committed objects and read leases do not expire.

## Failure semantics

Everything fails closed and no extent is reused, so no failure corrupts a
reader.

| Failure | Behaviour |
| --- | --- |
| `ReserveWrite` reply lost | Same-request retries replay the cached grant without renewing its deadline. If retries fail or the reply leaves the replay window, lazy write expiry unblocks the key |
| Client dies mid-write or holding leases | Its writes expire lazily; its read leases stay counted until it re-registers under its `client_id` or a reset |
| `FinishWrite` reply lost | The client reports its store failed; if the commit landed, a later `ReserveWrite` sees `EXISTS_VALID` |
| Orchestrator restarts | The startup marker survives, so it comes up `RESET_REQUIRED` with a new epoch and refuses every request but `DescribeRegion` (which reports it) until an offline reset |
| Second orchestrator for a region | Refused while the marker names a live process |
| Layout, size or visibility mismatch | `RegisterClient` fails |

Write expiry does not prove a remote DMA stopped: the old extent remains
consumed, and late completion or client cleanup cannot affect a replacement
reservation. Set the TTL above the expected transfer and RPC duration; slow
writes may expire too. Read lease TTL and client fencing are deferred.
Reset is offline: stop every client, stop the orchestrator, remove
`<state-dir>/<region-id>.marker`, start
both again. A clean stop with no clients deletes the marker itself.

The marker's hostname and PID are local liveness hints, not Kubernetes node
identity or client discovery. A replacement pod with a different hostname and
the same state directory is refused while the old marker remains. A stable
hostname alone cannot establish that two processes share a PID namespace.
Cross-pod recovery requires the offline reset above; automatic ownership
transfer and fencing are outside this milestone.

```bash
lmcache memory --region-id pool-a --capacity-gb 512 --alignment 2M \
    --visibility-mode software_fenced --listen 0.0.0.0:7700 \
    --state-dir /var/lib/lmcache/memory/pool-a
```

`--visibility-mode` (`software_fenced` for memory several hosts share without
cache coherence, `coherent` for one host) is advertised to clients and must
match theirs; `--max-batch-entries` caps one RPC (default 4096).

## Code map and tests

`state.py` holds `RegionState`, the pure-Python state machine; `service.py`
the gRPC servicer and reply window; `server.py` the CLI arguments, startup
marker and `serve` (exposed as `lmcache memory` by
`lmcache/cli/commands/memory.py`); `client.py` the client with deadlines,
retries and batch splitting; `api.py` and `_codec.py` the dataclasses and
their protobuf mapping. The bindings under `_proto_gen/` are generated at
build time like the MP protos; only `memory_orchestrator_pb2.pyi` is tracked.

`tests/v1/memory_orchestrator/test_state.py` covers the state machine (one
writer per key, reads on `VALID` only, all-or-nothing `OUT_OF_SPACE`, monotonic
allocation, idempotency, stale epoch and incarnation, retirement,
`RESET_REQUIRED`); `test_server.py` drives the real process over gRPC (full
flow, replay, startup marker, batch splitting and rollback).

## Follow-ups

External eviction needs a delete/eviction RPC in the reclamation follow-up.
A metadata-only delete could hide a key, but would not recover capacity with
this monotonic allocator. Extent reuse needs coordinated reader/writer
retirement and protection against late accesses; this PR keeps committed
objects immutable rather than offering a delete that implies reclamation.

Read lease TTL and client fencing (`Renew`, quarantine of orphaned extents);
reclamation and eviction (`RequestEvict`, `VALID -> EVICTING -> FREE`,
generation bump on reuse); several regions per orchestrator (a region table,
no wire change); region-scoped fleet cache events.

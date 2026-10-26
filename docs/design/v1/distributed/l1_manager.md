# L1Manager: staging objects and write tags

Design of the write path in `lmcache/v1/distributed/l1_manager.py`.

## Problem

Before this design, `reserve_write` created the object directly in the
resident table, write-locked. Every other component saw it immediately:
`reserve_read` returned `KEY_NOT_READABLE`, and a second `reserve_write`
returned `KEY_NOT_WRITABLE`. Two prefetch requests loading the same L2 object
therefore collided: the second reservation failed, the prefetch controller
abandoned its whole L2 load (all-or-nothing) and reported a non-hit for a
chunk that was seconds away from being resident.

## Contract

Two kinds of objects live in the manager:

| | resident object | staging object |
|---|---|---|
| owner | the key | `(key, tag)` -- one per writer tag |
| created by | admission of a staging object | `reserve_write` of a key that is not resident |
| readers | `reserve_read` / `unsafe_read` / `finish_read` | invisible (`KEY_NOT_EXIST`) |
| writers | `reserve_write` refuses it (`KEY_NOT_WRITABLE`) | another tag may stage the same key |
| ends | `delete`, eviction, temporary object read out | admission, `finish_write_and_delete`, reclaim after write-lock expiry |

**Tag.** A plain `str` naming the writer (a transaction): the prefetch
controller uses `prefetch:<request_id>`, `StorageManager` uses
`storage_manager` for engine stores, the serde wrapper `serde_wrapper`. The
default is `""`. Writers sharing a tag exclude each other on a key
(`KEY_NOT_WRITABLE`), which keeps the de-duplication engine stores rely on;
writers with different tags never block each other.

**Admission.** `finish_write`, `finish_write_and_reserve_read` and
`finish_write_and_delete` take the tag used at reservation and act on the
staging object `(key, tag)`:

- key not resident: the staging object *becomes* the resident object (tag
  dropped). `finish_write_and_reserve_read` read-locks it in the same
  critical section.
- key resident (another writer admitted first): the staging object is
  discarded (memory freed, debug log, `SUCCESS` returned -- the data is in
  L1 either way). `finish_write_and_reserve_read` takes its read locks on the
  resident object and returns *that* `MemoryObj`, so the caller always ends
  up holding what readers see.
- `finish_write_and_delete` never admits; it frees the staging object.

**Expiry and eviction.** A staging object keeps its write TTL lock. Once the
lock expires the reservation is abandoned:

- `finish_write*` on it returns `KEY_IN_WRONG_STATE` and leaves it in place;
- `is_key_evictable(key)` is true and `delete([key])` reclaims every expired
  staging object of the key (all of them with `force=True`). `delete` then
  returns `SUCCESS` when nothing is left for the key, and `KEY_IS_LOCKED`
  while a live staging object remains -- a live reservation pins the key
  even if its resident object is unlocked. `clear()` reclaims expired
  staging objects too.

The eviction policy learns about the key from `on_l1_keys_reserved_write`
(`L1EvictionPolicy` maps it to `on_keys_created`), so the regular eviction
loop -- `get_eviction_actions(key_eligible_filter=is_key_evictable)` followed
by `delete` -- covers abandoned reservations with no special casing. Policies
and controllers only ever see original keys; tags never leave the manager.

**Listener and event semantics.**

| step | listeners | event bus |
|---|---|---|
| `reserve_write` | `on_l1_keys_reserved_write(keys)` | `L1_WRITE_RESERVED` (`keys`, `tag`) |
| admission (non-temporary) | `on_l1_keys_write_finished` / `on_l1_keys_finish_write_and_reserve_read` | `L1_WRITE_FINISHED` / `L1_WRITE_FINISHED_AND_READ_RESERVED` |
| admission lost to a resident object | `on_l1_keys_reserved_read` (only for `_and_reserve_read`) | `L1_READ_RESERVED` |
| staging discarded / reclaimed, key still resident or still staged | none | none (debug log) |
| staging discarded / reclaimed, key now absent | `on_l1_keys_deleted_by_manager` | none |
| resident object deleted | `on_l1_keys_deleted_by_manager` | `L1_KEYS_EVICTED` |

Discards are deliberately not published as `L1_KEYS_EVICTED`: the coordinator
cache-event reporter and the L1 byte metrics treat that event as the deletion
of an admitted object.

## Staging area accounting

`get_staging_memory_usage()` returns the bytes held by staging objects (a
subset of `get_memory_usage()`'s used bytes); `report_status()` adds
`staging_object_count` / `staging_bytes`,
and the gauge `lmcache_mp.l1_staging_bytes` is registered next to the L1
usage gauges. It is meant for observability and for future in-flight
prefetch planning (how much L1 is committed to loads that have not landed).

## Residual race

A prefetch request still aborts its L2 load when a key becomes resident
between its L1 lock pass and its `reserve_write`
(`KEY_NOT_WRITABLE`, reported as `L2_PREFETCH_FAILED{reason="l1_contended"}`).
That window is the L2 lookup latency, not the L2 load time as before; see
`storage_controllers/prefetch_controller.md`.

## Contract-anchoring tests

`tests/v1/distributed/test_l1_manager.py` (`TestStaging*` classes) and
`tests/v1/distributed/test_prefetch_controller.py::TestConcurrentPrefetchSameKeys`.

## Asynchronous resident reads

A lookup read generation rejects completion from an expired lifetime, but
neither its TTL nor its shared reader count proves that a device copy has
finished. `pin_read_buffers` validates the lookup's original generation and
retains each source buffer under a unique transfer tag. `finish_read_transfer`
removes that pin after the stream finishes. If the original generation has
expired, it does not decrement a later reader's count. Duplicate tag completion
is harmless. Normal deletion and temporary-object cleanup respect these pins;
`force=True` retains its explicit unsafe semantics and requires quiescent I/O.

The MP Session arbitrates cancellation and transfer ownership. It starts with
the actual read count per retained ObjectKey. A worker claims one share per
key before pinning; FREE atomically takes only the remaining shares. Both
successful and unregistered-worker RETRIEVE use that ledger. Per-key claims
allow disjoint head/tail requests while rejecting repeated or overlapping
claims by the same worker instance. MLA instances can share a KV worker ID,
so the existing `(instance_id, worker_id)` wire identity distinguishes them.
Claims cannot exceed that Session's original reservation. A changed instance
ID does not provide proof of stable physical-rank identity across restarts.

The completion receipt contains only immutable original owner/key/generation
groups and the transfer tag. It does not consult mutable Session state. Pins
rejected through the documented L1 error results retire their claimed original
shares immediately; successful pins remain until stream completion. If
recording or publishing completion fails, the handler synchronizes first and
then completes the tag. Failed synchronization keeps the pins in place.
An unexpected exception from an L1 implementation after partial pinning does
not prove which copies or claims completed: known tag pins can settle, while
unvisited reader shares may remain until their TTL. The concrete L1's pin
operation returns ordinary failures as `L1Error` and performs no device calls.

This protocol has no lookup-attempt ID in a RETRIEVE or FREE message. Identical
lookups reusing a request ID cannot distinguish a late message from an earlier
attempt solely from its model/token/range fields. Original-generation checks
and per-instance claims are not a replacement for an attempt ID or a client
ordering guarantee.

Public contract tests live in `test_l1_interface.py`, `test_multi_l1.py`,
`test_session.py` and `test_lookup_lock_generation.py`. The latter runs actual
lookup, FREE and RETRIEVE handlers against real CPU L1/L2 storage while
controlling only device completion ordering.

## Lookup read-lock cleanup

Each shared read-lock lifetime has a process-local generation. Acquisition after expiry or after the reader count reaches zero creates a fresh generation, including when the same object key is recreated. Prefetch captures this value at L1 acquisition or L2 admission; its cleanup and the session's `free_lookup_locks` release only matching generations. A missing session cannot authorize key-only cleanup.

Readers within a live lifetime retain the existing shared count and TTL refresh behavior. Normal key-only read/completion and P2P APIs are unchanged; delayed completions on those paths and reuse of one request ID across lookup attempts are outside this cleanup fix. Storage traces record successful releases without serializing process-local generations; rejected releases remain visible in the completion event's `failed_keys`.

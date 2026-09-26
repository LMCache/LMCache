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

## Write-back holds

**Problem (#5350).** The `StoreController` learns about a new key from
`on_l1_keys_write_finished`, but only takes its read lock later, on its own
thread. In between, the key is resident and unlocked, so `is_key_evictable`
is true and the eviction loop may discard it. `reserve_read` then returns
`KEY_NOT_EXIST` and the chunk is never written to L2, even when L2 has room.
In cache terms: dirty data evicted before write-back.

**Contract.** A listener registered with `register_write_back_listener`
gets, per admitted non-temporary key, one *write-back hold* taken inside the
admission critical section, before the listener is notified:

- `is_key_evictable(key)` is false while any hold is outstanding, so the
  eviction loop skips the key.
- `delete()` ignores holds. Explicit deletes (API, store policies dropping a
  key from L1 after L2 success) keep their behavior.
- `release_write_back_holds(keys)` drops one hold per key and publishes
  nothing. Holds are TTL locks with the read TTL, so a hold that is never
  released stops pinning after `read_ttl_seconds`.

```text
finish_write ──(hold)──> StoreListener queue ──> _process_new_keys:
                                                   reserve_read per target
                                                   release_write_back_holds
            evictable?  no ───────────────────────────────────────> yes (after
                                                                   store locks)
```

The `StoreController` releases the holds after every target adapter has
taken its read lock (or no adapter was selected), and releases holds of keys
still queued when it stops. A hold normally lasts one store-loop iteration.

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
`storage_controllers/prefetch_l1_lock_pass.md`.

## Contract-anchoring tests

`tests/v1/distributed/test_l1_manager.py` (`TestStaging*` classes),
`tests/v1/distributed/test_prefetch_controller.py::TestConcurrentPrefetchSameKeys`
and `tests/v1/distributed/test_l1_write_back_hold.py` (write-back holds).

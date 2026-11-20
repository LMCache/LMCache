# StorageManager prefetch interface

Design notes for the prefetch entry points of
`lmcache/v1/distributed/storage_manager.py`: `submit_prefetch_task`,
`query_prefetch_status` and `wait_prefetch_status`. The storage manager owns
the request handle and the logging; planning, locking and loading live in the
prefetch controller (see
[`storage_controllers/prefetch_controller.md`](storage_controllers/prefetch_controller.md)).

## Why grouped keys

A hybrid model stores each chunk of a request as several objects: one per
*object group* (full attention, sliding window, recurrent state, ...) and one
per *kv rank* (tensor-parallel shard). An earlier interface flattened all of
those objects into a single `list[ObjectKey]` with an implicit layout
(`chunk -> object group -> kv rank`), and every consumer of the result bitmap
re-derived that layout to fold it.

The interface makes the grouping explicit. Every `(object group, kv rank)`
is a **row**, and the prefetch result is one bitmap per row.

```
              chunk 0   chunk 1   chunk 2  ...
row 0  g0 r0  [ key ]   [ key ]   [ key ]        <- GroupedObjectKeys(object_group_id=0)
row 1  g0 r1  [ key ]   [ key ]   [ key ]
row 2  g1 r0  [ key ]   [ key ]   [ key ]        <- GroupedObjectKeys(object_group_id=1,
row 3  g1 r1  [ key ]   [ key ]   [ key ]                       sliding_window_size=w)
```

## Public types (`lmcache/v1/distributed/api.py`)

| Type | Purpose |
|---|---|
| `GroupedObjectKeys` | One row: `keys` (chunk-ordered, `keys[i]` covers tokens `[i*chunk, (i+1)*chunk)`), `object_group_id`, `layout_desc` (L1 write-buffer layout for L2 loads), `sliding_window_size` (`-1` full attention, `w >= 1` window). |
| `PrefetchTaskSpec` | `key_groups` (one `GroupedObjectKeys` per `(object group, kv rank)`, in any order; all of the same `group_size`), `num_kv_readers`, `fetching_policy`, `lock_mode`. |
| `FetchingPolicy` | `"prefix"`: only the longest prefix every row can serve under its window. `"full"`: complete columns across all rows, with gaps between columns allowed; sliding-window rows are refused. |
| `PrefetchLockMode` | `LOCK`: the caller reads the objects and releases them with `finish_read_prefetched`. `NO_LOCK`: warm-up, nothing stays locked; loaded objects are permanent. |
| `PrefetchHandle` | Opaque; carries `prefetch_request_id` (`-1` for an already-complete empty request), `external_request_id`, `total_requested_keys`, `submit_time` and the per-row `sliding_windows`. |
| `PrefetchResult` | `hit_cells`, `l1_hit_cells`, `l2_hit_cells`: one bitmap per row, in `key_groups` order; `l1_hit_count` / `l2_hit_count` properties. `found_cells` reports availability before staging; `l1_owners` identifies the managers holding retained locks. |
| `ipc_key_to_grouped_object_keys` | Builds the rows of a request from an `IPCCacheServerKey`, the chunk hashes, the object groups to read, the per-group layouts and the registration's `AttnWindowDesc`. |

### Contract

- `submit_prefetch_task(spec, external_request_id, skip_l2) -> PrefetchHandle`
  makes the objects the policy retains resident in L1: L1 hits are
  read-locked (under `LOCK`) and the rest is loaded from every attached L2
  adapter, planned in one step over the union of what L1 and L2 hold.
  `skip_l2` restricts the prefetch to what is already in L1; such a request,
  or one submitted with no L2 adapter attached, is served on the calling
  thread and its result is available when the call returns.
- `query_prefetch_status(handle) -> PrefetchResult | None` returns `None`
  while loading, otherwise the result: bit `i` of `hit_cells[k]` is set iff
  `key_groups[k].keys[i]` is resident (and read-locked under `LOCK`), split
  into `l1_hit_cells` (already in L1) and `l2_hit_cells` (loaded by this
  request; disjoint, union equals `hit_cells`). Each result is returned once.
  Callers fold `hit_cells` with `bitmap_ops.fold_unfold_grouped(rows,
  windows)`, passing each row's own `sliding_window_size`, to get the chunk
  hit count.
- `wait_prefetch_status(handle, timeout) -> bool` blocks until the result is
  published or the timeout elapses; it does not consume the result.
- Validation happens in `PrefetchTaskSpec.__post_init__` and raises
  `ValueError`: empty `key_groups`, key groups of different sizes, an unknown
  `fetching_policy`, and `num_kv_readers < 1`.

### Callers

| Caller | Rows | Policy / lock |
|---|---|---|
| `multiprocess/modules/lookup.py` (engine lookup) | every registered object group x every kv rank, via `ipc_key_to_grouped_object_keys` | `"prefix"`, `LOCK` |
| blend prefix leg (`modules/blend/lookup.py`) | the leg's read groups (attention + recurrent) x ranks | `"prefix"`, `LOCK` |
| blend sparse leg | the blend read groups (attention + aux) x ranks over the de-duplicated matched chunk hashes | `"full"`, `LOCK` |
| P2P receiver (`modules/p2p_controller.py`) | **one row per object group** holding the peer's keys in request order, ranks mixed. The peer only asks "are these resident"; it carries no chunk layout. If the groups receive different numbers of keys, or a group has no layout, the lookup logs an error and reports every key as a miss. | `"full"`, `LOCK`, `skip_l2=True` |
| warm prefetch (`warm_prefetch.py`, `cache_control/key_resolver.py`) | every registered object group x every kv rank, via `ipc_key_to_grouped_object_keys` -- the same rows as engine lookup | `"full"`, `NO_LOCK` |

### Fold

`bitmap_ops.fold_grouped`, `unfold_grouped` and `fold_unfold_grouped`
(`csrc/lmcache_native/fold.cpp`) take a list of row bitmaps paired 1:1 with a
list of window sizes and return the model-wide hit length and the per-row
retain masks; no ordering of the rows is assumed. The controller folds the
same way at finish, so a caller's fold of `hit_cells` reproduces the
controller's decision.

## Tests

- `tests/v1/distributed/test_distributed_storage_manager.py`: single-row and
  two-group (full attention + sliding window) prefetches through the public
  interface, `"full"` gap retention, `NO_LOCK`.
- `tests/v1/distributed/test_object_key_parallel.py`: one row per kv rank,
  folded with `fold_unfold_grouped`.
- `tests/v1/distributed/test_bitmap_ops.py`: grouped kernels against the flat
  kernels on random presence.
- `tests/v1/multiprocess/test_query_lookup_hits.py`, `test_p2p_controller.py`,
  `test_warm_prefetch.py`, blend tests: the rows each caller submits.
- `tests/v1/mp_observability/trace/test_codecs.py`: trace round-trips of the
  request types and of `PrefetchHandle.sliding_windows`.

## Internal write overflow and object ownership

The CLI configures one or more L1 managers. The internal `_l1_managers`
constructor input also accepts existing managers for tests; `StorageManager`
owns their lifetime. Key-only completion and legacy single-region/runtime
Device-DAX APIs require a single manager. Serving reads carry exact owners;
reporting and memory checks include every configured manager.

`OrderedWritePolicy` supplies stable manager IDs in explicit order, primary first
by default. Construction validates and captures that order; later mutations of
the supplied policy do not reconfigure the manager. `reserve_write` visits each
candidate synchronously and retries only
its `OUT_OF_MEMORY` subset. Successful keys and terminal conflicts do not advance;
exceptions propagate. Each L1 retains its allocation and staging rules. A failed
batch is not split: two 4 KiB objects can fail on two L1s with 4 KiB free each,
even though their combined free space is sufficient. Failed keys remain omitted
from the returned object mapping.

`L1Manager.reserve_write` stamps each new object with its process-local integer
`l1_manager_id`, including direct and prefetch reservations. `MemoryObj` starts
unowned, rejects a different live owner, and resets ownership when an allocator
reuses an object for a new lifetime. The tag is not a writer tag, key field, or
serialized `MemoryObjMetadata` field. The manager registry is O(managers); there
is no second object-to-owner table.

```text
reserve_write -> objects with owner IDs -> prepare_write_completion(objects)
                                                |
                                     [(owner_id, [ObjectKey, ...]), ...]
                                                |
                                     GPU copy completes on stream
                                                |
                                     finish_write_by_owner(payload)
```

The native callback uses that MessagePack-compatible payload, never Python
objects or pointers. Its handler resolves the captured IDs and passes the
original StorageManager writer tag to each L1's `finish_write`. Changing placement
order cannot redirect completion. Unknown or unset owners fail explicitly; the
legacy `finish_write(keys)` remains valid only with one L1. Existing staging and
stream ordering retain allocations; the tag is neither a lifetime pin nor a write
generation. It adds no stale-callback or crash-recovery guarantee.

Completion preparation runs inside the store's failure boundary before success
is recorded. Invalid ownership skips admission and reports `MP_STORE_END` with
zero stored objects. As with copy failures, staging reservations retain their
write-TTL behavior because queued device writes can still reference the buffers.

When tracing is enabled, both single-L1 completion entry points record one
existing `finish_write(keys)` trace call. Process-local owner IDs are not written
to the storage trace, so the unchanged dispatcher can replay it against a fresh
manager. Multi-L1 trace replay remains unsupported.

`test_multi_l1.py`, `test_l1_owner.py`, and `test_l1_owner_completion.py` cover
overflow, batch cleanup, independent same-key copies, recycled ownership, and
serialized completion.

## Configured peer L1 serving

`StorageManagerConfig.l1_manager_configs` lists tagged DRAM, Device-DAX, or GDS
configurations. The legacy singular config aliases the first entry. Each manager
owns its allocator, eviction controller, and store controller. Initialization
rolls back already-created managers, adapters, and controllers on failure.

L2 affinity is a fixed mapping from `L2AdapterConfigBase.affinity_tag` to an L1
identity. Stores and serde use that L1, and prefetch reloads allocate there. There
is no pluggable affinity policy. Unknown targets and GDS targets for host-buffer
adapters are rejected. Runtime adapter attachment updates the controller's
mapping on its own loop thread; removal drops it after draining.

Prefetch uses synchronous L1 calls and the existing per-manager lock maps. Its
policy discards redundant copies and returns `PrefetchResult.l1_owners` for retained
keys. Each session keeps its own owner map. Multi-L1 reads and releases require
that map; they do not search for another copy by key. Stream callbacks carry
owner/key groups captured before enqueue. A failed peer lookup releases earlier
reservations, and a failed GPU retrieve releases all retained groups after
already-enqueued work finishes.

GDS slab context lifetime belongs to its L1. GPU buffers register after L1
construction, transfers resolve the slab by object owner, and shutdown drains
GPU work before freeing the slab. One active GDS slab per process remains a hard
limit because native stream registration is shared. Device-DAX peers own disjoint
mappings; duplicate device ownership is rejected.

Status exposes manager and controller dictionaries keyed by tag. Capacity is
summed by physical backing medium, while usage is the sum of allocator usage.
Prometheus gauges distinguish managers by `l1_tag` and `backend`; L1 operation
counters and optional lifecycle histograms include `l1_tag`. Lifecycle sampling
tracks `(tag, key)` so one copy's eviction cannot end another copy's lifetime.
Single-L1 status aliases and key-only completion remain
compatible. Single-region descriptors, P2P, shared-memory transfer, and legacy
Device-DAX hotplug remain guarded for multi-L1 configurations.

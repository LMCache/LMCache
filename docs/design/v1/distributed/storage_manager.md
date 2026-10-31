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

- `submit_prefetch_task(spec, external_request_id) -> PrefetchHandle`
  makes the objects the policy retains resident in L1: L1 hits are
  read-locked (under `LOCK`) and the rest is loaded from every attached L2
  adapter, planned in one step over the union of what L1 and L2 hold. The
  spec's `allowed_l1_memory_tags` / `allowed_l2_memory_tags` restrict which
  L1 managers count as hit sources and which L2 adapters are used (see
  `memory_tags.md`). A spec that allows no L2 adapter, or one submitted with
  no L2 adapter attached, is served on the calling thread and its result is
  available when the call returns.
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
| P2P receiver (`modules/p2p_controller.py`) | **one row per object group** holding the peer's keys in request order, ranks mixed. The peer only asks "are these resident"; it carries no chunk layout. If the groups receive different numbers of keys, or a group has no layout, the lookup logs an error and reports every key as a miss. | `"full"`, `LOCK`, `allowed_l1_memory_tags={(DRAM, LOCAL)}`, `allowed_l2_memory_tags=frozenset()` |
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

## Write placement and object ownership

The CLI configures one or more L1 managers; `StorageManager.__init__` builds
them with `create_l1_managers(config.l1_manager_configs)` and owns their
lifetime. `l1_managers()` returns them in placement order and
`get_l1_manager_by_tags(tags)` the one manager with the given memory tags
(`ValueError` for zero or several matches). Reporting and memory checks
include every configured manager.

`reserve_write(keys, layout)` asks the configured `WritePolicy`
(`StorageManagerConfig.write_policy`, default `"default"`; there is no CLI
flag yet) for a mapping `l1_manager_id -> keys`, validates it (registered
ids, every key placed exactly once) before touching any L1, and reserves each
group in its selected manager. A key its L1 cannot allocate is omitted from
the result: there is no overflow to another L1. The default policy sends every
key to the first L1 in placement order. See
`l1_manager/l1-manager-types.md` for the policy registry.

`BaseL1Manager.reserve_write` stamps each new object with its process-local
integer `l1_manager_id`, including direct and prefetch reservations.
`MemoryObj` starts unowned, rejects a different live owner, and resets
ownership when an allocator reuses an object for a new lifetime. The tag is not
a writer tag, key field, or serialized `MemoryObjMetadata` field. The manager
registry is O(managers); there is no second object-to-owner table.

```text
reserve_write(keys, layout) -> {key: MemoryObj}     objects carry their owner id
                |
                | l1_owners = l1_owners_of(objects)   {key: l1_manager_id}
                v
   GPU copy completes on stream; the callback (kind "finish_write")
   carries the owners dict as its msgpack payload, never a MemoryObj
                |
                v
   finish_write(keys, l1_owners)  -> groups the keys by owner
                                  -> each L1's finish_write(keys, tag)
```

`finish_write(keys, l1_owners)` takes the owners the way
`finish_read_prefetched` does, but requires them: every caller holds the
objects `reserve_write` returned, so there is no owner-less completion. The
caller builds `l1_owners` from those objects with the module-level
`l1_owners_of(objects)` (it reads `MemoryObj.get_l1_manager()` and raises
`ValueError` for an unowned object), so completion never re-runs the policy
and the storage manager keeps no per-key state. A key whose owner is missing
or unregistered raises `ValueError`. The stream callbacks in
`lmcache_driven_transfer` and `qstore`
submit kind `"finish_write"` with the owners dict; the shared-memory store
path keeps the same dict in its pending-write table. Existing staging and
stream ordering retain allocations. The owner tag is neither a lifetime pin
nor a write generation and adds no stale-callback or crash-recovery guarantee.

The owners are built inside the store's failure boundary before success is
recorded. Invalid ownership skips admission and reports `MP_STORE_END` with
zero stored objects. As with copy failures, staging reservations retain their
write-TTL behavior because queued device writes can still reference the buffers.

`prepare_read_completion` and `finish_read_by_owner` (typed with
`L1WriteCompletion`) are the read-side pair from before this design. They
still work but are deprecated (`lmcache_deprecate`, one warning per process):
new callers pass `l1_owners` to `finish_read_prefetched` themselves.

When tracing is enabled, `finish_write` and `finish_read_prefetched` redact
`l1_owners` and record key-only calls, so process-local owner ids never reach
the trace. The replay dispatcher keeps the owners of its own `reserve_write`
replays in `ReplayContext.write_owners` and hands them to the matching
`finish_write`; reads replay without owners, which fan out. A trace therefore
replays against a fresh manager and against a multi-L1 target alike.

`test_multi_l1.py`, `test_write_policy.py`, `test_l1_owner.py`, and
`test_l1_owner_completion.py` cover placement, invalid policy answers,
owner-grouped completion, independent same-key copies, recycled
ownership, and serialized completion.

## Configured peer L1 serving

`StorageManagerConfig.l1_manager_configs` lists tagged DRAM, Device-DAX, or GDS
configurations; `create_l1_manager` turns each into a `DramL1Manager`,
`DevDaxL1Manager` or `GdsL1Manager`. The legacy singular config aliases the
first entry. Each manager owns its allocator, eviction controller, and store
controller. Initialization rolls back already-created managers, adapters, and
controllers on failure.

L2 affinity is resolved from memory tags (see `memory_tags.md`): each adapter's
config lists the L1 memory tags it wants, most preferred first, and
`resolve_l1_affinity` picks the first L1 in placement order that has the most
preferred matching tags. Stores and serde use that L1, and prefetch reloads
allocate there. There is no pluggable affinity policy. An adapter that no L1
matches, such as a host-buffer adapter in a GDS-only server, is rejected.
Runtime adapter attachment updates the controller's mapping on its own loop
thread; removal drops it after draining.

Prefetch uses synchronous L1 calls and the existing per-manager lock maps. Its
policy discards redundant copies and returns `PrefetchResult.l1_owners` for retained
keys. Each session keeps its own owner map. `read_prefetched_results`,
`finish_read_prefetched` and `unsafe_read` take that map as `l1_owners`;
with it, each key goes exactly to its owner. Without it a single L1 serves
every key, as before. With several L1s the storage manager probes them in
placement order with `unsafe_read` and the first L1 holding the key
read-locked serves or releases it, logging a warning on every such call
(`unsafe_read` rather than `finish_read`, because an L1 without the key
answers `unsafe_read` silently). That release is correct only while a key is
read-locked in at most one L1, which holds today because the prefetch policy
keeps one copy per cell and releases the others; engine lookups and retrieves
always pass owners under multi-L1 (`engine_context.get_read_owners`), so the
probe never runs on the hot path. Stream callbacks carry owner/key groups
captured before enqueue. A failed peer lookup releases earlier reservations,
and a failed GPU retrieve releases all retained groups after already-enqueued
work finishes.

GDS slab context lifetime belongs to its L1. GPU buffers register after L1
construction, transfers resolve the slab by object owner, and shutdown drains
GPU work before freeing the slab. One active GDS slab per process remains a hard
limit because native stream registration is shared. Device-DAX peers own disjoint
mappings; `create_l1_managers` rejects duplicate device ownership.

Status exposes manager and controller dictionaries keyed by tag. Capacity is
summed by physical backing medium (`backend_type`, as
`l1_capacity_bytes_by_backend`), while usage is the sum of allocator usage.
Prometheus gauges distinguish managers by `l1_tag` and `backend`; L1 operation
counters and optional lifecycle histograms include `l1_tag`. Lifecycle sampling
tracks `(tag, key)` so one copy's eviction cannot end another copy's lifetime.
Single-L1 status aliases remain. P2P startup still requires a single DRAM L1
(`l1_exposes_single_memory_region`); the P2P controller resolves it with
`get_l1_manager_by_tags((DRAM, LOCAL))`, registers its `get_l1_memory_desc()`
with the transfer channel, and keeps its lookups and reads owner-less. The
`/reconfigure/dax/l1/*` routes act on the one `(CXL, LOCAL)` L1
(`_devdax_l1()`), answering `409` when there is none or more than one.

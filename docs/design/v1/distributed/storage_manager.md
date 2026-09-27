# StorageManager prefetch interface

Design notes for the prefetch entry points of
`lmcache/v1/distributed/storage_manager.py`:
`submit_prefetch_task` and `query_prefetch_status`. This is the first PR of the
MP prefetch-controller refactor; it changes the **interface** only. The L1
probe, the fold and the prefetch controller keep their current logic behind a
transitional adapter (see [Transitional adapter](#transitional-adapter)).

## Why grouped keys

A hybrid model stores each chunk of a request as several objects: one per
*object group* (full attention, sliding window, recurrent state, ...) and one
per *kv rank* (tensor-parallel shard). Before this change the caller flattened
all of those objects into a single `list[ObjectKey]` with an implicit layout
(`chunk -> object group -> kv rank`, stride `num_object_groups * world_size`),
and every consumer of the result bitmap re-derived that layout to fold it. The
layout lived in five callers and three result consumers.

The interface now makes the grouping explicit. Every `(object group, kv rank)`
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
| `FetchingPolicy` | `"prefix"`: only the longest prefix every object group can serve under its window; every object group has the same number of rows. `"full"`: every found object, gaps included. |
| `PrefetchLockMode` | `LOCK`: the caller reads the objects and releases them with `finish_read_prefetched`. `NO_LOCK`: warm-up, nothing is locked. |
| `PrefetchHandle` | Opaque request handle: `prefetch_request_id`, `external_request_id`, `total_requested_keys`, `submit_time`, and per-row `sliding_windows` (used to fold the result). |
| `ipc_key_to_grouped_object_keys` | Builds the rows of a request from an `IPCCacheServerKey`, the chunk hashes, the object groups to read, the per-group layouts and the registration's `AttnWindowDesc`. |

### Contract

- `submit_prefetch_task(spec, external_request_id, skip_l2) -> PrefetchHandle`
  read-locks (under `LOCK`) the L1-resident objects the policy retains and
  loads the rest from L2 asynchronously. `skip_l2` restricts the prefetch to
  what is already in L1.
- `query_prefetch_status(handle) -> PrefetchResult | None` returns `None`
  while loading, otherwise the `api.PrefetchResult`: `hit_cells` (one bitmap
  per `spec.key_groups` row in row order; bit `i` of `hit_cells[k]` is set
  iff `key_groups[k].keys[i]` is resident, and locked under `LOCK`) split
  into `l1_hit_cells` (already in L1) and `l2_hit_cells` (loaded from L2 by
  this request). Each result is returned once. Callers fold `hit_cells` with
  `bitmap_ops.fold_unfold_grouped(rows, windows)`, passing each row's own
  `sliding_window_size`, to get the chunk hit count; no stride arithmetic is
  needed.
- Validation happens in `PrefetchTaskSpec.__post_init__` and raises
  `ValueError`: empty `key_groups`, key groups of different sizes, and
  `num_kv_readers < 1`.

### Callers

| Caller | Rows | Policy / lock |
|---|---|---|
| `multiprocess/modules/lookup.py` (engine lookup) | every registered object group x every kv rank, via `ipc_key_to_grouped_object_keys` | `"prefix"`, `LOCK` |
| blend prefix leg (`modules/blend/lookup.py`) | the leg's read groups (attention + recurrent) x ranks | `"prefix"`, `LOCK` |
| blend sparse leg | the blend read groups (attention + aux) x ranks over the de-duplicated matched chunk hashes | `"full"`, `LOCK` |
| P2P receiver (`modules/p2p_controller.py`) | **one row per object group** holding the peer's keys in request order, ranks mixed. The peer only asks "are these resident"; it carries no chunk layout. If the groups receive different numbers of keys, or a group has no layout, the lookup logs an error and reports every key as a miss. | `"full"`, `LOCK`, `skip_l2=True` |
| warm prefetch (`warm_prefetch.py`, `cache_control/key_resolver.py`) | object group 0 x ranks | `"full"`, `NO_LOCK` |

### Prerequisite: grouped fold / unfold

`bitmap_ops.fold_grouped`, `unfold_grouped` and `fold_unfold_grouped`
(`csrc/lmcache_native/fold.cpp`) compute exactly what the flat `fold` /
`unfold` / `fold_unfold_ranked` compute, but over a list of row bitmaps paired
1:1 with a list of window sizes; no ordering of the rows is assumed. Consumers
fold the per-row result directly instead of interleaving it back into the flat
layout. The flat kernels remain for the prefetch controller.

## Internal types

The prefetch controller now consumes `PrefetchTaskSpec` directly, so the
transitional shims that once bridged the storage manager to the flat
`PrefetchRequestSpec` are gone (removed with the prefetch-controller-v2 PR,
#5346).

`TrimPolicy`, `PrefetchMode` and `PrefetchRequestSpec` moved from `api.py` to
`internal_api.py`; nothing outside the prefetch controller and the trace
codecs (`lmcache/v1/mp_observability/trace/codecs.py`, which still
encodes/decodes them for recorded traces) should build them.
`TrimPolicy.SEGMENTED_PREFIX` has no public equivalent (the storage manager
never dispatched it); blend's segmented retention is derived from the per-row
result instead.

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
  grouped types and of `PrefetchHandle` (including its optional
  `sliding_windows` field).

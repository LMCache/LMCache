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
row 0  g0 r0  [ key ]   [ key ]   [ key ]        <- GroupedKeys(object_group_id=0)
row 1  g0 r1  [ key ]   [ key ]   [ key ]
row 2  g1 r0  [ key ]   [ key ]   [ key ]        <- GroupedKeys(object_group_id=1,
row 3  g1 r1  [ key ]   [ key ]   [ key ]                       sliding_window_size=w)
```

## Public types (`lmcache/v1/distributed/api.py`)

| Type | Purpose |
|---|---|
| `GroupedKeys` | One row: `keys` (chunk-ordered, `keys[i]` covers tokens `[i*chunk, (i+1)*chunk)`), `object_group_id`, `layout_desc` (L1 write-buffer layout for L2 loads), `sliding_window_size` (`-1` full attention, `w >= 1` window). |
| `PrefetchTaskSpec` | `key_groups` (rows, **group-major / rank-minor**: the rows of one object group are adjacent, one per kv rank), `num_kv_readers`, `fetching_policy`, `lock_mode`. |
| `FetchingPolicy` | `"prefix"`: only the longest prefix every object group can serve under its window; rows must form a chunk grid (equal lengths, equal rows per group). `"full"`: every found object, gaps included; rows may be ragged. |
| `PrefetchLockMode` | `LOCK`: the caller reads the objects and releases them with `finish_read_prefetched`. `NO_LOCK`: warm-up, nothing is locked. |
| `PrefetchHandle` | Opaque; carries `row_lengths` so the result can be reported per row. |
| `ipc_key_to_grouped_keys` | Builds the rows of a request from an `IPCCacheServerKey`, the chunk hashes, the object groups to read, the per-group layouts and the registration's `AttnWindowDesc`. |

### Contract

- `submit_prefetch_task(spec, external_request_id, skip_l2) -> PrefetchHandle`
  read-locks (under `LOCK`) the L1-resident objects the policy retains and
  loads the rest from L2 asynchronously. `skip_l2` restricts the prefetch to
  what is already in L1.
- `query_prefetch_status(handle) -> list[Bitmap] | None` returns `None` while
  loading, otherwise one bitmap per `spec.key_groups` row in row order:
  bit `i` of `rows[k]` is set iff `key_groups[k].keys[i]` is resident (and
  locked under `LOCK`). Callers fold the rows with
  `bitmap_ops.fold_unfold_grouped(rows, world_size, windows)` to get the
  chunk hit count; no stride arithmetic is needed.
- Validation happens in `PrefetchTaskSpec.__post_init__` and raises
  `ValueError`: empty `key_groups`, non-adjacent rows of one object group,
  `num_kv_readers < 1`, and under `"prefix"` ragged rows or object groups with
  different row counts.

### Callers

| Caller | Rows | Policy / lock |
|---|---|---|
| `multiprocess/modules/lookup.py` (engine lookup) | every registered object group x every kv rank, via `ipc_key_to_grouped_keys` | `"prefix"`, `LOCK` |
| blend prefix leg (`modules/blend/lookup.py`) | the leg's read groups (attention + recurrent) x ranks | `"prefix"`, `LOCK` |
| blend sparse leg | the blend read groups (attention + aux) x ranks over the de-duplicated matched chunk hashes | `"full"`, `LOCK` |
| P2P receiver (`modules/p2p_controller.py`) | **one row per object group** holding the peer's keys in request order, ranks mixed. The peer only asks "are these resident"; it carries no chunk layout. Rows may be ragged. | `"full"`, `LOCK`, `skip_l2=True` |
| warm prefetch (`warm_prefetch.py`, `cache_control/key_resolver.py`) | object group 0 x ranks | `"full"`, `NO_LOCK` |

### Prerequisite: grouped fold / unfold

`bitmap_ops.fold_grouped`, `unfold_grouped` and `fold_unfold_grouped`
(`csrc/lmcache_native/fold.cpp`) compute exactly what the flat `fold` /
`unfold` / `fold_unfold_ranked` compute, but over row bitmaps
(`rows[g * num_ranks + r]`). Consumers fold the per-row result directly
instead of interleaving it back into the flat layout. The flat kernels remain
for the prefetch controller.

## Transitional adapter

The storage manager's L1 probe and the prefetch controller still consume the
flat, chunk-major `PrefetchRequestSpec`. Three private, **deprecated** shims in
`storage_manager.py` bridge the two worlds and are removed by the
prefetch-controller-v2 PR (`TODO(prefetch-v2)` markers):

- `_to_controller_spec(spec)` builds the legacy payload: `group_layout_descs`
  keyed by the rows' real object group ids (the controller allocates L1
  buffers per `key.object_group_id`), an `AttnWindowDesc` with one window per
  object group in row order and `world_size = rows per group`,
  `"prefix" -> TrimPolicy.PREFIX`, `"full" -> TrimPolicy.SPARSE`,
  `LOCK -> PrefetchMode.LOOKUP`, `NO_LOCK -> PrefetchMode.WARM`.
- `_flatten_rows(spec)`: uniform rows are interleaved chunk-major (today's
  `chunk -> group -> rank` order, which the flat fold expects); ragged rows
  (`"full"` only, never folded) are concatenated.
- `_split_rows(found, row_lengths)`: the exact inverse, applied to the flat
  result bitmap in `query_prefetch_status`.

`TrimPolicy`, `PrefetchMode` and `PrefetchRequestSpec` moved from `api.py` to
`internal_api.py` and are deprecated: nothing outside the storage manager and
the prefetch controller should build them. `TrimPolicy.SEGMENTED_PREFIX` has no
public equivalent (the storage manager never dispatched it); blend's segmented
retention is derived from the per-row result instead.

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
  new types and of `PrefetchHandle.row_lengths`.

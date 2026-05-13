# StorageManager read path

## Scope

This document covers the L1 read APIs in
`lmcache/v1/distributed/storage_manager.py` that are used after a
prefetch or lookup has reserved read locks:

- `read_prefetched_results()`
- `read_prefetched_results_partial()`
- `finish_read_prefetched()`
- `PartialReadResult`

The APIs all read from L1 memory. L2 lookup and prefetch decide which
objects should be present before this path runs.

## Full-read API

`read_prefetched_results(keys)` is the all-or-nothing read API. It calls
L1 `unsafe_read(keys)` and yields:

- `list[MemoryObj]` when every key is readable.
- `None` when any key cannot be read.

On failure, or if the caller raises while inside the context manager,
the storage manager releases read locks for the keys that were read
successfully and publishes `SM_READ_PREFETCHED_FINISHED`.

On normal success, the caller owns the read locks and must later call
`finish_read_prefetched(keys)` when it is done consuming the memory
objects.

## Partial-read API

`read_prefetched_results_partial(keys)` is the partial-recovery variant.
It never turns a partial miss into `None`. Instead, it yields a
`PartialReadResult`:

```python
@dataclass(frozen=True)
class PartialReadResult:
    good_objs: list[MemoryObj]
    good_indices: list[int]
    bad_indices: list[int]
```

The index lists are positions in the original `keys` list:

- `good_objs[i]` corresponds to `keys[good_indices[i]]`.
- `bad_indices` contains positions whose keys were not readable from L1.
- The order of `good_indices` follows the order of the input keys.

Example:

```python
keys = [key0, key1, key2, key3]

with storage_manager.read_prefetched_results_partial(keys) as result:
    # Suppose key1 is missing.
    assert result.good_indices == [0, 2, 3]
    assert result.bad_indices == [1]
```

The partial API is used by multiprocess retrieve so the server can copy
the chunks that are still available and report exactly which GPU blocks
were not filled.

## Lock ownership

The partial context manager follows the same ownership rule as the full
read API for successfully-read objects:

| Exit path | Read locks for good keys |
|---|---|
| Normal context exit | Remain held by the caller |
| Caller exception | Released by the context manager |
| Internal read exception | Released by the context manager |

On normal exit, callers must call:

```python
good_keys = [keys[i] for i in result.good_indices]
storage_manager.finish_read_prefetched(good_keys)
```

Keys in `bad_indices` did not produce `MemoryObj` values and do not have
read locks to release.

## Failure classification

Both read APIs share the same internal classification step:

1. Call L1 `unsafe_read(keys)`.
2. Preserve successful objects and their original input positions.
3. Track failed key positions separately.
4. Split failures into `KEY_NOT_EXIST` and `KEY_NOT_READABLE` buckets for
   anomaly reporting.

`KEY_NOT_EXIST` and `KEY_NOT_READABLE` after prefetch are treated as L1
read-failure anomalies because the caller is expected to read only after
reserving the objects.

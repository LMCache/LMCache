# Multiprocess partial retrieve

## Scope

This document covers the partial-recovery behavior in the multiprocess
`RETRIEVE` request path:

- Protocol response in `lmcache/v1/multiprocess/protocols/engine.py`
- Server retrieve implementation in `lmcache/v1/multiprocess/server.py`
- Interaction with `StorageManager.read_prefetched_results_partial()`

## Protocol contract

`RETRIEVE` returns:

```python
tuple[bytes, tuple[bool, list[int]]]
```

The outer `bytes` value is the CUDA event IPC handle. The inner tuple is:

- `bool`: `True` if the server completed the retrieve path and copied all
  available chunks. This can still be true when some chunks were missing.
- `list[int]`: GPU block IDs that were not filled because their source
  chunks were missing from L1.

On full success, the failed block list is empty:

```python
(event_handle, (True, []))
```

On partial success, the failed block list contains the exact destination
GPU block IDs for missing chunks:

```python
(event_handle, (True, failed_block_ids))
```

On an exception that prevents retrieve from completing, the server returns
all requested GPU block IDs as failed:

```python
(event_handle, (False, list(gpu_block_ids)))
```

Callers must treat `success=True` as "the server completed the operation",
not as "every requested block was filled". The authoritative per-block
failure signal is `failed_block_ids`.

## Why there are two copy loops

The full-hit path uses `_retrieve_loop()`. It batches contiguous chunks
because the memory objects and destination GPU block IDs have the same
dense ordering.

The partial-hit path uses `_partial_retrieve_loop()`. It processes one
chunk at a time because missing chunks make the source objects sparse:

```text
keys:             [k0, k1, k2, k3]
good_indices:     0       2   3
bad_indices:          1
gpu block ranges: [b0] [b1] [b2] [b3]
```

`good_objs` contains only `[obj0, obj2, obj3]`, so batching it as a dense
list would scatter `obj2` into `b1`. The partial loop uses
`good_indices` to slice the correct destination block range for each
object.

## Retrieve flow

1. Convert the IPC key into per-chunk `ObjectKey` values.
2. Stage all destination GPU block IDs once.
3. Enter `read_prefetched_results_partial(obj_keys)`.
4. Convert every `bad_index` into its GPU block-ID range and append it to
   `failed_block_ids`.
5. Copy available chunks:
   - Use `_retrieve_loop()` when there are no bad indices.
   - Use `_partial_retrieve_loop()` when some chunks are missing.
6. On normal context exit, schedule `finish_read_prefetched(good_keys)` on
   the host callback path so read locks are released after GPU work is
   enqueued.
7. Return the event IPC handle and `(True, failed_block_ids)`.

## Lock ownership

`read_prefetched_results_partial()` leaves read locks held on normal exit.
`MPCacheEngine.retrieve()` therefore builds:

```python
prefetched_keys = [obj_keys[i] for i in partial_result.good_indices]
```

and calls `finish_read_prefetched(prefetched_keys)` after the GPU copy has
been enqueued. Missing chunks never acquired read locks and are not passed
to `finish_read_prefetched()`.

If an exception occurs inside the context manager, the storage manager
releases any successfully acquired read locks before the exception leaves
the context.

## APC-aligned skips

Both loops honor `skip_first_n_tokens`. This prevents retrieve from
overwriting APC-shared prefix blocks. The full loop computes the skip for
each dense batch; the partial loop computes it for each surviving chunk
using that chunk's original index.

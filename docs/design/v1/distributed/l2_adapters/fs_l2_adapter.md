# Python filesystem L2 lookups

`lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py` provides the Python `fs`
adapter. Its lookups inspect the filesystem directly, rather than using a
process-local existence index, so files created or removed by another node
remain visible.

## Batch scheduling

A lookup submits its key list to the adapter's asyncio loop. The loop offloads
the entire metadata walk to one default-executor worker, awaits its bitmap,
publishes the result under the adapter lock, and signals the lookup eventfd.
The worker checks keys in input order and sets one bit per existing file.

```text
submit_lookup_and_lock_task(keys)
  -> adapter event loop
     -> executor worker: stat keys in order -> bitmap
     -> publish bitmap and signal eventfd
query_lookup_and_lock_result(task_id) -> bitmap, once
```

Each metadata syscall still observes the real filesystem. Batching removes an
executor submission and event-loop resumption per key; it does not reduce the
number of syscalls or introduce an existence cache. The event loop remains
available while a worker waits for storage, so unrelated lookups and I/O tasks
can progress.

A batch occupies one executor worker until its metadata walk finishes. Separate
batches use a dedicated pool with the same Python default worker sizing as the
event loop's executor. Store and load tasks keep using the event loop's
executor, so slow metadata calls cannot occupy their workers.

Closing the adapter cancels the coroutine that publishes the result. An
in-progress metadata syscall may return after cancellation, but the worker
checks cancellation before starting another key. The adapter discards the
cancelled batch's result.

## Results and races

The result bitmap has the same length and ordering as the submitted list,
including duplicate keys and missing files. Empty batches also publish a
completion. Results are consumed once through the existing public query method.

The filesystem adapter does not hold per-key eviction locks. A file removed
between lookup and load can still become a load miss. This behavior is unchanged
by moving the metadata walk to a worker.

## Tests

`tests/v1/distributed/test_fs_l2_adapter_lookup.py` covers sparse results at
multiple batch sizes, duplicates, empty batches, consume-once queries and
external creates/deletes. A test blocks one `os.stat` call and confirms a second
lookup completes before the first is released, protecting the non-blocking
adapter contract without testing executor internals. Existing store/load/delete
and O_DIRECT tests exercise the surrounding filesystem behavior.

The shutdown case blocks a metadata read, closes the adapter, then releases the
read and confirms no result is published.

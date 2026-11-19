# Store source ownership and shutdown

The store controller borrows resident L1 objects for asynchronous L2 writes.
A read-lock TTL bounds a reservation, not the lifetime of an admitted I/O:
recycling a source while an adapter still reads it can corrupt cached data.

For each adapter and batch, the controller:

1. Reserves reads and records their generations.
2. Calls `pin_read_buffers` with a process-unique transfer tag. Only the objects
   returned by that call are submitted; failed claims release their original
   generations through `finish_read`.
3. Keeps each successful pin until `pop_completed_store_tasks` reports a
   terminal success or failure, then calls `finish_read_transfer` once.

```text
L1 write -> reserve generation -> pin(tag) -> adapter submission
                                     |             |
                              rejected submit      terminal result
                                     +-------> release(tag)
```

Each replica owns a separate reservation and pin. Expiration of its reservation
does not make the source evictable. Completing an old generation cannot release
a newer reader's reservation. The existing read-pin interface supplies these
guarantees; the controller does not introduce another ownership mechanism.

## Adapter contract

A submission exception must leave no queued or running access to caller buffers.
After admission, the adapter must return a task ID and report failure only after
all reads belonging to that task have stopped. Both success and failure results
are buffer-lifetime fences, not just logical outcomes.

The controller cannot repair an adapter that reports completion early or admits
work and then raises without returning its ID. In particular, native allocation
failure during partial dispatch and external cancellation require backend-level
handling. This change does not prove SDK/RDMA teardown semantics. Serde-owned
temporary buffers have a separate lifetime and are outside this protection.

## Shutdown

`stop()` closes key admission, discards unsubmitted notifications, and keeps the
loop polling until every admitted store has a terminal result. L1 callbacks
after stop are ignored under the listener lock, including after its notifier is
closed. Runtime adapter add/remove requests are rejected once shutdown starts;
previously queued operations are applied before the loop exits.

```text
stop admission -> drain admitted stores -> release pins -> close notifier
                                                        -> adapter.close()
                                                        -> L1.close()
```

`StorageManager` already calls controller stop before adapter and L1 close.
Adapters must remain alive and continue publishing completions throughout the
drain. An adapter that never completes can block shutdown indefinitely. A
controller timeout must not release pins: bounded shutdown needs an adapter
fence that actually stops I/O, such as a separately validated isolated worker.
Explicit force deletion and concurrent external adapter/L1 close are unsupported
while I/O is active.

## Validation

`tests/v1/distributed/test_store_controller.py` uses real L1 state and native
extensions with gated asynchronous mock stores. It covers expiry, new reader
generations, replica ownership, rejected submissions, terminal failures, and
shutdown admission/draining. These tests do not measure model-serving or RDMA
performance.

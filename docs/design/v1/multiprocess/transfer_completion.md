# Transfer Completion: Stream-Ordered Release

Module: `lmcache/v1/multiprocess/transfer_completion.py`

## Problem

A `STORE`, `RETRIEVE`, `STORE_Q` or `CB_RETRIEVE_PRE_COMPUTED` handler enqueues
device work on a transfer stream and returns while that work is still running.
Two things it borrows must outlive the handler:

1. **The worker's producer event.** The handler imports the worker's event and
   queues a stream wait on it. The queued wait refers to the imported event
   until the stream consumes it. Dropping the import at handler return lets
   the event be destroyed while the wait is still pending.
2. **The reply.** The worker treats the reply as "the transfer is complete".
   The handler used to return at once with a freshly recorded server event for
   the worker to import and poll. That event was a handler local, so the server
   dropped it before the worker imported it, and whether the import then
   succeeded depended on how the device runtime treats a handle whose event is
   gone.

Both are the same shape of bug: a device object released at a point the server
*inferred* to be safe rather than one it *observed*.

## Contract

- An imported event is held until the stream that waits on it has consumed
  the wait.
- A transfer reply is sent only after the stream has run the work enqueued for
  that request.
- The server exports no device events. The `bytes` element of the reply is
  always empty. A non-empty handle is still understood by workers, so a newer
  worker keeps working against an older server.

## Mechanism

`TransferCompletion` lives on `MPCacheServerContext.transfer_completion`, one
per server process, and reuses the stream-ordered host callback path from
`native_completion.py`:

```text
handler (request thread)            transfer stream            dispatcher thread
wait_for_producer(handle, target)
  import; hold under handle         wait(event)
                                    host func: release(handle) -> drop the import
enqueue copy kernels                copy ...
                                    host func: finish_write    -> commit
reply_when_done(target, ok)         host func: resolve(id)     -> future.set_result
return Future                                                  -> transport sends reply
```

``target`` is whatever owns the transfer stream: the cache context for the
standard handlers, a ``TransferStreams`` triple for blend's private retrieve
stream.

Each release callback is queued on the same stream as the operation it
releases, directly behind it, so it fires only once that operation has been
consumed. The dispatcher drains callbacks in queue order, so the storage
commit runs before the worker can see the reply.

Handlers that exit before enqueuing the producer wait (unknown instance,
block-id underflow, blend no-op reasons) reply immediately with
`(b"", False)` or `(b"", scatter_ran)`: nothing was enqueued for the request,
so there is nothing for the reply to wait for. Every exit after the producer
wait defers, including failure exits, because kernels for the request may
already be in flight.

Imports are keyed by the worker's handle and held in a list, because two
handlers may import the same handle (a `STORE` and a `STORE_Q` of one forward);
each consumed wait releases one entry.

## Transport

`MessageQueueServer` sends a blocking handler's reply from a done-callback on
the handler's executor future. When the handler's result is itself a `Future`,
the callback is chained onto it, so the request thread is never blocked. The
gRPC servicer waits on the future on its worker thread before encoding.
Handlers annotate `T | Future[T]`; the gRPC codec strips the `Future` member
when validating the annotation against the service contract.

## Worker side

Unchanged. `DeviceMessagingFuture` already treats an empty event handle as "no
device event to wait for" and reports completion as soon as the reply arrives.
`get_finished` polls it once per step as before.

## Teardown

`cache_context.close()` synchronizes the transfer stream before unmapping the
KV cache, so every queued release callback has been recorded by the time the
context is dropped; the dispatcher drains them on its next tick, and the
transfer module's `close()` drains once more when the dispatcher stops.

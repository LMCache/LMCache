# Logical Sparse Prefetch Leases

## Problem

A serving runtime may know which logical KV chunks the next attention layer is
likely to use. It needs to ask LMCache to stage those chunks without exposing
the serving engine's physical page IDs, and it must keep the objects protected
until the GPU copy has finished.

The contract in this document is used by the SGLang adapter, but LMCache itself
does not depend on SGLang. The public boundary carries `ObjectKey` values,
request generations, layer IDs, and logical result bitmaps.

## Ownership

- The adapter owns the request/layer mapping from predicted physical pages to
  logical `ObjectKey` chunks.
- LMCache owns the prefetch controller request, L1 read locks, L2 load state,
  and the generation attached to the logical lease.
- The serving runtime owns GPU destination pages and its page table. It cannot
  reuse a destination page until the retrieve completion event has been
  observed.

No component releases a read lock merely because an RPC was submitted. A
successful release requires the controller operation to have completed, or a
retrieve copy stream to have been synchronized when the caller already
consumed the result bitmap.

## Lifecycle

```text
submit(instance, request, generation, layer, ObjectKeys)
    -> one active job for the exact four-part identity
    -> controller lookup/load retains the logical lease
wait/query
    -> retrieve(ObjectKeys, destination mapping, producer event)
    -> copy stream completes
    -> release consumed logical keys
```

The identity `(instance_id, request_id, generation, layer_id)` prevents a late
result from being applied to a reused request row. A second submit for the
same identity does not replace the first job; its competing handle is cleaned
up separately, and an uncertain cleanup remains reachable for retry.

Cancellation and release are idempotent. If a request is already gone, the
operation is a successful no-op. If the controller or connection cannot
confirm cleanup, the caller receives failure and the adapter keeps its cleanup
record rather than treating the remote lease as released.

## Retrieve failure ordering

Sparse retrieval may submit one or more H2D copies before a later object fails.
The read context therefore defers its normal lock release while the retrieve
is active. On failure, the transfer module first synchronizes the copy stream;
only then does it release the keys resolved for that job. If synchronization
cannot be completed, the job and its lease remain live so an explicit cleanup
can retry safely.

## Integration boundary

LMCache does not receive SGLang block IDs. The SGLang adapter computes logical
keys with the configured token-hash contract, sends the logical sparse request,
and supplies the physical destination mapping only to the registered local
transfer context. The adapter's completion event is retained until the RPC
future finishes. This keeps logical cache ownership in LMCache and physical
page ownership in the serving runtime, avoiding a second storage protocol or a
second owner for the same destination page.

This design covers one-layer lookahead. Multi-layer prediction, pipeline
parallelism, and speculative decoding require additional identity and
scheduling rules and are outside this change.

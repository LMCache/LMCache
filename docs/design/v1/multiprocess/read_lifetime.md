# Experimental MP read lifetime v1

This implementation separates a lookup's queue reservation from an active
transfer's allocation lifetime. It is opt-in for ordinary vLLM
`LMCacheMPConnector` retrieval with `lmcache_driven` transfers. Other paths
retain their existing behavior.

## Motivation and invariants

An anonymous TTL count cannot identify the request finishing a read. If A
expires, B reserves the same object, and A later finishes, A can consume B's
count. Expiration during asynchronous H2D can also make its source allocation
evictable before the device reads it.

The new path guarantees:

- Each lookup attempt has a fresh `lookup_id`, independent of `request_id` and
  object identity. Late cancellation of A cannot release B's reservation.
- Each object/lookup lease has its own monotonic deadline and reader shares.
  `consumer_id` identifies a TP worker, which can claim that lease once.
- Acquisition validates that exact live lease and creates a unique active
  handle while holding the L1 state lock. Expired reservations return a miss;
  they cannot borrow another reader's shares.
- Active handles refer to the original allocation and have no TTL. Eviction,
  forced deletion, forced clearing and request cleanup cannot free it.
- Device completion releases each active handle once. Replayed completion
  handles cannot decrement another allocation's count.

## Integration flow

```text
scheduler: new lookup UUID
  -> LOOKUP(key.lookup_id)
  -> PrefetchTaskSpec.read_owner
  -> exact per-object L1 owners and named shares
  -> QUERY/WAIT(request_id, lookup_id)
  -> connector request metadata.lookup_id
  -> worker RETRIEVE(key.lookup_id, worker_id)
  -> StorageManager.acquire_read(all retained groups, exact L1 owners)
  -> asynchronous copies using acquired MemoryObjs
  -> ActiveReadCompletionTracker.defer(handles, device, stream)
  -> platform CompletionEvent.is_complete()
  -> StorageManager.finish_active_reads(manager/handle groups)
```

All required groups are acquired before the first cache copy. A partial miss
rolls back every acquired handle and submits no cache copy. Sliding-window
prefixes and auxiliary groups follow the existing lookup lock model.

The active completion tracker records a platform event after the final source
use and polls every 5 ms, starting lazily on the first named transfer. It does
not rely on native host callback delivery: some platform fallbacks enqueue
those callbacks immediately without stream ordering. Lost callbacks therefore
do not strand active read completions either. Polling adds an event and CPU
work per transfer; performance has not yet been benchmarked.

On event-recording failure the tracker drains that stream before releasing.
If draining also fails, it retains the handles and propagates the error.
Query or release failures retain the batch for retry. Shutdown stops producers,
drains the tracker, and then closes storage. L1 refuses close while active
handles remain. A device that never completes can retain memory indefinitely;
there is intentionally no unsafe timeout escape.

## Cancellation and identity transport

`FREE_LOOKUP_LOCKS` removes only unclaimed shares for the supplied lookup and
range; active handles have separate ownership. Session destruction atomically
takes its recorded owners and rejects later result handoffs. A result that
loses this race releases its own shares. A lazy orphan-result reaper consumes
finished prefetch results after their sessions disappear, so cancellation does
not leak completed backend results. An in-progress prefetch settles through
its normal controller path before its result is discarded.

APC shrink gets a new UUID, releases the old attempt's shares and clears the
old pending ranges. Polls that do not submit a new lookup preserve the UUID
already sent. Eager prefetch uses the same identity handoff. The scheduler
retains attempt IDs until request end even after lookup-result bookkeeping is
removed for metadata handoff.

ZMQ carries the field structurally; new operations use string names without
changing frozen legacy operation IDs. gRPC adds `lookup_id` at field 11 of
`IpcCacheServerKey` (10 remains reserved), plus optional lookup IDs in status
and end-session messages. The opt-in scheduler requires
`get_read_lifetime_version() == 1` from every server. Matching client/server
builds are required; there is no fallback to anonymous locks for a named read.
An incompatible engine-driven worker context is rejected at registration.

## Boundaries of this version

- No CacheBlend, engine-driven transfer, SGLang or TensorRT-LLM migration.
- No claim of exactly-once request replay across process restart. Callers must
  generate a new UUID for each attempt and never reuse it for different work.
- Legacy anonymous APIs keep their old TTL semantics. They are not covered by
  these guarantees.
- Force clear remains destructive to legacy reservations when no active reads
  exist. While active reads exist it uses the normal protected clear path.
- Full vLLM serving, multi-node workloads and performance validation remain
  separate from the focused lifecycle, connector/RPC and H200 H2D tests.

## Validation

Public-contract tests cover independent deadlines, stale free/acquire,
per-worker claims, duplicate completions, temporary objects, exact multi-L1
routing, rollback, cancellation/result handoff, ZMQ/gRPC identity, scheduler
cleanup, APC relookup, device-event failures and partial copy failures.

The standalone H200 validation under the experiment directory queues a real
asynchronous copy behind device work, expires its 1-second queue TTL, cancels
its remaining shares and attempts normal and forced deletion/clear. The GPU
must retain the original value `11`. The completion tracker then releases the
allocation automatically; the pool reuses the exact address for value `99`
without changing the GPU result. Three expired runs and three fresh controls
are required. This isolates source lifetime; it is not a full vLLM workload.

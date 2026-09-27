# L2 Prefetch Load Deadline with Recompute Fallback

Optional, default-off. Bounds how long a read-locked L2 prefetch may spend, from
the moment it enters the `PrefetchController` through queueing, L2 lookup, and L2
load. On expiry the caller receives the grouped cells already usable under the
request's `prefix` or `full` fetching policy and recomputes the rest, instead of
waiting for a slow or stuck L2. `NO_LOCK` warm prefetches have no waiting reader
and are never armed.

## Configuration

`StorageManagerConfig.prefetch_load_timeout` (CLI `--l2-prefetch-load-timeout`),
seconds. `None` (default) disables it; a value that is not a finite positive
number (including `nan` and `inf`) is rejected at
startup. It is threaded into `PrefetchController(l2_load_timeout=, clock=)`; the
clock is injectable so tests can advance time without sleeping.

This is a **server-side cache policy** and is independent of the client-side
transport bound `lmcache.mp.mq_timeout`. Set `mq_timeout` comfortably larger, so
the fallback result returns over a healthy RPC rather than the transport timing
out first.

## Two lifecycles

A request carries two small state machines:

- **Caller** (`CallerState`): `ACTIVE → PUBLISHED` (normal) or
  `ACTIVE → PUBLISHED_TIMEOUT` (fallback). The caller reads one immutable result.
- **Resource** (`ResourceState`): `ACTIVE → RETIRED` (normal) or
  `ACTIVE → DRAINING → RETIRED` (timeout). While `DRAINING`, late I/O still owns
  its write buffers and L2 read locks.

Normal completion publishes and retires in one step. A timeout must not: it
publishes the fallback once, releases the L1 read locks outside the published
set, and enters drain-only. Late adapter I/O then runs to completion and, only
then, finalizes its buffers, releases its L2 locks, and retires the request.
**A timed-out request never frees memory an adapter is still using, and never
re-publishes.** There is no cancellation: late I/O drains, it is not aborted.

Convergence is guarded by `_publish_result_once` (`CallerState`) and
`_retire_request_once` (`ResourceState`). `_poll_load_results` is the sole
per-adapter completion path: active requests admit successful cells with reader
locks, while draining requests admit them resident but unlocked; failed cells
are deleted in both cases.

## Scheduling (single loop thread)

Deadlines are stamped under the submission lock, so the *armed* entries of the
pending queue are deadline-monotonic. Unarmed `NO_LOCK` entries carry no deadline
and are interleaved freely, so both the poll bound and the expiry sweep scan past
them to the first armed entry rather than inspecting only the queue head. The
loop bounds its poll by the nearest deadline — the in-flight minimum or that
first armed queue entry — clamped to the poll interval and rounded up so a
sub-millisecond remainder does not busy-spin. A request that goes past its
deadline between the sweep and its own admission is re-checked when it is popped
and takes the queued fallback instead of starting L2 work. Once per iteration, **after** processing completions, an
expiry sweep fires due requests. Because expiry runs after completion processing
on the one thread, a completion signaled in the same wake is finalized first and
wins the race; only genuinely-pending adapters remain when the fallback is
computed. When disabled, nothing is armed and the loop never reads the clock.

## Fallback per phase

```
        submit ─▶ queue ─▶ LOOKUP ─▶ PLAN_AND_LOAD ─▶ finish
deadline in:      (a)        (b)          (c)
```

- **(a) Queued (never admitted or admitted late):** does the cheap synchronous
  L1 lock + policy plan, touches the selected keys for LRU parity with normal
  completion, skips only L2, and publishes that grouped L1-only result. An
  L1-resident hit is not discarded as a miss; the L2 portion is recomputed.
- **(b) Lookup:** only finalized L1 hits are usable (an L2 lookup that returned
  but did not load into L1 does not count). The late lookup still takes its L2
  read lock when it completes; the drain releases it.
- **(c) Plan-and-load:** usable = L1 hits ∪ loads already completed. Under
  `prefix`, `fold_unfold_grouped` applies the attention window of every key-group
  row, so a pending cell can truncate the common prefix. Under `full`, every
  completed cell is reported, including non-prefix cells. Pending adapters keep
  their write buffers and L2 locks until they complete.

## Observability

`report_status` exposes `deadline_timeout_count` and `draining_request_count`
next to the active/pending/phase counts, so an operator can see drainers holding
`max_in_flight` slots under a slow L2 (intended backpressure: queued requests
get a bounded-latency fallback and slots free once the slow I/O drains). One
`L2_PREFETCH_DEADLINE` event per timeout carries `phase` (queued/lookup/load),
budget, elapsed, and retained/missed chunk counts (low-cardinality; no
request-id label); the L2 metrics and logging subscribers consume it.

## Buffer ownership during the drain

The unit of ownership is the adapter *load task*, not the key. As soon as an
adapter's task returns, every cell in that adapter's grouped load plan is
resolvable — the ones that loaded *and* the ones that did not — so all of them
are finalized immediately by `_poll_load_results`. Only a still-pending
adapter's reservations stay held. Resolving just successful cells would pin a
failed cell's L1 write reservation for the whole of the slowest adapter's drain.

## Deliberately out of scope

No task-cancellation protocol; no dynamic wait-vs-recompute cost model; no change
to `NO_LOCK` warm prefetches (the deadline is unarmed for them) or to
`mq_timeout` semantics; no
storage-adapter interface change; a slot-free drain pool is possible follow-up.

**Shutdown and TTL are explicitly *not* strengthened by this change.**
`PrefetchController.stop()` joins its own loop thread and then releases the
requests it still tracks; it does not obtain any quiescence guarantee from the L2
adapters, because `L2AdapterInterface.close()` does not define one and most
in-tree adapters cancel rather than join their in-flight loads. That is the
behaviour on `dev` today and this change neither improves nor worsens it. The
drain is also unbounded, while the L1 write lock protecting a reserved buffer is
a TTL lock (default 600 s). Both are pre-existing lifecycle gaps tracked
separately, not solved here.

# L2 Prefetch Load Deadline with Recompute Fallback

Optional, default-off. Bounds how long a lookup-mode L2 prefetch may spend, from
the moment it enters the `PrefetchController` through queueing, L2 lookup, and L2
load. On expiry the caller receives the subset already usable under the trim
policy and recomputes the rest, instead of waiting for a slow or stuck L2.

## Configuration

`StorageManagerConfig.prefetch_load_timeout` (CLI `--l2-prefetch-load-timeout`),
seconds. `None` (default) disables it; a non-positive value is rejected at
startup. It is threaded into `PrefetchController(l2_load_timeout=, clock=)`; the
clock is injectable so tests advance time without sleeping.

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

Convergence is in three idempotent functions — `_publish_result_once` (guarded by
`CallerState`), `_retire_request_once` (guarded by `ResourceState`), and
`_finalize_write_reserved` (the single place a reserved L1 write buffer is
resolved, shared by normal completion, the deadline finalize, and the drain).

## Scheduling (single loop thread)

Deadlines are stamped under the submission lock, so the pending queue is
deadline-monotonic. The loop bounds its poll by the nearest deadline — the
in-flight minimum or the queue head — rounded up so a sub-millisecond remainder
does not busy-spin. Once per iteration, **after** processing completions, an
expiry sweep fires due requests. Because expiry runs after completion processing
on the one thread, a completion signaled in the same wake is finalized first and
wins the race; only genuinely-pending adapters remain when the fallback is
computed. When disabled, nothing is armed and the loop never reads the clock.

## Fallback per phase (reuses `build_trim_mask`)

```
        submit ─▶ queue ─▶ LOOKUP ─▶ PLAN_AND_LOAD ─▶ finish
deadline in:      (a)        (b)          (c)
```

- **(a) Queued (never admitted):** does the cheap synchronous L1 lock + trim,
  skipping only L2, and publishes that L1-only subset. An L1-resident hit is not
  discarded as a miss; the L2 portion is recomputed.
- **(b) Lookup:** only finalized L1 hits are usable (an L2 lookup that returned
  but did not load into L1 does not count). The late lookup still takes its L2
  read lock when it completes; the drain releases it.
- **(c) Plan-and-load:** usable = L1 hits ∪ loads already completed. Under
  `PREFIX` a still-pending key is a hole that truncates the prefix, so a later
  key that finished loading is not reported (it becomes resident-unlocked and is
  reused by a subsequent request). Pending adapters keep their write buffers and
  L2 locks until they complete.

## Observability

`report_status` exposes `deadline_timeout_count` and `draining_request_count`
next to the active/pending/phase counts, so an operator can see drainers holding
`max_in_flight` slots under a slow L2 (intended backpressure: queued requests
get a bounded-latency fallback and slots free once the slow I/O drains). One
`L2_PREFETCH_DEADLINE` event per timeout carries `phase` (queued/lookup/load),
budget, elapsed, and retained/missed chunk counts (low-cardinality; no
request-id label); the L2 metrics and logging subscribers consume it.

## Deliberately out of scope

No task-cancellation protocol; no dynamic wait-vs-recompute cost model; no change
to `WARM` (the deadline is unarmed for it) or to `mq_timeout` semantics; no
storage-adapter interface change; a slot-free drain pool is possible follow-up.

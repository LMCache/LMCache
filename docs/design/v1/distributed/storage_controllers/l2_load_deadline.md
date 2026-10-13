# L2 Prefetch Load Deadline

## Configuration

`StorageManagerConfig.prefetch_load_timeout` (CLI `--l2-prefetch-load-timeout`)
sets a finite positive budget in seconds for read-locked L2 prefetches,
including queueing, lookup, and load. `None` disables it; `NO_LOCK` warm
prefetches are exempt. The controller accepts an injectable monotonic clock.

On expiry, return the hits already usable under the fetching policy and let
the caller recompute the rest. This is independent of `lmcache.mp.mq_timeout`,
which bounds the client-side RPC wait. Set the RPC timeout larger so it does
not mask the fallback with a transport error.

## Caller and Resource Lifetimes

The caller and its outstanding I/O have separate lifetimes:

- Caller: `ACTIVE → PUBLISHED` or `ACTIVE → PUBLISHED_TIMEOUT`.
- Resources: `ACTIVE → RETIRED` or `ACTIVE → DRAINING → RETIRED`.

`_publish_result_once` publishes one immutable result and wakes waiters.
Timeout releases read locks outside that result but leaves pending loads'
staging buffers and L2 locks intact. Late completion finalizes each adapter's
successful and failed cells, releases its L2 locks, and retires the request
without publishing again.

Active loads acquire caller read locks. Draining loads use
`L1Manager.finish_prefetch`, admitting residents without read locks or a
redundant L2 store. Multi-L1 loads preserve adapter affinity; results report
read-lock owners in `l1_owners`. Drain cleanup never consumes those locks.

## Scheduling and Fallback

Deadlines are stamped under the submission lock in submission order. Polling
is bounded by the nearest in-flight deadline or first armed queued deadline,
skipping unarmed entries. Completion processing precedes the expiry sweep.
Admission rechecks the deadline before starting L2 I/O; an expired lookup
cannot transition to load. Disabled deadlines never read the clock.

Fallback depends on the phase:

- Queued: reserve L1 hits, apply the fetching policy, refresh their LRU recency,
  and publish without L2 I/O.
- Lookup: publish only L1 hits. Release existing L2 locks immediately and late
  lookup locks when their results arrive.
- Load: publish L1 hits plus completed loads. Pending adapters keep their
  reservations until completion, including when other adapters have failed.

`prefix` uses grouped attention windows to select the common prefix.
`full` retains columns completed by every row, including non-prefix columns;
`found_cells` separately reports L1/L2 discovery rather than usable hits.

Drainers still occupy `max_in_flight` slots. Queued requests can expire while
those slots remain occupied; admission resumes when the I/O drains.

## Observability and Limits

`report_status` exposes `deadline_timeout_count` and `draining_request_count`.
Each timeout emits `L2_PREFETCH_DEADLINE` with phase, budget, elapsed time, and
retained/missed chunk counts. Logging and metrics subscribers consume it;
metric labels use phase, not request IDs.

This does not add cancellation, change adapter interfaces, or alter RPC timeout
semantics. Existing shutdown and write-lock TTL limits remain: `stop()` joins
the controller, not adapter I/O, and an unbounded drain can outlive a write-lock
TTL. Adapter quiescence and TTL protection are separate lifecycle work.

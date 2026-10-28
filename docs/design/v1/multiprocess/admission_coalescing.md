# Admission Coalescing

## Problem

In multiprocess mode, KV retrieves complete asynchronously at staggered
times. The scheduler can only admit a request whose retrieve has
completed, and a step boundary typically sees at most one ready request,
so prefill tails are admitted one per step. Every prefill step carries a
large fixed cost, so spreading N tails over N steps wastes engine time
that admitting them together on fewer, fuller steps would save.

## Mechanism

When a retrieve completes, the worker adapter may hold it for one more
step so that it is reported together with a sibling retrieve and both
prefill tails land on the same step boundary.

Holding happens in `get_finished()`; the user enables it with a single
switch:

```
"lmcache.mp.admit_coalesce": true
```

Everything else is derived automatically (see Parameters).

## Parameters

| Symbol | Meaning | Source | Default |
|---|---|---|---|
| switch | enable the mechanism | user config | false |
| B | per-step token budget | vLLM `max_num_batched_tokens`, passed by the connector | — |
| alpha | fraction of B that counts as a full step | module constant | 0.75 |
| T | hold deadline, from retrieve submission | module constant | 3.0 s |
| K | number of ready retrieves to coalesce | module constant | 3 |
| d | typical retrieve duration (submission to report) | runtime EWMA | learned |

Per-request state: submission timestamp, prefill tail in tokens
(packed token count minus the cached range, read from the op because
new metadata fields do not survive `SchedulerOutput` serialization),
and the shared duration EWMA.

## Gate

```
on each get_finished() call (once per engine step):

    ready    = retrieves whose future completed this call
    inflight = submitted retrieves still pending        # retrieve_futures - ready

    report everything (no hold) if any of:
        K or more ready                                  # coalesce target reached
        no inflight sibling                              # nothing to wait for
        any ready tail >= alpha * B                      # fills a step alone
        sum(ready + inflight tails) < alpha * B          # cannot fill a step together
        now - oldest_ready_submit >= T                   # deadline expired
        d is unknown                                     # no duration estimate yet
        min over inflight of (d - waited) > T - (now - oldest_ready_submit)
                                                        # earliest sibling arrives too late

    otherwise: hold the ready retrieves for one step
               (they stay in retrieve_futures and are re-evaluated next call)

after reporting: update the duration EWMA from each reported
                 (non-held) request: submit -> report time
```

## Why each release condition

- **K reached / no sibling in flight** — with nothing left to wait for,
  holding is pure latency.
- **A tail alone fills a step** — a sibling cannot join an already-full
  step, so delaying the request cannot reduce the step count. Smaller
  tails are coalesced regardless of fill: merging two partial tails
  into one step still saves a whole step's fixed cost.
- **Deadline** — bounds the added latency of any held request. It is
  measured from submission because that is the wall time the request
  has already invested.
- **Sibling ETA** — the earliest in-flight sibling is estimated to need
  `d - waited` more seconds. If even it cannot arrive before the
  deadline expires, holding cannot produce a batch. Until `d` has at
  least one observation, the gate does not hold at all.

## Guarantees

- A completed retrieve is reported exactly once: held ids stay in
  `retrieve_futures` and are re-reported only when released.
- Nothing is dropped on failure paths: unhealthy drains report
  everything immediately.
- Disabled (default): the code path is unchanged.

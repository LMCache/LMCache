# L2 Adapter Circuit Breaker Design

This document describes the connection circuit breaker used by remote L2
adapters: what trips it, how it recovers, and the contract it offers callers
and operators.

## Motivation

A remote L2 backend can become briefly unreachable — a node-local proxy
restarts, a gateway stalls, a link drops. Every request issued during such a
window fails, and retrying them adds load without producing cache hits. The
breaker exists to stop paying that cost: after a few consecutive
connection-class failures the adapter stops dialling the backend and fails
requests locally.

The breaker must also *stop* stopping. L2 is an optimization, not a source of
truth, so an adapter that has given up permanently is strictly worse than one
that occasionally wastes a request: it silently drops the cache tier for the
lifetime of the process, and the only remedy is a restart. A breaker with no
recovery path converts a one-minute backend blip into a multi-day L2 outage.

## Which adapters have one

Two adapters talk to a remote service over a connection that can fail this
way and implement the breaker:

| Adapter | Failure classifier | Config prefix |
| --- | --- | --- |
| `S3L2Adapter` | `_is_connection_error(error_msg: str)` — matches `CONNECTION_REFUSED`, `SOCKET`, `DNS`, `TIMEOUT` in the CRT error string | `s3_breaker_*` |
| `BigtableL2Adapter` | `_is_connection_error(exc: BaseException)` — `DeadlineExceeded`, `ServiceUnavailable`, `TimeoutError`, `ConnectionError` | `breaker_*` |

Only connection-class failures count. Application-level errors — a 404, a
permission denial, a malformed key — are real failures for that request but
say nothing about reachability, so they never move the breaker.

## States

The breaker has two states plus a time-gated transition between them.

```
                  max_connection_failures consecutive
                      connection errors
   ┌──────────┐  ─────────────────────────────────►  ┌────────┐
   │  CLOSED  │                                       │  OPEN  │
   │ requests │                                       │ submits│
   │ dispatch │  ◄─────────────────────────────────   │ short- │
   └──────────┘   cooldown elapsed: next submit       │ circuit│
        ▲         re-arms and probes the backend      └────────┘
        │                                                  │
        └──────── probe succeeds: failures reset ──────────┘
                  probe fails (x max_connection_failures):
                  re-OPEN with the cooldown doubled
```

- **CLOSED** — normal operation. Each request outcome is fed to
  `_record_connection_outcome`. A success resets the consecutive-failure
  count and the cooldown; a connection-class failure increments the count.
- **OPEN** — `connection_disabled` is set and a `time.monotonic()` deadline
  (`_breaker_reopen_at`) is armed. Every submit short-circuits: store tasks
  complete unsuccessfully, lookup and load return all-zero bitmaps, `delete`
  is a no-op, and listing raises. Nothing touches the backend.
- **Re-arming** — the first submit after the deadline passes clears
  `connection_disabled` and the failure count, then proceeds normally. That
  request is the probe. If it succeeds the adapter is simply CLOSED again; if
  the backend is still broken, it and the next two failures re-OPEN the
  breaker with a longer cooldown.

There is deliberately no dedicated half-open state limiting traffic to a
single in-flight probe. Doing that correctly requires per-request epoch
tokens so that outcomes from requests dispatched before the trip are not
mistaken for the probe's result. The cost avoided is at most
`max_connection_failures` wasted requests per cooldown against a backend
that is still down — negligible for a best-effort cache tier, and cheaper
than the complexity.

## Backoff

The cooldown starts at the configured initial value and doubles on every
consecutive trip, capped at the configured maximum. Any successful request
resets it to the initial value, so an adapter that recovers does not carry
penalty from an unrelated earlier outage.

| Config field (S3 / Bigtable) | Default | Meaning |
| --- | --- | --- |
| `s3_breaker_initial_cooldown_s` / `breaker_initial_cooldown_s` | `5.0` | How long the breaker stays OPEN after the first trip. |
| `s3_breaker_max_cooldown_s` / `breaker_max_cooldown_s` | `300.0` | Upper bound for the doubled cooldown. |

Both must be positive, and the maximum must be greater than or equal to the
initial value — otherwise the backoff would shrink on each trip rather than
grow. `from_dict` rejects violations with `ValueError`.

`max_connection_failures` remains a class attribute (`3`) on both adapters.

## Thread safety

All breaker state (`_connection_failures`, `_connection_disabled`,
`_breaker_cooldown_s`, `_breaker_reopen_at`) is guarded by the adapter's
`self._lock`.

- `_breaker_blocks_locked()` **requires the caller to already hold the
  lock** — every submit path takes it to allocate a task id, so the check
  costs no extra acquisition. The `_locked` suffix marks the requirement.
- `_record_connection_outcome()` acquires the lock itself. It runs on the
  adapter's event-loop thread, while the submit paths run on caller threads.

Outcomes reported while the breaker is already OPEN are **ignored**. They
belong to requests dispatched before the trip, and the backend they describe
is the one already accounted for. Counting them would push
`connection_failures` far past `max_connection_failures` (making the field
meaningless as a diagnostic) and, once re-arming is possible, would let a
backlog of stale failures re-trip the breaker for an outage that has already
ended.

## Observability

`report_status()` reports the breaker on both adapters:

| Field | Meaning |
| --- | --- |
| `connection_failures` | Consecutive connection-class failures, `0` when healthy. Never exceeds `max_connection_failures`. |
| `connection_disabled` | `True` while OPEN. |
| `breaker_retry_in_seconds` | Seconds until the next probe is allowed; `0.0` when CLOSED. |

`breaker_retry_in_seconds` is what distinguishes "degraded and recovering on
its own" from "degraded and stuck", so it belongs in any alert or dashboard
built on adapter status. `is_healthy` is `False` for the whole time the
breaker is OPEN, including the cooldown.

State transitions are logged: each failure at `ERROR` with its
`(n/max)` count, the trip at `ERROR` with the cooldown, re-arming at `INFO`
with the endpoint being probed, and recovery at `INFO`.

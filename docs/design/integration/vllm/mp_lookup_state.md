# MP-mode lookup state

`LMCacheMPSchedulerAdapter` turns each vLLM request's lookup into one
`LookupState` (`lmcache/integration/vllm/mp_lookup_state.py`). The object owns
the whole exchange with every LMCache server: LOOKUP, QUERY_PREFETCH_STATUS
until the prefetch finishes, aggregation of the per-server hits, and release of
tail locks the request will never retrieve. RPC completions drive it through
future done-callbacks, so a QUERY_PREFETCH_STATUS goes out the moment its
server acks, not one scheduler step later.

## Adapter surface

| Adapter call | `LookupState` call |
|---|---|
| `maybe_submit_lookup_request` | create, `submit()` (sends LOOKUP to every server) |
| `check_lookup_result` | `on_scheduler_check()`, then `get_result()`; an unresolved empty-range lookup answers `start` |
| `cleanup_lookup_result` | `close()` and forget the object; an unresolved empty-range lookup is retired instead |
| `end_session` | `drain()` before END_SESSION; for a retired lookup, END_SESSION is deferred until it has a result |

`on_scheduler_check()` returns the servers whose RPC passed its deadline; the
adapter marks them unhealthy. Health events and the per-request dicts
(`_lookups`, `_retiring_lookups`) are
touched only on the scheduler thread.

## Empty-range lookups

A lookup whose `[start, end)` holds no chunk has nothing to load: vLLM's
prefix cache covers every whole chunk (the APC-covered lookup of
`docs/design/v1/multiprocess/modules/apc_covered_lookup.md` with `c0 == end`),
or the prompt is shorter than one chunk. With `lmcache.mp.save_unfull_chunk`
the lookup also covers the prompt's unfull tail, which may be loadable, so
`[start, end)` is empty only when the prompt ends on a chunk boundary that the
prefix cache covers. A serving server reports
`start` for it (the connector never sends `start > end`), so
`check_lookup_result` returns that at once
instead of None. Waiting would hold the request back a scheduler step (one
empty step on an idle engine, a whole forward pass under load) for a reply
that cannot change what the request loads. A declining server (KV load
policy, or no registered layout) replies 0; nothing is loaded either way, but
a check that already returned `c0` let the connector count `[0, c0)` as
stored, as it does for every served APC-covered lookup.

The LOOKUP is still sent: it refreshes the covered prefix's recency, emits the
lookup events, and records the lookup key that `end_session` uses to touch the
request's whole range (without it the server logs "Session ... not found" at
every END_SESSION; a prompt shorter than one chunk records no lookup key
either way, and the server logs "has no lookup ipc key" instead). Its server prefetch job is removed only by the
QUERY_PREFETCH_STATUS sent when the server acks the LOOKUP; the connector sends
no FREE for this range. A `close()` before that ack would suppress the query
and leave the job behind, which the early answer makes likely: the request is
admitted, and may finish, before the ack.

So `cleanup_lookup_result` retires an unresolved empty-range lookup instead of
closing it. A retired lookup keeps running on its callbacks; it holds no lock,
and nothing waits for it. Each later `maybe_submit_lookup_request` and
`end_session` call gives every retired lookup an `on_scheduler_check()`, so a
failed status query is re-sent and a missed RPC deadline marks the server
unhealthy, as for any lookup. While it has no result, `maybe_submit_lookup_request`
sends no new LOOKUP under its request id, because the old status query could
consume the new lookup's job; the request then reads a miss (0) and computes
the prompt itself. That happens only when a request is looked up again
within about one round trip of its early answer: a preemption right after
admission, or an APC shrink after a failed allocation. A
retired lookup is dropped once it has a result, or after the MQ timeout
without one (then closed, as `cleanup_lookup_result` closes other lookups).

`end_session` for a request whose retired lookup has no result yet sends no
END_SESSION: that lookup's LOOKUP or status query, handled after END_SESSION,
would recreate the session until the session TTL. The END_SESSION is sent by
the call that drops the retired lookup. With a local server the lookup
normally has its result by the time the request finishes: the request runs at
least one forward pass after admission, far longer than a round trip. The
deferral matters for a slow or remote server.

## Per-server state machine

Each server runs its own chain; no server waits for another's ack.

```
 submit()                  ack (own server)                                    reply int
 ───────► LOOKUP_IN_FLIGHT ─────────────────► QUERY_PREFETCH_STATUS_IN_FLIGHT ─────────► RESOLVED
               │                                │                  │        ▲
               │ deadline passed                │ deadline passed  │        │ re-sent by the callback if a check
               ▼                                │     reply None / │        │ ran since the query was sent,
           TIMED_OUT ◄──────────────────────────┘        exception ▼        │ otherwise by the next check
       (lookup result = 0, final)                            NEEDS_REQUERY ─┘
```

When every server is RESOLVED the result is `min(hits) * tokens_per_chunk`.
Each server that reported more than the minimum gets a `free_lookup_locks` for
`[min, hits)`, because those chunks are locked but will never be retrieved.

The result is **final** once set, whether by resolution, a timeout or
`close()`. A final state sends nothing more, and late replies are ignored.

## Why the rules are what they are

**QUERY_PREFETCH_STATUS only after the server's own ack.** LOOKUP and
QUERY_PREFETCH_STATUS share the server's request pool and can be reordered. A
QUERY_PREFETCH_STATUS handled before its LOOKUP finds no job and answers 0, a
spurious miss whose prefetch locks are never released (#4935). The race is
between two RPCs to the same server, so each server's ack is enough; there is
no cross-server barrier.

**At most one QUERY_PREFETCH_STATUS per server per scheduler check.** While a
prefetch runs the server answers `None` immediately, so re-sending on every
`None` would spin an RPC loop on the shared client receive thread and the
server pool.
The reply callback re-sends only if a check has run since that query was sent;
otherwise the next check sends it. Example timeline for one server:

```
check k      ─ marks q1 "checked"
q1 → None    ─ callback: q1 was checked → send q2 now
q2 → None    ─ callback: q2 not checked → NEEDS_REQUERY
check k+1    ─ sends q3
```

**At most one QUERY_PREFETCH_STATUS in flight per server.** Two concurrent
queries for one request race on the server's job: one gets the hit count, the
other `None`, and every later query gets 0.

**Deadlines count only time an RPC is in flight.** Each LOOKUP and
QUERY_PREFETCH_STATUS gets its deadline when it is sent, and a server in
NEEDS_REQUERY has nothing in flight, so it cannot expire. vLLM does not check
waiting requests while the running batch or the token budget is full, so a
deferred re-query can wait a long time; counting that wait would mark a server
unhealthy although it answered every query at once.

**Failures.** A failed LOOKUP is logged and still followed by a
QUERY_PREFETCH_STATUS; a server that never got the LOOKUP answers 0. A failed
QUERY_PREFETCH_STATUS is treated like `None` and never raised into the
scheduler. A query that fails on every attempt is re-sent once per check
without a limit; a server that is down stops being queried once the heartbeat
marks it unhealthy.

**Blocking mode.** With `lmcache.mp.nonblocking_lookup_status=false`,
`on_scheduler_check()` also waits, up to each RPC's deadline, for the
QUERY_PREFETCH_STATUS replies in flight. It never waits on a query that a
reply handled during that wait re-sent. LOOKUP acks are never waited for.

## Threading

Replies are handled on whichever thread completes the future: the shared ZMQ
client loop, a gRPC callback thread, or inline when the future is already done.
The scheduler thread calls everything else.

- One `threading.Lock` guards the state. Every send happens under it, and
  sends only enqueue, so `close()` cannot miss an RPC that is in flight.
- The lock is never held while registering a done-callback or waiting on a
  future. `add_done_callback` runs the callback inline on a completed future,
  and that callback takes the same lock.
- A handler acts only on its server's current future (`future is
  server.future`), so a reply is handled once whether its callback runs or
  `on_scheduler_check()` picks it up first. Callbacks only reduce latency;
  `on_scheduler_check()` handles any completed reply whose callback has not
  run, so it is never mistaken for a timeout.
- `drain()` waits on the raw futures. A completed future means the server has
  handled that RPC, which is the ordering END_SESSION needs.

## Known gaps

- `request_finished` calls `cleanup_lookup_result` before `end_session`, and
  cleanup forgets the state, so `end_session` has nothing to drain for an
  aborted mid-lookup request.
- A request id reused while the old state still has a QUERY_PREFETCH_STATUS
  in flight can let that query consume the new lookup's job. Connector paths
  reuse an id only after the old lookup resolved, or, for a retired
  empty-range lookup, after it has a result or timed out, so this does not
  occur today.
- Retired lookups advance only when the scheduler calls
  `maybe_submit_lookup_request` or `end_session`. On an idle engine a
  deferred END_SESSION waits for the next such call. With
  `lmcache.mp.nonblocking_lookup_status` set to False, that check also waits
  for a retired lookup's status reply in flight, as it does for any lookup.

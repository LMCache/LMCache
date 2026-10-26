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
| `maybe_submit_lookup_request` | wait for prior cleanup ACKs, then create and `submit()` |
| `check_lookup_result` | poll a deferred replacement, or `on_scheduler_check()` then `get_result()` |
| `cleanup_lookup_result` | close and transfer unconfirmed RPCs to the adapter |
| `end_session` | close and take unconfirmed RPCs; callbacks submit END after each server's ACKs |

`on_scheduler_check()` returns the servers whose RPC passed its deadline; the
adapter marks them unhealthy. Health events and the per-request dict are
touched only on the scheduler thread.

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
`close()`. A final state sends no new lookup or status RPCs. Late replies
still retire their raw handler acknowledgements.

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
- `close_and_take_pending_rpcs()` closes sending and transfers raw pending
  or failed futures once, under the same lock. A local deadline does not
  cancel a server handler. A failed future is not a successful handler ACK.

## Session end ordering

For each server, END_SESSION follows successful acknowledgements of every
admitted LOOKUP, QUERY_PREFETCH_STATUS and FREE_LOOKUP_LOCKS. This includes
the unequal-hit tail release and explicit releases from the adapter. A slow
server does not hold up END on another server.

The scheduler's `end_session()` remains a `None`-returning notification and
does not wait. Existing transport futures hold the completion callbacks
that submit END. The adapter retains no history of completed request IDs.
If cleanup arrives first, the adapter temporarily owns its unconfirmed
futures until the request-finished END notification. If END arrives first,
it takes the lookup and cleanup has nothing to transfer again.

An unhealthy flag only stops new cache work; it does not discard ordering
responsibility for already-admitted RPCs. A failed predecessor is logged
and prevents END on that server. This does not claim remote cleanup
succeeded, and it does not add transport recovery or an ID-reuse protocol.
Closing a request client during shutdown is likewise not a server ACK.

## Replacement lookup ordering

An APC shrink cleans up the partial lookup and requests a full-prefix lookup
with the same request ID. The partial lookup's result can be final while its
unequal-hit tail FREE is still running. If the replacement overtakes that
FREE, the server can resolve the old release against the replacement's session
metadata. A late QUERY_PREFETCH_STATUS can likewise consume the new lookup's
job when a caller cleans up before resolution.

`maybe_submit_lookup_request` therefore checks the prior lookup's captured
RPCs before sending a replacement to any server. Pending ACKs return False
and establish one deadline using the existing MQ timeout. The caller keeps
calling on scheduler ticks, as the vLLM connector already does before checking
each lookup result. Successful ACKs are pruned; no key or callback to send the
replacement is retained. True still means a LOOKUP was actually submitted.

`check_lookup_result` returns None while the replacement is deferred, even
after its prerequisites finish, until the caller next attempts submission.
An RPC failure or expired wait makes this replacement a stable cache miss.
Timeouts mark the outstanding servers unhealthy. Recovery or a late ACK does
not revive that failed replacement; cleanup or END must cancel it first.
Pending/failed RPCs remain available to END ordering, regardless of this
request-side deadline. Cleanup and END discard the deferred wait, so no
callback can launch a cancelled replacement.

The deadline is measured from the first blocked replacement attempt and is
not extended by repeated checks. Replies already complete at the next poll
are consumed before testing the deadline. The wait state and captured RPC map
are scheduler-thread-owned; completion callbacks only update their existing
future state. There is no completed-request history or additional timeout
configuration.

This orders calls issued through this adapter. It does not add attempt IDs to
the wire or make arbitrary out-of-order server handler calls safe. Other
clients must enforce the same ordering before reusing a live request ID, and
request IDs must not be reused after END while old handlers remain in flight.

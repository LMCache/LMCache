# APC-covered lookup skip (non-pin method)

vLLM's prefix cache (APC) often already holds the first `N` tokens of a
request. LMCache looked them up anyway — read-locking them and pulling them
from L2 into L1 — even though vLLM will never read them.

```
[========= APC already has this =========][==== LMCache loads ====]
0                                         c0                     end
 \__ locked + prefetched for nothing __/
```

There is no flag and no handshake: the connector sets the lookup key's
`start` to the chunk-aligned APC boundary `c0`, and the server skips
everything below the key's own range. A key starting at 0 behaves exactly as
before. The server and connector ship together, so the two sides cannot
disagree about which prefix was skipped.

## What changes

`LookupModule.lookup` honors `key.start`. For the covered prefix `[0, c0)`
it:

- **touches** it in **L1 and L2** — no bytes move, nothing is promoted. Both
  tiers matter: a real load refreshes recency in both, so touching only L1
  would leave the keys cold to L2 and get them evicted.
- **does not** lock or prefetch it. The prefetch covers `[c0, end)` only;
  hit counts shift back to absolute at status time. The hit-rate metrics
  do not shift: they cover `[c0, end)` only, and the covered prefix is
  reported separately as `apc_covered_tokens` on the event, so it never
  reads as an L1 hit.

So the prefix is no longer *promoted* into L1 on every lookup — that
promotion was the waste. Cache contents are unchanged; only placement is.

`end_session` then touches the **whole** request range in both tiers. The
split above gives one request two recencies — the covered prefix is touched
when the lookup runs, the retrieved tail when it is read — so an L1-only
refresh at the end would leave the prefix the older L2 entry and make the
most-shared data the first thing L2 evicts.

## The shrink race

A WAITING request's APC blocks are unprotected until `allocate_slots`, so the
APC hit can shrink while the lookup is in flight, leaving the frozen `c0`
covering a gap nobody owns.

**This branch does not pin.** It re-looks-up instead — the request is *not*
handed back to vLLM to recompute. Set the sticky `need_full_lookup` (so the
request never skips again — one shrink each, no churn, survives preemption),
record the abandoned lookup's locked range `[c0, ret)` in
`pending_free_ranges`, and submit the full lookup from token 0 in the same
scheduler step. LMCache serves the whole prefix `[0, ret')`; the request
keeps its cache hit, it only stops skipping.

**The stale locks ride the next FREE.** `LOOKUP` and `FREE_LOOKUP_LOCKS`
share the server thread pool, so with `max_cpu_workers > 1` a release sent at
shrink time could land *after* the replacement's `begin_lookup`. Instead the
connector keeps the stale range and passes it as `extra_ranges` on the
request's next `free_lookup_locks` — sent from `update_state_after_alloc`,
strictly after the replacement lookup's result was consumed, so there is
nothing left to race. Each extra range ends at its own lookup's hit boundary
and resolves self-contained: full-attention groups release `[start, end)`,
sliding-window groups translate from the range's `end`, never from the
replacement lookup's hit.

Pinning the covered blocks instead is implemented on the `core/skip-apc-pin`
branch. Non-pin won: no block-pool pressure, no coupling to vLLM's
`BlockPool`, and a shrink costs one wasted lookup rather than held GPU
blocks.

## The invariant

Read locks are anonymous refcounts **shared across prefix-sharing requests**,
so releasing one you never took corrupts another request. Every release goes
through `resolve_prefetched_obj_keys`, whose per-group range start is floored
to its key's `start` — and `free_lookup_locks` first floors the freed range
to the session lookup key's `start` — so a release cannot reach into the
skipped prefix.

## Flow

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant C as Connector
    participant L as LookupModule
    participant SM as Storage

    S->>C: get_num_new_matched_tokens(APC hit)
    Note over C: freeze c0
    C->>L: LOOKUP(key, start=c0)
    L->>SM: touch [0,c0) in L1+L2 - no lock, no prefetch
    L->>SM: reserve_read + prefetch [c0,end)

    S->>C: get_num_new_matched_tokens (re-poll)
    L-->>C: hit - async result read this poll
    Note over C: compare fresh APC hit to frozen c0
    alt APC hit shrank below c0
        Note over C: need_full_lookup = true; remember [c0,ret)
        C->>L: LOOKUP(key, start=0) - same step
        C-->>S: (None, True) - re-poll
        L->>SM: reserve_read + prefetch [0,end)
        S->>C: update_state_after_alloc
        C->>L: free_lookup_locks([0,hit), extra=[c0,ret))
        Note over C,L: LMCache still serves it - no recompute
    else normal
        C-->>S: need_to_load = hit - APC hit
        S->>C: update_state_after_alloc
        C->>L: free_lookup_locks([0, hit)) - floored to c0
    end
```

# APC-covered lookup skip (pin method)

When the serving engine's prefix cache (vLLM APC) already covers the first `N`
tokens of a request, the MP-server lookup no longer read-locks or L2-prefetches
the LMCache objects for that covered prefix — it only LRU-touches them (so they
stay warm) and prefetches just the uncovered tail `[N, end)`. Gated by
`lmcache.mp.skip_covered_lookup` (default off ⇒ identical to today).

The APC hit can shrink while the async lookup is in flight (a WAITING request's
matched blocks are unprotected until `allocate_slots`). **This branch handles that
race by pinning**: freeze the covered boundary `c0` and pin the covered GPU blocks
for the lookup window so a shrink cannot happen. An alternative *non-pin
(re-lookup)* method is implemented in `core/skip-apc-nonpin`; the two are weighed
in the standalone comparison doc (`apc_covered_lookup_comparison.md`).

## Design

Shared covered-skip mechanics:

- The server (`LookupModule.lookup`) reads `covered_chunks` from the key's
  `request_configs`, **touches** the covered prefix `[0, c0)` but does **not**
  read-lock or L2-prefetch it, and submits a prefetch for only the uncovered
  sub-range `chunk_hashes[c0:]`; hit counts are offset back to absolute at status
  time.
- The touch spans **both tiers** (`StorageManager.touch_cached_keys` -> `L1Manager`
  **and** every L2 adapter). This matters: a lookup that really loads a key
  refreshes recency in L1 *and* L2, so if the skip only refreshed L1 the covered
  keys would look cold to L2 and be evicted - losing data a non-skipping lookup
  would have kept, and lowering later hit rates. The touch moves no bytes and
  promotes nothing into L1; each tier ignores keys it does not hold.
- Consequence by design: the covered prefix is no longer *promoted* into L1 on
  every lookup (that promotion was the redundant read being removed). It stays
  wherever it already lives, which frees L1 for data that is actually retrieved.
  Cache **contents** are unchanged versus the feature being off; only the tier
  placement of the covered prefix differs.
- Every lock-release resolves through `resolve_prefetched_obj_keys`, whose
  per-group range start is clamped to `c0`, so a release never drops a lock the
  lookup did not take (which would corrupt a concurrent prefix-sharing request).

Pin-specific handling (this branch):

- **Acquire** (`_acquire_covered_resources`, at lookup submit): re-derive the
  covered GPU blocks from `request.block_hashes` via `BlockPool.get_cached_block`
  and `pool.touch` them. Best-effort — any missing precondition skips pinning and
  the bypass guard preserves correctness. Bounded by
  `lmcache.mp.max_pinned_apc_blocks` to avoid starving the pool.
- **Release** (`_release_covered_resources`, idempotent): `pool.free_blocks` of
  the recorded block ids on every terminal path (admitted / bypass / abort).
- **Bypass guard** (`_handle_covered_shrink`): if the APC hit shrank below `c0`
  and no pin is held (pool unbound / cache miss / budget exhausted), free the
  locks and recompute locally.

## Workflow — module interactions

Participants: **Scheduler** (vLLM), **Connector** (`LMCacheMPConnector`),
**Adapter** (`LMCacheMPSchedulerAdapter`), **LookupModule** (server),
**Storage** (`StorageManager`/`L1Manager`), **BlockPool** (vLLM GPU block pool).

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant C as Connector
    participant A as Adapter
    participant L as LookupModule
    participant SM as Storage
    participant BP as BlockPool

    S->>C: get_num_new_matched_tokens(num_computed = APC hit)
    Note over C: c0 = align(num_computed) then covered_chunks
    C->>BP: get_cached_block + touch [PIN covered blocks]
    C->>A: maybe_submit_lookup_request(covered_chunks)
    A->>L: LOOKUP(key, covered_chunks)
    L->>SM: touch covered [0,c0) in L1+L2 - no lock / no prefetch
    L->>SM: reserve_read + prefetch uncovered [c0,end)
    C->>A: check_lookup_result
    A-->>C: LookupOutcome(hit, stored)

    alt APC hit shrank below c0 and no pin held
        C->>A: free_lookup_locks
        C->>BP: free_blocks (unpin)
        C-->>S: (0, False) bypass, recompute locally
    else normal
        C-->>S: need_to_load = hit - num_computed
        S->>C: update_state_after_alloc
        C->>BP: free_blocks (release pin, idempotent)
        C->>A: free_lookup_locks([0, vllm_hit)) - server clamps to [c0,ret)
    end
```

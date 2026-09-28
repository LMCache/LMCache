# APC-covered lookup skip (non-pin / re-lookup method)

When the serving engine's prefix cache (vLLM APC) already covers the first `N`
tokens of a request, the MP-server lookup no longer read-locks or L2-prefetches
the LMCache objects for that covered prefix — it only presence-checks and
LRU-touches them (so they stay warm) and prefetches just the uncovered tail
`[N, end)`. Gated by `lmcache.mp.skip_covered_lookup` (default off ⇒ identical to
today).

The APC hit can shrink while the async lookup is in flight (a WAITING request's
matched blocks are unprotected until `allocate_slots`). **This branch handles that
race without pinning**: leave the covered GPU blocks unpinned and, if the APC hit
shrinks `c0 -> c0'`, re-look-up just the exposed gap `[c0', c0)`. This mirrors
vLLM's native offloading connector. An alternative *pin* method is implemented in
`core/skip-apc-pin`; the two are weighed in the standalone comparison doc
(`apc_covered_lookup_comparison.md`).

## Design

Shared covered-skip mechanics:

- The server (`LookupModule.lookup`) reads `covered_chunks` from the key's
  `request_configs`, **touches** the covered prefix `[0, c0)` in L1 (via
  `StorageManager`/`L1Manager`) but does **not** read-lock or L2-prefetch it, and
  submits a prefetch for only the uncovered sub-range `chunk_hashes[c0:]`; hit
  counts are offset back to absolute at status time.
- Every lock-release resolves through `resolve_prefetched_obj_keys`, whose
  per-group range start is clamped to `c0`, so a release never drops a lock the
  lookup did not take (which would corrupt a concurrent prefix-sharing request).

Non-pin-specific handling (this branch):

- No GPU-block pinning: `_acquire_covered_resources` / `_release_covered_resources`
  are no-ops, so there is no block-pool pressure.
- **Re-lookup on shrink** (`_handle_covered_shrink`): when the current aligned APC
  hit drops below the frozen `c0`, the completed lookup skipped the now-uncovered
  gap. Free the stale `[c0, ret)` locks, drop the lookup state, and reset the
  per-lookup tracker fields so the next scheduler poll re-submits with the current
  (smaller) covered boundary — the fresh lookup read-locks and fetches the gap
  from LMCache (kept retrievable by the covered-range touch). Returns `(None,
  True)` so the scheduler re-polls.
- Trade-off: soft guarantee — a gap chunk evicted before the re-fetch just
  shortens the hit (partial local recompute) rather than corrupting; no
  availability risk from held blocks.

## Workflow — module interactions

Participants: **Scheduler** (vLLM), **Connector** (`LMCacheMPConnector`),
**Adapter** (`LMCacheMPSchedulerAdapter`), **LookupModule** (server),
**Storage** (`StorageManager`/`L1Manager`).

```text
Legend: S=Scheduler  C=Connector  A=Adapter  L=LookupModule  SM=Storage

 S --get_num_new_matched_tokens(num_computed = APC hit)--> C
 C :  c0 = align(num_computed) -> covered_chunks   (no pin)
 C --maybe_submit_lookup_request(covered_chunks)--------> A
 A --LOOKUP(key, covered_chunks)------------------------> L
 L --peek + touch covered [0,c0)  (no lock / no prefetch)-> SM
 L --reserve_read + prefetch uncovered [c0,end)---------> SM
 C --check_lookup_result--------------------------------> A
 A --returns LookupOutcome(hit, stored)-----------------> C

 shrink c0 -> c0'  (gap exposed):
   C --free stale [c0,ret) locks + cleanup_lookup_result-> A
   C :  reset per-lookup state; re-submit gap [c0',c0) next poll
   C --> S :  (None, True)  re-poll
 normal:
   C --> S :  need_to_load = hit - num_computed
   S --update_state_after_alloc-------------------------> C
   C --free_lookup_locks([0, vllm_hit))----------------> A   [server clamps to [c0, ret)]
```

# PrefetchController: L1-first pin pass and the L1+L2 hit

Design of the prefetch flow in
`lmcache/v1/distributed/storage_controllers/prefetch_controller.py`.

## Invariant

> Every key counted toward the reported hit is **lock-held from the moment it
> is discovered** (L1 read lock, or L2 lookup lock carried by an L1 write
> reservation) until the request completes. There is never an
> observed-but-unlocked instant, so a concurrent eviction can shrink what gets
> discovered but can never invalidate a hit already reported.

Corollary: every fold (`build_trim_mask`) consumes **lock-acquisition
results**, never the lock-free peek.

Vocabulary: **pin** = take an L1 read lock; **skip** = never lock (stays
evictable); **loading** = L1 write reservation carrying an L2 lookup lock;
**unpin** = return the read lock.

## Key intervals and per-step actions

Sliding-window (SW) view; for full attention every in-L1 key inside the
hit is needed (no out-of-window segments). Keys in chunk order, drawn in
general position — the two windows can touch, overlap, or coincide (L2 may
extend the hit by less than a window, or not at all); segments may then be
empty or overlap, and the rightmost applicable segment's action wins.

```
              |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
                                            ^ L1 hit length               ^ L1+L2 hit length

out of L1-hit sw : in L1, behind the L1 hit's window — never needed again
in L1-hit sw     : the window that makes the L1 hit servable
out of L2-hit sw : between the L1 hit and the final (L1+L2) window
in L2-hit sw     : the final window; ends at the L1+L2 hit length
remaining        : past the L1+L2 hit
```

Actions per segment at each step:

```
                            |out of L1-hit sw|in L1-hit sw|out of L2-hit sw|in L2-hit sw| remaining  |
                                                          ^ L1 hit length               ^ L1+L2 hit length

pin pass    (SW in L1)        |      skip      |    pin     |  pin if in L1  |pin if in L1|pin if in L1|
step 1 plan (L2 candidates)   |       -        |     -      |   candidate    | candidate  | candidate  |
step 2 fold (SW in L1)        |      skip      |unpin@finish|  unpin@finish  |  keep pin  |unpin@finish|
step 3 rsrv (SW in L2 ∖ L1)   |       -        |     -      |       -        |  loading*  |     -      |
step 4      (L2 locks)        |      free      |    free    |      free      | keep plan  |    free    |
step 5 load (SW in L2 ∖ L1)   |       -        |     -      |       -        |  loading   |     -      |
finish      (SW in L2 ∖ L1)   |       -        |     -      |       -        |load→pinned |     -      |
finish      (SW in L1)        |      skip      |   unpin    |     unpin      |   pinned   |   unpin    |

(pins are held from the pin pass until finish — the folds only decide
 the retained set; every read lock outside it is released at finish.)
(*) all-or-nothing: any reservation failure (OOM or contention)
    abandons the whole L2 load and finishes with the L1 hit.
```

Notes:

- **Pin pass** locks everything L1 has *except* out of L1-hit sw. Those keys
  can never enter the final retained set — L2 results only extend the hit
  rightward, so the final window's left edge never moves back over them
  (monotonicity). They keep aging for eviction, which is what frees the heads
  of multi-round conversations.
- **Step 2** is where the window moves: if L2 extends the hit, the final
  window (in L2-hit sw) sits right of the L1-hit window, and pins left behind
  it fall out of the retained set (released at finish).
- **Finish is the single reconciler** (`_finish_request`), reached by every
  path — normal load completion, pure L1 hit, no adapters, all reservations
  dropped: failed loads delete their buffer, the final fold decides the
  retained set, every read lock outside it is released, all remaining L2
  lookup locks are returned, and the hit is reported if no earlier step did.
  The result is the retained bitmap: every key in it is pinned for the
  retriever. This works because lock state is *tracked on the request*
  (``l2_locked``, ``l1_pinned_keys``, ``write_reserved_keys``,
  ``hit_reported``) and releases subtract from it, so reconciliation is
  idempotent from any intermediate state.
- **WARM mode**: the pin pass only peeks (existence hint, no locks — WARM
  promises nothing to a retriever), and finalize leaves loaded keys unlocked
  (`finish_write` only).

## Why all-or-nothing on reservation failure

A reservation failure is either OOM (no L1 room for the load buffers) or
KEY_NOT_WRITABLE — the key appeared in L1 after the pin pass (typically a
concurrent request loading a shared prefix). Either way, the request
abandons the entire L2 load and finishes with the L1 hit: no re-fold, no
plan re-trim, no partial salvage. Promotion of a contended key
(``reserve_read`` after the failure) would be *safe* (lock-then-count
preserves the invariant) but reopens mid-flight mutation of the pinned set;
partial salvage kept three extra reconciliation steps alive for a marginal
win. Both were rejected for simplicity: the engine recomputes the unserved
suffix and re-stores it, so a truncated hit self-heals in one round.
Failures stay observable via the ``l1_oom`` / ``l1_contended`` reasons on
``L2_PREFETCH_FAILED``. Note the trade: a WARM request also aborts on first
contention, and partial-OOM no longer loads the pre-OOM prefix.

## Contract-anchoring tests

`tests/v1/distributed/test_prefetch_controller.py`:

- `TestConcurrentEvictionRace` — a deterministic evictor
  (`EvictionRacingL1Manager`, fires between L1 manager calls) cannot shrink
  the promised prefix; a key present in L1 or L2 at submit survives to the
  result.
- `test_l1_suffix_extends_l2_prefix` — L1 has chunks 2–4, L2 has 0–1 → hit is
  the union prefix (5), L2 loads only 0–1.
- `TestSlidingWindowClaims` — an out-of-sw key is deletable at the evictor's
  *first* opportunity (never pinned), without affecting the hit.

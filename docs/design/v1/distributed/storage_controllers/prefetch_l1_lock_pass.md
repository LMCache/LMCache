# PrefetchController: L1-first lock pass and the L1+L2 hit

Design of the prefetch flow in
`lmcache/v1/distributed/storage_controllers/prefetch_controller.py`.

## Invariant

> Every key **required for** the reported hit is **lock-held from the moment
> it is discovered** (L1 read lock, or L2 lookup lock carried by an L1 write
> reservation) until the request completes. There is never an
> observed-but-unlocked instant for a required key, so a concurrent eviction
> can shrink what gets discovered but can never invalidate a hit already
> reported.

Sliding-window carve-out: chunks behind the L1 prefix hit's own window are
released during the lock pass, before the L2 lookup. L2 only extends the hit
(`H >= H1`), so the final window's left edge `H - w >= H1 - w` never moves
left — a chunk behind the L1 window can never re-enter any window, so it is
never required. Releasing it early lets the allocator reclaim it while a slow
L2 lookup (RDMA today, a remote/network adapter later) is outstanding. The
release is restricted to the L1 prefix hit (`Bitmap(num_keys, H1 * stride)`):
an L1 *suffix* that L2 later connects into the hit is out of the L1-only hit,
not behind its window, so it stays locked.

Corollary: every fold (`build_trim_mask`) consumes **lock-acquisition
results** — the lock pass observes L1 by locking it (`reserve_read`), so
there is no separate observation that can go stale.

Vocabulary: **lock** = take an L1 read lock; **unlock** = return the read
lock; **loading** = L1 write reservation carrying an L2 lookup lock.

**LRU is decoupled from locking**: locking and unlocking never refresh
eviction recency (`on_l1_keys_read_finished` is a no-op for the eviction
policy). `_finish_request` explicitly touches the retained keys — the ones
the request actually serves — via `L1Manager.touch_keys`. Evictable keys are locked while
the request decides, released once the folds rule them out, and their
recency is never refreshed, so multi-round conversations age out their
heads naturally.

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

lock pass    (SW in L1)        |     unlock     |    lock    |      lock      |    lock    |    lock    |
step 1 plan (L2 candidates)   |       -        |     -      |   candidate    | candidate  | candidate  |
step 2 fold (SW in L1)        |   (unlocked)   | keep lock**|     unlock     | keep lock  |   unlock   |
step 3 rsrv (SW in L2 ∖ L1)   |       -        |     -      |       -        |  loading*  |     -      |
step 4      (L2 locks)        |      free      |    free    |      free      | keep plan  |    free    |
step 5 load (SW in L2 ∖ L1)   |       -        |     -      |       -        |  loading   |     -      |
finish      (SW in L2 ∖ L1)   |       -        |     -      |       -        |load→locked |     -      |
finish      (SW in L1)        |     unlock     |   unlock   |     unlock     |   locked   |   unlock   |

(out-of-L1-hit-sw chunks are released during the lock pass, before the L2
 lookup — step 2's unlock of that segment is then a no-op. Other locks the
 fold leaves outside the retained set are released right after step 2 —
 before the reservation — so the allocator can evict them for the load
 buffers; no-load finishes release them at finish. Finish reconciles
 idempotently from any of these.)
(*) all-or-nothing: any reservation failure (OOM or contention)
    abandons the whole L2 load and finishes with the L1 hit.
(**) the L1 hit's own window: kept as the fallback promise if the
     L2 load never lands.
```

Notes:

- **Lock pass** locks everything L1 has in one atomic `reserve_read`:
  observation and acquisition are the same call, so nothing can be counted
  without being held. Keys the folds rule out are released early (before
  the reservation) or at finish; locking is recency-neutral, so holding
  them briefly costs lock churn only, not LRU position.
- **Step 2** is where the window moves: if L2 extends the hit, the final
  window (in L2-hit sw) sits right of the L1-hit window, and locks left behind
  it fall out of the retained set (released at finish).
- **Finish is the single reconciler** (`_finish_request`), reached by every
  path — normal load completion, pure L1 hit, no adapters, all reservations
  dropped: failed loads delete their buffer, the final fold decides the
  retained set, every read lock outside it is released, all remaining L2
  lookup locks are returned, and the hit is reported if no earlier step did.
  The result is the retained bitmap: every key in it is read-locked for the
  retriever. This works because lock state is *tracked on the request*
  (``l2_adapter2readlocks``, ``l1_readlocks``, ``write_reserved_keys``,
  ``hit_reported``) and releases subtract from it, so reconciliation is
  idempotent from any intermediate state.
- **WARM mode**: locks like LOOKUP, but finish releases every read lock it
  holds (there is no retriever to hand them to) and leaves loaded keys
  unlocked (`finish_write` only).

## Why lock-all + explicit touch

An earlier revision peeked L1 lock-free and locked selectively, to avoid
LRU-bumping evictable chunks every round. That rationale was wrong on the
facts: `reserve_read` never refreshed recency — `finish_read`'s touch on
release did. Decoupling LRU from locking entirely (release is
recency-neutral; `_finish_request` touches the retained set, the moment
usefulness is actually decided) makes lock-all correct and deletes the
peek API, the selective-claim math, and the peek-to-lock misalignment corner.

`_lock_l1_keys` still observes L1 by locking it (`reserve_read` over every
key), preserving observation-is-acquisition. It then runs one L1-only
`build_trim_mask` and releases just the out-of-L1-hit-sw segment — the
provably-evictable subset guaranteed by the monotonicity carve-out above, bounded
to the L1 prefix hit. This is not the deleted lock-free selective claim: keys
are locked first, then a safe subset is released; nothing is counted while
unlocked.

## Why all-or-nothing on reservation failure

A reservation failure is either OOM (no L1 room for the load buffers) or
KEY_NOT_WRITABLE — the key appeared in L1 after the lock pass (typically a
concurrent request loading a shared prefix). Either way, the request
abandons the entire L2 load and finishes with the L1 hit: no re-fold, no
plan re-trim, no partial salvage. Promotion of a contended key
(``reserve_read`` after the failure) would be *safe* (lock-then-count
preserves the invariant) but reopens mid-flight mutation of the locked set;
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
  *first* opportunity (never locked), without affecting the hit.

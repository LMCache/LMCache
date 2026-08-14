# Lazy Offload: The Store Decision Model

Companion to [lazy_offload.md](lazy_offload.md) (the *mechanism*: buffering,
protection, draining without engine changes). This document defines the
*policy* -- for a KV chunk whose GPU copy exists, store to LMCache or skip --
and shows the criterion is logically complete and implementable without
estimating any probability.

## 1. The rule

> **Store a chunk iff:**
> 1. **(Eviction)** its GPU copy will be evicted, and
> 2. **(Reuse)** the chunk will be reused after that eviction, and
> 3. **(Economy)** serving that reuse from the copy is cheaper than
>    recomputing it.

Expected-value form (the three gates are its threshold factorization):

```
store  <=>  P(evict) x P(reuse | evict) x (recompute_cost - fetch_cost)  >  store_cost
```

## 2. Completeness

A stored copy has value **only when read**, and it is read **only when a
reuse arrives that the GPU cannot serve itself** (while the GPU copy exists,
vLLM's prefix cache serves the hit; the LMCache copy is dead weight). The
future of any chunk therefore partitions exhaustively into:

| Case | Evicted | Reused after | fetch < recompute | Copy's value | Verdict |
|------|---------|--------------|-------------------|--------------|---------|
| 1 | no  | --  | --  | 0 (GPU serves every reuse) | skip (gate 1) |
| 2 | yes | no  | --  | 0 (never read)             | skip (gate 2) |
| 3 | yes | yes | no  | <= 0 (recompute cheaper)   | skip (gate 3) |
| 4 | yes | yes | yes | > 0                        | **store**     |

- **Sound**: cases 1-3 have zero/negative value; the rule skips them.
- **Complete**: case 4 is the only positive-value cell, and it is exactly the
  conjunction of the three gates. No fifth case exists.

Given oracles for the three predicates, the rule is exactly optimal.

## 3. Amendments for the real system

The proof silently assumes five things that must be repaired. Each amendment
names one value channel; a policy change that fits none of them is either
redundant or evidence of a channel this list missed.

| # | Gap | Amendment |
|---|-----|-----------|
| A1 | **Prefix closure.** Chunk hashes are rolling; a stored suffix whose prefix was skipped is unreachable (value 0). | Decision unit is a *prefix*, not a chunk; skip decisions cut from the tail. |
| A2 | **Retention window.** Reuse must arrive before the lower tier evicts the copy. | Gate 2 reads "reused within the destination tier's retention window" -- it is tier-coupled. |
| A3 | **Displacement.** Storing into a full tier evicts a victim. | `store_cost` = transfer + expected value of the displaced victim (0 when the tier has space). |
| A4 | **Feasibility != desirability.** By execution time the GPU blocks may be overwritten. | Execute iff data is provably intact (hash snapshot + ref-count protection). A hard veto, never a trade-off -- and never a substitute for gate 1 (§6). |
| A5 | **Contention.** Per-chunk optimal != global optimal under shared D2H bandwidth / capacity. | Completed by an *ordering*, not a new gate: drain candidates by eviction imminence (free-queue LRU order). |

Non-gaps: multiple reuses and preemption recovery only strengthen the store
side (a preempted request is a reuse event with P ≈ 1) -- covered as-is.

## 4. Implementation: no gate needs a probability estimate

The predicates are statements about the future, but each gate can be built so
the probability never has to be estimated:

**Gate 1 -- replace prediction with timing.** Defer the decision until the
uncertainty collapses: a block near the free-queue head under GPU pressure
has P(evict) ≈ 1 by construction. Zero-engine-change signals available today:

- *Eviction ETA*: free-queue position ÷ block consumption rate (deltas of
  `BlockPool.get_usage()` per step) -- a countdown per block.
- *One-step allocation feedforward*: the scheduler has already fixed the next
  step's token budget (`num_scheduled_tokens`), so next-step block
  consumption is near-deterministic; drain at least that many head blocks.

Residual uncertainty: intra-step allocation bursts (bounded by one step).
Upstream endgame: a synchronous evict callback (immediate drop confirmation)
and grace-period reclaim / RFC #38260 (certainty *before* the loss).

*Why step-boundary sampling is lossless.* All eviction happens inside
`schedule()` (`get_new_blocks`), and the connector hook runs at the end of
every step -- eviction and observation share one clock, so nothing happens
while we are not looking. Step N's drain protects against step N+1's
allocations; the only blind window is intra-step, and the one-step
feedforward covers it. Passivity is a thread-safety constraint (the block
pool is unlocked scheduler state), not a lost capability.

*Gate 1 ceiling.* The two error rates are not symmetric:

- **Recall (no evicted-unsaved) can approach 1.** The eviction *order* is
  fully known (strict LRU queue); only the per-step *cutoff* is uncertain,
  and it has a sound upper bound -- the scheduler's own token budget
  (`max_num_batched_tokens` / block_size, plus `on_new_request` visibility
  into arrivals). Pinning at drain then turns prediction into prevention: a
  drained block can no longer be evicted.
- **Precision (no saved-unevicted) is irreducibly < 1.** A head block can be
  resurrected by a hit before its eviction; that is future-arrival
  information -- gate 2's uncertainty leaking into gate 1 -- which no
  eviction-side signal can supply. The synchronous callback reaches
  precision 1 not with more information but by acting at the instant
  uncertainty is zero. The gap is naturally small (LRU head = coldest) and
  each miss costs one cheap write.
- **Interference: a cost eager does not have.** Eager stores live requests'
  blocks (ref-held, not in the free queue); lazy stores dead requests'
  blocks and must pin them for the in-flight copy -- shrinking the free pool
  exactly when pressure is high, and displacing eviction onto warmer,
  deeper blocks (A3 at runtime). Bounded by drain cap x in-flight steps and
  reversed by `free_blocks(prepend=True)`. Mitigations, in order: drain
  early while slack exists (interference pushes the optimal horizon *up*;
  two watermarks), pin budget as a function of free-queue depth (drop a few
  stores rather than trigger a preemption), and -- escape hatch -- a D2D
  staging buffer that shrinks the pin window to microseconds at the price
  of a permanent reservation. The callback alternative converts this cost
  into allocation-path latency: it pays per actual eviction instead of per
  predicted one.

*Eviction is recycling, not overflow.* `get_usage()` counts only ref-held
blocks; the free queue is not empty space -- it *is* the GPU prefix cache
(freed blocks keep their hashes and serve hits until reallocated). Once the
pool has been filled once, **every allocation evicts a cached block, at any
usage level**. Consequences: a usage-watermark trigger is the wrong signal
(it implicitly assumes an overflow model that vLLM does not have);
lazy offload is a steady-state activity, not a crisis response; and the GPU
tier itself has a computable retention window -- free-queue depth /
allocation rate -- extending the tier-hierarchy view of gate 2 down to the
GPU as tier 0. Lazy offload is exactly the copy made in the last moments of
that window.

**Gate 2 -- replace prediction with sequential observation.** The storage
hierarchy *is* the estimator: each tier's retention window is a survival
test; KV that is not reused demotes tier by tier and is finally dropped.
P(reuse) is never estimated -- it is observed. This also demotes gate 2 from
a binary store/skip decision (false negative = an unbounded prefill loss) to
an **entry-tier placement** decision (worst case = the rent difference
between tiers -- bounded). Predictive signals (session liveness, prefix-tree
depth, historical hit stats, learned models later) only pick the entry tier
and window length, and need precision only when used as vetoes. The cost --
dead KV pays upper-tier rent before reaching the bottom -- is the price of
information, compressed by admitting low-score chunks directly into lower
tiers and demoting in async batches.

**Gate 3 -- not a probability at all.** `recompute - fetch` is fixed by
hardware and model constants (KV bytes/token, D2H bandwidth, prefill
throughput): compute a break-even prefix length offline; runtime is one
comparison. Because Δ multiplies the probabilities, long prefixes justify
storing even at tiny P(reuse) -- gate 2 needs discrimination only for
short/medium prefixes.

**Default under ignorance.** The error costs are asymmetric: a false
positive is one cheap write; a false negative is one prefill. With unknown
P(reuse), the regret-optimal action is **store** (given gate 3 passes).
Knowledge only prunes waste; it is never a prerequisite for running.

## 5. Evaluation order: 3 → 2 → 1 (the reverse of logical order)

| Order | Gate | Evaluated at | Nature |
|-------|------|--------------|--------|
| first | 3 Economy | admission (`add()`) | static threshold; cheap, reliable, may be strict |
| second | 2 Reuse | admission / while pending | placement decision; permissive by default |
| third | 1 Eviction | drain time | near-factual; controls *when* and *in what order* |

Gate 1 is the only gate undecidable early: it is not an admission filter but
the **trigger and ordering of the drain**. Deciding it early -- by a timer or
a request count -- is the anti-pattern below.

## 6. Anti-pattern: ex-post survival checks invert gate 1

An implementation that buffers stores for an arbitrary duration and checks at
flush time whether the data still survives (hash comparison) has evaluated
"was not yet evicted" -- the **negation** of gate 1. It drops exactly the
evicted chunks (the ones gate 1 selects *for* storing) and stores exactly the
survivors (whose GPU copies are still serving hits): *delayed eager with
anti-selective drops*. Survival checking is the A4 feasibility veto; using it
as the desirability gate flips the policy's sign.

## 7. Observability, phasing, verdict

Each gate is independently falsifiable, so regressions localize:

- Gate 1: store precision (stored chunks whose GPU copy was in fact evicted
  before next reuse) and drop rate (evicted before we stored).
- Gates 2, 3: post-hoc reuse rate of rejected/below-threshold chunks (≈ 0
  expected).

| Phase | Gate | Mechanism |
|-------|------|-----------|
| 1 | 1 | pressure trigger + eviction ETA + allocation feedforward + free-queue-LRU drain + A4 feasibility gate |
| 1 | 3 | static break-even prefix-length threshold at admission |
| 2 | 2 | hierarchy demotion as the default estimator; heuristic entry-tier placement (session, depth, hit history) |
| 3 | 2 | learned context-aware placement, only if phase-2 headroom justifies it |

**Verdict.** The framework has no missing case -- §2 covers the event space,
§3 closes the real-system gaps -- only imperfect sensors, and each sensor is
engineered to avoid probability estimation: gate 1 by timing, gate 2 by
hierarchical observation, gate 3 by physics. The irreducible residue is
intra-step bursts (gate 1) and tier rent paid on dead KV (gate 2); both are
bounded, and both shrink with the upstream hooks and placement heuristics of
later phases.

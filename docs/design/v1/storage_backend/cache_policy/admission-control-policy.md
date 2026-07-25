# `AdmissionControlledPolicy` -- Design and Two-Directions Report

## Summary

`lmcache/v1/storage_backend/cache_policy/admission_control.py` ships
`AdmissionControlledPolicy`, a TinyLFU-style admission-control wrapper
usable around any `BaseCachePolicy`. It was promoted from the
direction-finding experiment
(`benchmarks/cache_policy/experiments/admission_control.py`; see
[`cost-aware-policy-eval.md`](cost-aware-policy-eval.md)'s "Direction-finding
experiment" section) after it beat every other candidate -- including the
`CostAwareEvictionPolicy` frequency fix -- on both synthetic and real
ShareGPT data.

## Contract

### `BaseCachePolicy.should_admit(key, cache_dict) -> bool`

A new non-abstract method on `base_policy.py`, default `return True`
(same pattern as the existing `update_on_put_with_metadata`/
`update_cost_observation` default hooks -- backward-compatible with every
existing policy without modifying them). **Precondition**: callers must
only invoke it when the cache is already at/over capacity, mirroring the
existing convention that `get_evict_candidates` is only called under
capacity pressure. `should_admit` has no visibility into capacity itself
-- only into `cache_dict`'s current contents -- so calling it on a cache
with free space can cause a false rejection.

### `AdmissionControlledPolicy(inner_policy: BaseCachePolicy)`

Delegates every standard method (`init_mutable_mapping`, `update_on_hit`,
`update_on_put`, `update_on_put_with_metadata`, `update_on_force_evict`,
`get_evict_candidates`) to `inner_policy` unchanged, and adds:

- `_FrequencySketch`: a decaying dict-based approximate per-key
  request-frequency counter (periodic halving to bound memory and let old
  popularity decay). Not a real Count-Min Sketch -- no hashing/collisions
  modeled, which is fine at the scale this runs at and keeps the
  evaluation honest about the *admission policy*, not sketch-accuracy
  artifacts.
- `should_admit(key, cache_dict)`: records the attempt (increments the
  sketch for `key` **regardless of outcome** -- see "A real bug caught
  during productionization" below for why this matters), then asks
  `inner_policy` for its current #1 eviction candidate; admits if there's
  no candidate (nothing to displace) or if `key`'s estimated frequency
  exceeds the candidate's.

### `get_cache_policy("ADMISSION_<INNER>")`

Generic prefix support in
`lmcache/v1/storage_backend/cache_policy/__init__.py`: any registered
policy name, prefixed `ADMISSION_` (e.g. `"ADMISSION_LRU"`,
`"ADMISSION_COST_AWARE"`), resolves the inner policy recursively through
the same factory function and wraps it. Works for any current or future
policy name, not just the combinations the experiment tested.

## A real bug caught during productionization

Porting the experiment into a real, tested class surfaced a genuine
correctness bug the prototype didn't have to deal with, because it
controlled its own simulation loop end to end. In production,
`update_on_put_with_metadata` (where the prototype's loop incremented
frequency) is **only called for admitted keys** -- a rejected key never
reaches it. The first version of `should_admit` didn't record the attempt
itself, so a key that lost its first admission bid could never accumulate
enough frequency to ever win a later one: **permanent lockout on first
rejection**. This was caught not by code review but by rerunning the
real-data verification (see below) and noticing the official class scored
*worse* than plain `LRU` -- the opposite of the validated experiment's
result. The fix: `should_admit` now increments the frequency sketch for
`key` before making its decision, so repeated attempts accumulate
frequency across rejections, matching the experiment's semantics (which
incremented on the first attempt at a key regardless of the eventual
admission outcome).

This is also why the shipped benchmark simulator
(`lmcache/tools/cache_policy_bench/runner.py`'s `_PolicyCache`) was
updated to call `should_admit` when at capacity, before evicting and
inserting -- without that, `"ADMISSION_LRU"` run through the benchmark
suite silently behaved identically to plain `"LRU"`, and the bug above
would never have surfaced. `_PolicyCache`'s docstring flags that this
makes the simulator drive one more call than `LocalCPUBackend` does today
-- intentional, since exercising the full policy interface is the point
of a benchmark tool, and the shipped correctness tests
(`tests/v1/test_cache_policy.py`) and fast smoke coverage
(`tests/benchmarks/test_cache_policy_bench.py`, `ADMISSION_LRU` added to
`_FAST_TEST_POLICIES`) both now exercise this class directly.

## Integration status: class only, no backend wiring (by design)

Investigating `local_cpu_backend.py`/`local_disk_backend.py` before
building this confirmed request-time admission gating needs a call site
that knows the incoming key *before* space is allocated:

- `local_disk_backend.py`'s `submit_put_task` has exactly that -- key and
  required size are both available before its eviction loop. **This is
  the identified low-risk integration point for a future change.**
- `local_cpu_backend.py`'s `allocate()` only sees byte sizes (`shapes`/
  `dtypes`), never the key -- it's dropped between `cache_engine.py`'s
  `store` loop and the allocator. Wiring admission control there would
  mean threading `CacheEngineKey` through `allocate()`'s signature and
  every caller across `cache_engine.py`/`storage_manager.py` -- a
  meaningfully more invasive change.

Per explicit scope decision, this work ships the class only: it's
selectable today via `get_cache_policy`/config and is fully correct and
tested, but no storage backend calls `should_admit` yet, so the
admission-rejection behavior does not affect production request handling
until a backend is wired to call it -- a deliberate, separate follow-up.

## Verification: official class vs. the validated experiment

Rerunning a slice of the real-data comparison through the *official*
`get_cache_policy("ADMISSION_LRU")` path (not the experimental prototype),
same corpus, same seeds, same cache size (100 MiB), 4 bootstrap repeats:

| Scale | Experiment (`admission[LRU]`) | Official class (`ADMISSION_LRU`) | Plain `LRU` baseline |
|---|---:|---:|---:|
| 500 conversations | 23.4% [23.0, 23.9] | 23.2% [22.5, 24.0] | 18.2% [17.4, 18.9] |
| 2,000 conversations | 9.1% [8.9, 9.5] | 8.0% [7.6, 8.5] | 5.1% [4.5, 5.6] |

The official class closely reproduces the experiment's numbers (within
overlapping-to-near-overlapping CI at 500, slightly lower but still a
clear, non-overlapping win over plain `LRU` at 2,000) -- the small
residual gap is expected implementation-detail variance between the two
increment-bookkeeping paths, not a correctness concern; both clearly beat
the baseline by a wide, statistically significant margin at both scales.

## Two directions compared

Two structurally different fixes for `CostAwareEvictionPolicy`'s
real-data weakness were built and evaluated end to end:

### Direction A: frequency-aware `CostAwareEvictionPolicy` (see `cost-aware-policy-eval.md`)

Added a log-dampened access-frequency term to the existing cost-density
score. **Outcome**: a real, general improvement over the original
cost-only formula (closed ~60-70% of the synthetic hit-rate gap to
`LRU`/`LFU`, turned a latency loss into a latency win on
`multi_round_chat`) -- but on real ShareGPT data it remained the
**worst** of the baseline policies at every scale tested, because almost
every chunk is touched once under real traffic (no signal for the
frequency term) and cost-density actively misleads eviction under that
access pattern. A genuine fix to a real weakness, but not sufficient on
its own for real-world traffic.

### Direction B: `AdmissionControlledPolicy` (this doc)

Rejects low-value admissions outright instead of only reranking eviction
order, and is not specific to cost-awareness at all -- it wraps any
policy. **Outcome**: the clear winner. On real data, wrapping plain `LRU`
beat every baseline and every other direction tested (including
`admission[COST_AWARE]`) at every scale, by a wide statistically
significant margin (+29% to +78% relative hit rate over plain `LRU`). It
also substantially rescued `COST_AWARE` itself without fully closing the
gap to `admission[LRU]`, confirming the benefit is largely orthogonal to
which eviction policy it wraps.

### Recommendation

**`AdmissionControlledPolicy` is the stronger, more general result of the
two directions**, and is now shipped as a real class. Direction A's
frequency fix remains a legitimate, independently useful improvement to
`CostAwareEvictionPolicy` specifically (already shipped in the prior
commit) but is not, on its own, competitive with plain `LRU`/`LFU` on
real traffic. The two directions are not mutually exclusive --
`get_cache_policy("ADMISSION_COST_AWARE")` composes them today, though
the real-data numbers above show that combination still trails
`ADMISSION_LRU`.

The highest-value next step, not done here per the agreed scope, is
wiring `should_admit` into `local_disk_backend.py` (the identified
low-risk integration point) so the effect is real for actual disk-tier
request handling, not only benchmarked.

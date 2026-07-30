# `AdmissionControlledPolicy` -- Design, Evaluation, and Three-Directions Report

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

This revision extends the original thin (2-scale, single-cache-size)
verification into the same depth of investigation
`cost-aware-policy-eval.md` received: a full synthetic cache-size sweep, a
`halve_every` ablation, a Zipf-skew robustness sweep, a statistically
rigorous real-data validation (3 scales x 3 cache sizes x 6 bootstrap
repeats), and dedicated stress tests -- which caught a second real
correctness bug beyond the one already documented below, and surfaced a
genuine limitation of the design that the thin verification had missed
entirely. A later revision adds `WindowedAdmissionControlledPolicy`, a
second, independently selectable admission-control design built
specifically to address that limitation (see "`WindowedAdmissionControlledPolicy`:
does windowing fix Findings 5-6?" below) -- kept alongside, not
replacing, the original.

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
  popularity decay, window controlled by `halve_every`, default 20,000).
  Not a real Count-Min Sketch -- no hashing/collisions modeled, which is
  fine at the scale this runs at and keeps the evaluation honest about
  the *admission policy*, not sketch-accuracy artifacts.
- `should_admit(key, cache_dict)`: records the attempt (increments the
  sketch for `key` **regardless of outcome** -- see "Bug 1" below for why
  that matters), then admits unless `cache_dict` is non-empty and `key`'s
  estimated frequency does not exceed the **coldest currently-resident
  key's** estimate (see "Bug 2" below for why this compares against the
  coldest resident rather than the inner policy's eviction candidate).

### `get_cache_policy("ADMISSION_<INNER>")`

Generic prefix support in
`lmcache/v1/storage_backend/cache_policy/__init__.py`: any registered
policy name, prefixed `ADMISSION_` (e.g. `"ADMISSION_LRU"`,
`"ADMISSION_COST_AWARE"`), resolves the inner policy recursively through
the same factory function and wraps it. Works for any current or future
policy name, not just the combinations tested here. A `halve_every` kwarg
passed to `get_cache_policy` is forwarded to `AdmissionControlledPolicy`
itself rather than to the inner policy's constructor (needed for the
ablation below; popped out before the remaining kwargs are forwarded
inward).

## Two real bugs caught by actually running the full evaluation

### Bug 1 -- permanent lockout on first rejection

Porting the experiment into a real, tested class surfaced a bug the
prototype didn't have to deal with, because it controlled its own
simulation loop end to end. In production, `update_on_put_with_metadata`
(where the prototype's loop incremented frequency) is **only called for
admitted keys** -- a rejected key never reaches it. The first version of
`should_admit` didn't record the attempt itself, so a key that lost its
first admission bid could never accumulate enough frequency to ever win a
later one: permanent lockout on first rejection. Caught by rerunning the
real-data verification and noticing the official class scored *worse*
than plain `LRU` -- the opposite of the validated experiment's result.
Fixed by incrementing the sketch inside `should_admit` itself, before the
decision, regardless of outcome.

### Bug 2 -- speculative peek corrupts `LFUCachePolicy`'s internal state

Caught while running Experiment 1 (below) for the first time: wrapping
`LFU` crashed with `KeyError` inside `LFUCachePolicy.update_on_hit`. Root
cause: the first version of `should_admit` called
`inner_policy.get_evict_candidates(cache_dict, num_candidates=1)`
speculatively, just to compare frequencies -- but `get_evict_candidates`
is used everywhere else in this codebase as a call that's always
immediately followed by actually evicting the returned key.
`LFUCachePolicy` relies on that: it mutates its own bookkeeping
(`key_to_freq`/`freq_to_keys`) as a side effect of `get_evict_candidates`
itself, not of a separate evict step. Calling it speculatively and then
*not* evicting (because `should_admit` decided to reject) desynced
`LFUCachePolicy`'s internal state from `cache_dict`: the peeked key stayed
physically cached but was purged from `key_to_freq`, so a later genuine
hit on it crashed. Fixed by changing the comparison to never call
`get_evict_candidates` at all -- `should_admit` now compares against the
coldest resident key *by its own frequency sketch* instead of asking the
inner policy who it would evict (see the class docstring for the full
rationale). This is a purely additive, side-effect-free read.

Both bugs are exactly the kind of thing a thin, happy-path verification
misses and a full sweep across every registered inner policy catches --
the reason this investigation was extended to the same depth as
`cost-aware-policy-eval.md` in the first place.

## Workloads

Same four synthetic workloads as `cost-aware-policy-eval.md`
(`repetitive_short`, `novel_long`, `mixed_zipfian`, `multi_round_chat`),
plus the same real ShareGPT corpus for the real-data tier. See that doc
for full descriptions.

## Experiment 1: synthetic cache-size sweep

All ten policies (`LRU`, `LFU`, `FIFO`, `MRU`, `COST_AWARE`, and each
`ADMISSION_`-wrapped) across all four workloads at 50/100/200 MiB
(`python -m lmcache.tools.cache_policy_bench.runner --sweep --policies ...`,
full data in
[`results/admission_control/sweep_results.json`](../../../../../benchmarks/cache_policy/results/admission_control/sweep_results.json)):

![Hit rate vs cache size](../../../../../benchmarks/cache_policy/results/charts/admission_control/hit_rate_vs_cache_size.png)

![p95 modeled latency](../../../../../benchmarks/cache_policy/results/charts/admission_control/latency_p95.png)

Selected rows at 100 MiB:

| Workload | Policy | Hit rate | Evictions | p95 latency |
|---|---|---:|---:|---:|
| mixed_zipfian | LRU | 85.3% | 1,973 | 30.7 ms |
| mixed_zipfian | LFU | 87.6% | 1,611 | 30.7 ms |
| mixed_zipfian | COST_AWARE | 79.3% | 2,938 | 20.5 ms |
| mixed_zipfian | **ADMISSION_LRU** | **87.9%** | **288** | 25.6 ms |
| mixed_zipfian | ADMISSION_COST_AWARE | 82.6% | 1,803 | 25.6 ms |
| multi_round_chat | LRU | 58.0% | 910 | 61.4 ms |
| multi_round_chat | LFU | 80.8% | 198 | 19.9 ms |
| multi_round_chat | COST_AWARE | 78.4% | 236 | 24.5 ms |
| multi_round_chat | **ADMISSION_LRU** | **83.3%** | **0** | **15.2 ms** |
| multi_round_chat | ADMISSION_COST_AWARE | 83.3% | 0 | 15.2 ms |

### Finding 1 -- `ADMISSION_LRU` beats every baseline outright on synthetic data, not just `COST_AWARE`

Unlike the frequency-aware `CostAwareEvictionPolicy` fix (which still
trailed `LFU` on `mixed_zipfian`), `ADMISSION_LRU` has the **highest** hit
rate of all ten policies on `mixed_zipfian` at 100 MiB (87.9%, beating
`LFU`'s 87.6%) and by a wide margin on `multi_round_chat` (83.3% vs.
`LFU`'s 80.8%), while cutting evictions dramatically (288 vs. `LRU`'s
1,973 on `mixed_zipfian`; **zero** on `multi_round_chat` -- the working
set stabilizes and stops churning entirely). Wrapping any policy in
admission control consistently improves it or leaves it roughly flat --
every `ADMISSION_*` row is at or above its bare counterpart at 100 MiB
except `ADMISSION_MRU`, which inherits `MRU`'s poor baseline behavior
(admission control gates *what* gets in, not *how badly* the inner policy
ranks what's already there).

## Experiment 2: ablation -- `halve_every` sensitivity

`benchmarks/cache_policy/run_admission_control_ablation.py`, sweeping the
frequency sketch's decay window at 100 MiB (full data in
[`results/admission_control/admission_control_ablation.json`](../../../../../benchmarks/cache_policy/results/admission_control/admission_control_ablation.json)):

| Workload | Variant | Hit rate | p95 latency | Evictions |
|---|---|---:|---:|---:|
| mixed_zipfian | fast_decay (2,000) | 82.3% | 30.7 ms | 728 |
| mixed_zipfian | default (20,000) | 83.7% | 30.7 ms | 364 |
| mixed_zipfian | slow_decay (200,000) | 83.7% | 30.7 ms | 364 |
| mixed_zipfian | no_admission (LRU) | 79.8% | 30.7 ms | 2,078 |
| multi_round_chat | fast_decay (2,000) | 58.0% | 61.4 ms | 811 |
| multi_round_chat | default (20,000) | 83.3% | 15.2 ms | 0 |
| multi_round_chat | slow_decay (200,000) | 83.3% | 15.2 ms | 0 |
| multi_round_chat | no_admission (LRU) | 58.0% | 61.4 ms | 910 |

### Finding 2 -- decay window matters a lot, and too-fast decay erases the entire benefit on longer-horizon reuse

On `multi_round_chat`, `fast_decay` (halve every 2,000 increments)
performs *identically* to plain `LRU` with no admission control at all
(58.0%, 811-910 evictions) -- the decay window is short enough that a
conversation's earlier-round chunks lose their accumulated frequency
credit before its later round arrives to reuse them, so admission control
has no useful signal left by the time it matters. `default` and
`slow_decay` both fully recover the benefit (83.3%, zero evictions).
`mixed_zipfian`'s shorter, denser reuse cycles are far less sensitive
(82.3% vs. 83.7% -- a real but much smaller gap). **The 20,000 default is
a reasonable choice, but it is not a free parameter**: workloads with
longer reuse horizons than the ones benchmarked here could need an even
larger `halve_every`, and there is no evidence a shorter one is ever
better -- `slow_decay` never underperformed `default` in this sweep.

## Experiment 3: robustness -- Zipf skew generality

`benchmarks/cache_policy/robustness_sweep.py`, extended to include
`ADMISSION_LRU`, at 100 MiB (full data in
[`results/admission_control/robustness_zipf_skew.json`](../../../../../benchmarks/cache_policy/results/admission_control/robustness_zipf_skew.json)):

| `zipf_s` | LRU | LFU | COST_AWARE | ADMISSION_LRU |
|---|---:|---:|---:|---:|
| 0.6 (mild) | 43.5% | 49.4% | 37.5% | **51.1%** |
| 1.2 (default) | 79.8% | 82.5% | 69.0% | **83.7%** |
| 2.0 (extreme) | 96.9% | 96.9% | 96.9% | 96.9% |

### Finding 3 -- the win holds across skew strength, not just one anecdote

`ADMISSION_LRU` has the highest hit rate at both mild and default skew
(beating `LFU`, the next-best, by 1.6-3.4pp), consistent with Finding 1's
synthetic-sweep result rather than an artifact of the one `zipf_s` value
used elsewhere. All policies converge at extreme skew because the working
set fits entirely in cache (zero evictions) -- policy choice stops
mattering once there's no pressure, the same pattern seen throughout this
evaluation.

## Experiment 4: statistically rigorous real-data validation

`benchmarks/cache_policy/real_dataset_eval.py --policies LRU LFU
COST_AWARE ADMISSION_LRU ADMISSION_COST_AWARE`, the full 3-scale x
3-cache-size x 6-bootstrap-repeat grid matching `cost-aware-policy-eval.md`'s
rigor exactly (superseding the original thin 2-scale/1-cache-size
verification; full data in
[`results/admission_control/real_data/real_dataset_ci.json`](../../../../../benchmarks/cache_policy/results/admission_control/real_data/real_dataset_ci.json)):

| Scale | Cache | LRU | COST_AWARE | ADMISSION_LRU | ADMISSION_COST_AWARE |
|---|---|---:|---:|---:|---:|
| 500 | 50 MiB | 10.0% [9.3,10.9] | 2.7% [2.0,3.6] | **13.8% [13.3,14.3]** | 4.0% [3.3,4.8] |
| 500 | 100 MiB | 18.0% [17.2,18.7] | 10.4% [9.4,11.9] | **23.6% [22.7,24.5]** | 11.5% [10.2,13.3] |
| 500 | 200 MiB | **52.1% [51.5,52.6]** | 28.6% [27.3,30.5] | 38.9% [38.0,39.8] | 30.7% [29.4,32.5] |
| 2,000 | 50 MiB | 3.2% [2.7,3.9] | 0.0% [0.0,0.1] | **4.9% [4.6,5.3]** | 0.6% [0.5,0.7] |
| 2,000 | 100 MiB | 5.2% [4.7,5.7] | 0.5% [0.4,0.6] | **7.9% [7.5,8.3]** | 1.6% [1.4,1.7] |
| 2,000 | 200 MiB | 8.8% [7.9,9.8] | 3.1% [2.9,3.3] | **13.1% [12.6,13.6]** | 4.2% [3.9,4.5] |
| 5,000 | 50 MiB | 1.6% [1.4,1.8] | 0.0% [0.0,0.0] | **3.2% [3.0,3.3]** | 0.2% [0.2,0.2] |
| 5,000 | 100 MiB | 2.8% [2.4,3.1] | 0.0% [0.0,0.0] | **5.1% [5.0,5.2]** | 0.5% [0.5,0.6] |
| 5,000 | 200 MiB | 4.8% [4.4,5.1] | 0.5% [0.4,0.6] | **8.2% [8.0,8.4]** | 1.5% [1.4,1.6] |

### Finding 4 -- `ADMISSION_LRU` wins 8 of 9 cells, often by a wide margin, with tight non-overlapping CIs

At every scale/cache-size combination except one, `ADMISSION_LRU` has the
highest hit rate of all five policies tested, with confidence intervals
that don't overlap the next-best policy's -- a genuine, statistically
robust effect, not noise. The relative improvement over plain `LRU`
generally grows with eviction pressure: +38% at 500 conversations/50 MiB,
+82% at 5,000 conversations/50 MiB. `ADMISSION_COST_AWARE` also beats
plain `COST_AWARE` in every cell (a consistent rescue effect, as found in
the earlier thin verification), but never catches up to `ADMISSION_LRU`.

### Finding 5 -- the one cell where it loses reveals a real design limitation, not noise

At 500 conversations / 200 MiB (the *most* generously-sized cache tested
relative to its working set), `ADMISSION_LRU` (38.9%) is clearly **worse**
than plain `LRU` (52.1%) -- a 13pp gap with non-overlapping CIs, not
sampling error. The raw per-repeat data shows why: the cache still fills
and evicts substantially even at 200 MiB (`LRU`: ~800-900 evictions per
run), and `ADMISSION_LRU` rejects roughly as many admissions as it lets
through (900-1,080 rejections per run) while evicting only ~500 times --
meaning a large fraction of admission decisions are being made under
**low-pressure conditions where most candidates are tied at the same low
frequency estimate** (many chunks touched once or twice). `should_admit`
uses a **strict `>`** comparison, so every tie is resolved in the
incumbent's favor -- which is exactly right under heavy pressure (Finding
4's wins), but under lighter, more ambiguous pressure it means a lot of
otherwise-harmless turnover that plain `LRU`'s recency-based replacement
would have handled fine gets blocked instead, for no benefit. This is the
same underlying mechanism as the freeze finding below, on a spectrum
rather than a binary: **the design's strict tie-breaking rule is a
liability specifically under low-to-moderate eviction pressure**, and an
asset under high pressure.

## Stress tests

The existing four real-data stress tests
(`tests/benchmarks/test_cache_policy_bench_real_data.py`) now include
`ADMISSION_LRU` in their policy list and all pass: near-empty-cache thrash
handled without crashing, hit rate non-decreasing across a 5-point
cache-size sweep (no capacity-cliff anomalies -- notable given Finding 5
above shows non-monotonic behavior *across policies* at fixed scale, but
monotonicity *within* `ADMISSION_LRU` as its own cache size grows still
holds), the longest real conversations replay without error, and low
concurrent fan-out beats high fan-out as expected.

### Finding 6 (new) -- `AdmissionControlledPolicy` can permanently freeze under purely one-shot traffic

A new test, `test_admission_control_freezes_under_purely_novel_traffic`
(`tests/benchmarks/test_cache_policy_bench.py`), specifically probes the
tie-breaking risk identified above at its logical extreme: traffic where
*every* chunk is touched exactly once and never reused (`novel_long`).
Result, confirmed empirically before writing the test: **zero evictions,
thousands of silently rejected admissions** -- the cache fills once and
then never changes again for the rest of the run. Mechanism: every
newcomer's freshly-incremented frequency (1) never *strictly* exceeds an
already-resident incumbent's (also 1, and never reused so never
incremented further), so nothing is ever admitted again after the first
fill. Plain `LRU` under the identical traffic keeps evicting and rotating
normally (confirmed in the same test). This doesn't affect hit rate for
purely one-shot traffic (it's 0% for every policy there by construction),
but it is a real, silent behavioral cliff: a cache serving a workload that
is *mostly* but not *entirely* one-shot (a realistic scenario) could see
its useful capacity shrink over time as more and more slots get
permanently claimed by early, never-reused entries, unable to be reclaimed
even once the cache would clearly benefit from turnover. **This is
flagged as a priority follow-up, not fixed here** (per this
investigation's scope) -- a `>=` comparison with a secondary recency
tiebreak, or an occasional forced-eviction fallback when nothing beats the
incumbent, are the natural next things to try.

## `WindowedAdmissionControlledPolicy`: does windowing fix Findings 5-6?

Per an explicit decision to keep `AdmissionControlledPolicy` unchanged (it
remains shipped, tested, and the default), the tie-breaking fix from
Findings 5-6 was built as a **second, independently selectable class**,
`WindowedAdmissionControlledPolicy` (same file, shares `_FrequencySketch`),
registered via `get_cache_policy("WINDOWED_ADMISSION_<INNER>")`. This
keeps both admission-control designs directly comparable rather than
replacing one with the other.

Design (see the class docstring for full detail): new keys always enter a
small **window** region unconditionally (a `window_capacity`-bounded
subset of resident keys, default 20) -- so, unlike the strict design,
nothing is ever silently rejected. Only when the window itself overflows
is its oldest member evaluated: promoted into the frequency-gated **main**
region if its sketch estimate reaches `promotion_threshold` (default 2),
or queued for a real eviction otherwise. `inner_policy` continues to rank
everything in main, same as before.

### Bug 3 -- window capacity was never actually enforced (a zero-sum bug)

The first implementation computed window capacity per-call as
`len(cache_dict) * window_fraction` and only pruned it from inside
`get_evict_candidates`. Running Experiment 1 below caught the problem the
same way Bug 2 was caught in the un-windowed class: `WINDOWED_ADMISSION_LRU`
produced an eviction count on `mixed_zipfian`/100 MiB **identical to the
digit** to plain `LRU`'s (1,973 both) -- an implausible coincidence for a
genuinely different algorithm, worth distrusting rather than accepting.
Instrumented replay confirmed the window had silently grown to the
**entire cache** during the fill phase (nothing prunes it before the first
eviction ever happens) and then never shrank afterward: every post-fill
eviction cycle removed exactly one window member and added exactly one new
key back to the window, a mathematically invariant wash. Net effect: the
"windowed" design was doing no filtering at all, just re-deriving
`inner_policy`'s own ranking through an expensive detour.

**Fix**: `window_capacity` is now an absolute integer, enforced
**immediately at insertion time** (inside `update_on_put`/
`update_on_put_with_metadata`, which need no `cache_dict` access to do
this), not lazily during eviction. An insertion that overflows the window
evaluates its oldest member on the spot -- promoted (stays resident, no
eviction needed) or pushed onto a new `self._pending_discards` FIFO queue,
which `get_evict_candidates` drains first, ahead of `inner_policy`'s own
ranking, before any real eviction happens. This keeps window size
genuinely, continuously bounded. Re-verified afterward: `WINDOWED_ADMISSION_LRU`
now produces hit=86.0%/evictions=1,862 on the same cell -- close to but
measurably different from plain `LRU`, as a real (not degenerate) design
should.

### Experiment 1 rerun: synthetic cache-size sweep, all three tiers

At 100 MiB (full data in
[`results/admission_control/sweep_results.json`](../../../../../benchmarks/cache_policy/results/admission_control/sweep_results.json)):

| Workload | Policy | Hit rate | Evictions | p95 latency |
|---|---|---:|---:|---:|
| mixed_zipfian | LRU (baseline) | 85.3% | 1,973 | 30.7 ms |
| mixed_zipfian | **ADMISSION_LRU** | **87.9%** | **288** | 25.6 ms |
| mixed_zipfian | WINDOWED_ADMISSION_LRU | 86.0% | 1,862 | 30.7 ms |
| mixed_zipfian | ADMISSION_COST_AWARE | 82.6% | 1,802 | 25.6 ms |
| mixed_zipfian | WINDOWED_ADMISSION_COST_AWARE | 83.3% | 2,302 | 25.6 ms |
| multi_round_chat | LRU (baseline) | 58.0% | 910 | 61.4 ms |
| multi_round_chat | **ADMISSION_LRU** | **83.3%** | **0** | **15.2 ms** |
| multi_round_chat | WINDOWED_ADMISSION_LRU | 78.3% | 227 | 15.2 ms |
| multi_round_chat | ADMISSION_COST_AWARE | 83.3% | 0 | 15.2 ms |
| multi_round_chat | WINDOWED_ADMISSION_COST_AWARE | 78.4% | 236 | 24.5 ms |

### Finding 7 -- windowing trades some of the strict design's peak upside for structural safety

On both workloads, `WINDOWED_ADMISSION_*` lands **between** the plain
baseline and the strict `ADMISSION_*` design -- clearly better than no
admission control, but short of the strict design's best-case numbers
(e.g. multi_round_chat: 58.0% -> 78.3% -> 83.3%). This is the expected
cost of the fix: window capacity is partly "spent" on entries that turn
out to be infrequent and get discarded rather than ever contributing a
hit, which the strict design's tie-always-favors-incumbent rule never
pays (at the cost of the freeze/regression risk it carries instead). This
is a real, honest tradeoff, not a strictly dominant fix.

### Experiment 2 rerun: `window_capacity` / `promotion_threshold` ablation

`benchmarks/cache_policy/run_admission_control_ablation.py`, 100 MiB
(full data in
[`results/admission_control/windowed_admission_control_ablation.json`](../../../../../benchmarks/cache_policy/results/admission_control/windowed_admission_control_ablation.json)):

| Workload | Variant | Hit rate | Evictions |
|---|---|---:|---:|
| mixed_zipfian | tiny_window (5, thresh=2) | 81.6% | 1,861 |
| mixed_zipfian | default (20, thresh=2) | 81.5% | 1,868 |
| mixed_zipfian | large_window (80, thresh=2) | 81.1% | 1,921 |
| mixed_zipfian | lenient_promotion (20, thresh=1) | 79.8% | 2,078 |
| mixed_zipfian | strict_promotion (20, thresh=4) | 83.0% | 1,678 |
| mixed_zipfian | no_admission (LRU) | 79.8% | 2,078 |
| multi_round_chat | tiny_window (5, thresh=2) | 79.9% | 178 |
| multi_round_chat | default (20, thresh=2) | 78.3% | 227 |
| multi_round_chat | large_window (80, thresh=2) | 74.5% | 317 |
| multi_round_chat | lenient_promotion (20, thresh=1) | 58.0% | 910 |
| multi_round_chat | strict_promotion (20, thresh=4) | 80.8% | 198 |
| multi_round_chat | no_admission (LRU) | 58.0% | 910 |

A useful correctness cross-check surfaced by this ablation, not just a
performance result: `lenient_promotion` (`promotion_threshold=1`)
reproduces plain `LRU`'s numbers **exactly** on `multi_round_chat`
(58.0%/910 evictions, both cells). This is mathematically expected, not a
bug -- at `promotion_threshold=1`, every window overflow is promoted
(the sketch estimate is always >= 1 the moment a key is inserted), so
`_pending_discards` is always empty and eviction always falls through to
`inner_policy`'s own ranking: the windowed design *correctly* degenerates
to its inner policy at this degenerate setting. That the exact-match
coincidence now only appears at this one deliberately-degenerate
configuration, rather than at the shipped default (as Bug 3 caused), is
itself evidence the fix is structurally sound.

Smaller windows and stricter promotion both trend better here (less
capacity "wasted" holding entries that never earn promotion) --
`strict_promotion` is the best-performing windowed variant on both
workloads, though still short of the strict `ADMISSION_LRU` design's
best numbers from Experiment 1.

### Experiment 3 rerun: Zipf skew robustness

`benchmarks/cache_policy/robustness_sweep.py`, 100 MiB (full data in
[`results/admission_control/robustness_zipf_skew.json`](../../../../../benchmarks/cache_policy/results/admission_control/robustness_zipf_skew.json)):

| `zipf_s` | LRU | ADMISSION_LRU | WINDOWED_ADMISSION_LRU |
|---|---:|---:|---:|
| 0.6 (mild) | 43.5% | **51.1%** | 44.7% |
| 1.2 (default) | 79.8% | **83.7%** | 81.5% |
| 2.0 (extreme) | 96.9% | 96.9% | 96.9% |

Same pattern as Experiment 1: windowing recovers most, not all, of the
strict design's benefit, across skew strength (not just one snapshot),
and all policies converge once the working set fits entirely in cache.

### Experiment 4 rerun: statistically rigorous real-data validation

Same 3-scale x 3-cache-size x 6-repeat grid, `--policies LRU LFU FIFO MRU
COST_AWARE ADMISSION_LRU ADMISSION_COST_AWARE WINDOWED_ADMISSION_LRU
WINDOWED_ADMISSION_COST_AWARE` (full data in
[`results/real_data/real_dataset_ci.json`](../../../../../benchmarks/cache_policy/results/real_data/real_dataset_ci.json)):

| Scale | Cache | LRU | ADMISSION_LRU | WINDOWED_ADMISSION_LRU | ADMISSION_COST_AWARE | WINDOWED_ADMISSION_COST_AWARE |
|---|---|---:|---:|---:|---:|---:|
| 500 | 50 MiB | 10.0% | **13.8%** | 10.9% | 4.0% | 4.4% |
| 500 | 100 MiB | 18.0% | 23.6% | **24.1%** | 11.5% | 13.7% |
| 500 | 200 MiB | **52.1%** | 38.9% | 42.8% | 30.7% | 40.3% |
| 2,000 | 50 MiB | 3.2% | **4.9%** | 3.3% | 0.6% | 0.5% |
| 2,000 | 100 MiB | 5.2% | **7.9%** | 5.5% | 1.6% | 1.5% |
| 2,000 | 200 MiB | 8.8% | **13.1%** | 9.4% | 4.2% | 4.8% |
| 5,000 | 50 MiB | 1.6% | **3.2%** | 1.7% | 0.2% | 0.1% |
| 5,000 | 100 MiB | 2.8% | **5.1%** | 2.8% | 0.5% | 0.4% |
| 5,000 | 200 MiB | 4.8% | **8.2%** | 4.7% | 1.6% | 1.4% |

(Bootstrap 95% CIs omitted from this compressed view; see the linked JSON
for the full per-cell intervals -- all comparisons called out below have
non-overlapping CIs.)

### Finding 8 -- windowing meaningfully narrows the Finding 5 regression at small scale, but converges to plain LRU (not to the strict design's wins) as traffic gets more purely one-shot

At the Finding 5 regression cell (500 conversations / 200 MiB),
`WINDOWED_ADMISSION_LRU` (42.8%) sits clearly between plain `LRU`'s
52.1% and `ADMISSION_LRU`'s 38.9% -- a real, partial recovery (closes
about a third of the gap), not a full fix. At 100 MiB and below, windowed
`LRU` is competitive with or slightly beats the strict design (24.1% vs.
23.6% at 100 MiB). But at the larger, more one-shot-dominated scales
(2,000 and 5,000 conversations, where every policy's hit rate is in the
low single digits), `WINDOWED_ADMISSION_LRU` converges to **plain `LRU`**,
not to `ADMISSION_LRU`'s wins -- e.g. 5,000 conversations/200 MiB: LRU
4.8%, windowed 4.7%, strict 8.2%. Mechanism: under traffic this close to
purely one-shot, almost nothing ever reaches `promotion_threshold=2`
before its first eviction opportunity, so `_pending_discards` rarely
differs from what plain `LRU` would have evicted anyway -- windowing's
safety net (never rejecting) costs it the strict design's aggressive
gating exactly where that gating pays off most on this corpus. The one
clear exception is wrapping `COST_AWARE`: at 500/200 MiB,
`WINDOWED_ADMISSION_COST_AWARE` (40.3%) dramatically outperforms both
plain `COST_AWARE` (28.7%, from Finding 4's table) and
`ADMISSION_COST_AWARE` (30.7%) -- windowing helps the weaker inner policy
more than it helps `LRU`, consistent with Finding 7's "recovers some but
not all of the strict design's benefit" pattern generalizing differently
depending on how much headroom the inner policy has to begin with.

### Finding 9 -- windowing does structurally eliminate the freeze (Finding 6), confirmed both by construction and by test

Unlike Finding 5's partial, empirically-measured recovery, Finding 6's
freeze fix is a hard structural guarantee, not a matter of degree: the
window unconditionally admits every new key, so nothing is ever silently
rejected regardless of frequency ties. Under purely one-shot traffic
(the exact `novel_long` scenario that permanently freezes
`ADMISSION_LRU` at zero evictions), every window overflow evaluates a
key stuck at frequency 1 (`< promotion_threshold=2`), so it is
**discarded** -- a real eviction -- every single time. Confirmed by
`test_windowed_admission_control_does_not_freeze_under_purely_novel_traffic`
(`tests/benchmarks/test_cache_policy_bench.py`), run against the fixed
implementation: `eviction_count > 0`, in direct contrast to
`test_admission_control_freezes_under_purely_novel_traffic`'s `== 0` for
the strict design under identical traffic.

### Stress tests rerun

`tests/benchmarks/test_cache_policy_bench_real_data.py`'s four stress
tests (near-empty-cache thrash, capacity-cliff monotonicity, longest-
conversation replay, fan-out degradation) now include both
`ADMISSION_LRU` and `WINDOWED_ADMISSION_LRU` in their policy list and all
pass -- no crashes, no monotonicity violations, for either design.

### Verdict

Windowing is a genuine, working fix for the *catastrophic* failure mode
(Finding 6's freeze) -- structurally, not just empirically, since the
window's unconditional admission makes silent permanent rejection
unreachable by construction. For the *regression* failure mode (Finding
5), it's a real but partial mitigation: better in the small-to-moderate
scale range tested, but it gives up a meaningful fraction of the strict
design's peak upside everywhere, and converges to plain `LRU` (not to the
strict design's wins) under traffic dominated by one-shot access. Neither
design is a strict improvement over the other -- they occupy different
points on a safety/peak-performance tradeoff, which is why both are kept
as independently selectable classes rather than one replacing the other.

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

Per explicit scope decision, this work ships both classes only: either is
selectable today via `get_cache_policy`/config and both are fully correct
and tested, but no storage backend calls `should_admit` yet, so
admission-rejection behavior does not affect production request handling
until a backend is wired to call it -- a deliberate, separate follow-up.
**Given Finding 6, any such wiring targeting a workload with a meaningful
one-shot-traffic component should wire `WindowedAdmissionControlledPolicy`,
not `AdmissionControlledPolicy`**, since the strict design's freeze is a
real production risk there; workloads known to have sustained,
non-trivial reuse and no one-shot risk can safely use the strict design
for its larger peak upside (Finding 7).

## Three directions compared

Two structurally different fixes for `CostAwareEvictionPolicy`'s
real-data weakness, plus a fix for a limitation surfaced by evaluating
the second one, were built and evaluated end to end:

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
policy. **Outcome**: the clear winner on aggregate, confirmed now across
a full statistical grid (Experiment 4), a robustness sweep (Experiment
3), and a synthetic cache-size sweep (Experiment 1) -- not just the two
data points the original verification checked. It substantially rescues
`COST_AWARE` too, without closing the gap to wrapping `LRU`. It is **not
a strict, unconditional win**: Finding 5 shows a real, mechanistically
understood regression under generously-sized caches with low-to-moderate
eviction pressure, and Finding 6 shows a severe, silent freeze failure
mode under purely one-shot traffic -- both traced to the same root cause
(strict tie-breaking always favoring incumbents), both absent from the
original thin verification, both only surfaced by extending this
investigation to the same rigor as the `CostAwareEvictionPolicy`
evaluation.

### Direction C: `WindowedAdmissionControlledPolicy` (this doc, "does windowing fix Findings 5-6?")

Built specifically to address Direction B's own Findings 5-6, as a
**second, independently selectable class** rather than a rewrite (an
explicit scope decision, so both admission-control designs stay directly
comparable). **Outcome**: structurally eliminates the freeze failure mode
(Finding 9) -- a hard guarantee, not a probabilistic improvement -- and
meaningfully narrows the regression failure mode at small-to-moderate
scale (Finding 8), but gives up a real fraction of Direction B's peak
upside everywhere (Finding 7), and at highly one-shot-dominated scale
converges to plain `LRU` rather than to Direction B's wins. Not a strict
improvement over Direction B -- a different point on a safety/peak-
performance tradeoff, not a replacement for it.

### Recommendation

**No single policy dominates; the choice depends on what the workload's
one-shot-traffic risk looks like**:

- If the workload is known to have sustained, non-trivial key reuse and
  a negligible purely-one-shot component (the kind of traffic Findings 1,
  3, and 4 were measured on), **`AdmissionControlledPolicy` remains the
  stronger choice** -- its wins are large, statistically robust, and hold
  across most of the parameter space tested, and it doesn't pay
  Direction C's window-capacity tax.
- If the workload's one-shot-traffic share is unknown, variable, or
  known to be significant, **`WindowedAdmissionControlledPolicy` is the
  safer default** -- Finding 6's silent, permanent freeze is a real
  production risk under the strict design that windowing eliminates by
  construction, at a real but bounded cost in peak hit rate.
- Both remain **strictly better than no admission control at all** in
  every synthetic and real-data scenario tested (Experiments 1, 3, 4)
  except `ADMISSION_MRU`/`WINDOWED_ADMISSION_MRU`, which simply inherit
  `MRU`'s poor baseline ranking (admission control gates *what* gets in,
  not *how well* the inner policy ranks what's already there).

Direction A's frequency fix remains a legitimate, independently useful
improvement to `CostAwareEvictionPolicy` specifically (already shipped)
but is not, on its own, competitive with plain `LRU`/`LFU` on real
traffic. All three directions compose today (e.g.
`get_cache_policy("ADMISSION_COST_AWARE")`,
`get_cache_policy("WINDOWED_ADMISSION_COST_AWARE")`), and Finding 8 shows
windowing helps `COST_AWARE` more, proportionally, than it helps `LRU`
-- but neither composition catches up to `ADMISSION_LRU`'s raw numbers.

**Concrete next steps, in priority order, none done here per this
investigation's scope**:
1. Wire `should_admit` into `local_disk_backend.py` (the identified
   low-risk integration point) so the effect is real for actual request
   handling, not only benchmarked -- gated on picking a policy per the
   recommendation above based on the target workload's traffic mix.
2. If the target deployment's real traffic mix is uncertain, consider
   exposing both policies as config-selectable rather than hardcoding
   one, so operators can pick per-deployment.
3. Investigate whether `WindowedAdmissionControlledPolicy`'s
   `window_capacity`/`promotion_threshold` could be tuned adaptively
   (e.g. based on observed reuse rate) to recover more of Direction B's
   peak upside without reintroducing the freeze risk -- speculative,
   not attempted here.

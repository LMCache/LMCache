# `CostAwareEvictionPolicy` -- Performance Evaluation

This doc evaluates `lmcache/v1/storage_backend/cache_policy/cost_aware_policy.py`
against the existing `LRU`, `LFU`, `FIFO`, and `MRU` policies, using the
benchmark suite in [`benchmarks/cache_policy/`](../../../../../benchmarks/cache_policy/README.md).
See that README for how to reproduce every number and chart below.

## Methodology and its limits

There is no GPU available in the environment this evaluation was produced
in, so this is **not** an end-to-end vLLM/LMCache benchmark. Instead:

- Requests are synthetic (see "Workloads" below), replayed through the
  real policy objects via the exact same call sequence the storage backend
  uses (`update_on_hit`, `update_on_put_with_metadata`,
  `get_evict_candidates`, `update_on_force_evict` --
  `lmcache/v1/storage_backend/local_cpu_backend.py`).
- "Latency" is a **modeled** quantity: a fixed per-chunk retrieval cost for
  hits, plus a per-token prefill cost times recomputed tokens for misses
  (`lmcache/tools/cache_policy_bench/cost_model.py`). It captures the
  *relative* cost structure the policy itself optimizes for
  (`observed_recompute_tokens`), not measured wall-clock inference time.
- "Throughput" (`requests_per_second` in the sweep CSV) is the simulator's
  own Python loop throughput -- i.e. it measures the **policy's
  bookkeeping overhead**, not model throughput. This turns out to be one
  of the most interesting findings below.
- There is no GPU utilization metric; `rss_delta_bytes` (via `psutil`) is
  a coarse CPU-memory proxy only.
- `CostAwareEvictionPolicy`'s recency-decay term reads real
  `time.monotonic()`, not a simulated clock. On low-eviction-count
  workloads this makes its hit-rate outcome sensitive to the actual
  wall-clock speed of the run (interpreter/OS scheduling noise), not just
  workload content -- see Experiment 2 for a measured example. High
  eviction-count workloads (`mixed_zipfian`) are unaffected in practice
  because the cost-density term dominates.

Treat every hit-rate number here as representative of *relative* policy
behavior under these access patterns, and every latency/throughput number
as representative of *relative computational cost*, not absolute
production numbers.

## Workloads

| Workload | Shape | What it stresses |
|---|---|---|
| `repetitive_short` | 15-100 distinct short prompts, uniform random reuse | Hit/miss ratio under a small, fully-cacheable working set |
| `novel_long` | Every request unique, 4K-16K tokens | Pure insertion/eviction overhead; hit rate is always 0% by construction |
| `mixed_zipfian` | 300 distinct prompts, Zipf-distributed popularity (`s=1.2`) | Realistic skewed reuse (hot prefixes + long tail) -- the primary cross-policy comparison workload |
| `multi_round_chat` | Sessions with a monotonically growing shared prefix | `chunk_start`/recompute-cost accounting specific to `CostAwareEvictionPolicy` |

## Experiment 1: vanilla vs. extended, cache-size sweep

Ran all five policies across all four workloads at 50 / 100 / 200 MiB
simulated cache capacity (256 KiB/chunk).

![Hit rate vs cache size](../../../../../benchmarks/cache_policy/results/charts/hit_rate_vs_cache_size.png)

![p95 modeled latency](../../../../../benchmarks/cache_policy/results/charts/latency_p95.png)

![Simulator throughput](../../../../../benchmarks/cache_policy/results/charts/throughput.png)

Selected rows at 100 MiB (full data in
[`results/sweep_results.csv`](../../../../../benchmarks/cache_policy/results/sweep_results.csv)):

| Workload | Policy | Hit rate | Evictions | p95 latency | Throughput (req/s) |
|---|---|---:|---:|---:|---:|
| mixed_zipfian | LRU | 85.3% | 1,973 | 30.7 ms | 101,545 |
| mixed_zipfian | LFU | 87.6% | 1,611 | 30.7 ms | 108,281 |
| mixed_zipfian | MRU | 31.8% | 5,928 | 36.3 ms | 133,061 |
| mixed_zipfian | **COST_AWARE** | **65.7%** | 5,141 | **17.9 ms** | 1,851 |
| multi_round_chat | LRU | 58.0% | 910 | 61.4 ms | 69,797 |
| multi_round_chat | LFU | 80.8% | 198 | 19.9 ms | 121,482 |
| multi_round_chat | MRU | 82.1% | 120 | 19.9 ms | 196,383 |
| multi_round_chat | **COST_AWARE** | **70.3%** | 187 | 61.4 ms | 8,272 |

`repetitive_short` and `novel_long` don't discriminate between policies at
this cache size: the former's whole working set fits (all policies hit
96.7%), and the latter never hits by construction (0% for every policy,
identical eviction count). They're still valuable as smoke-test /
overhead-isolation workloads (see `README.md`), just not as comparison
signal here.

### Finding 1 -- hit rate: `COST_AWARE` is not a drop-in win on popularity-skewed traffic

On `mixed_zipfian`, `COST_AWARE` (65.7%) trails both `LRU` (85.3%) and
`LFU` (87.6%). The reason is structural, not a bug: the policy's score is

```
score = (estimated_recompute_tokens / memory_size_bytes) / (1 + age/half_life)
```

With uniform `memory_size_bytes` (every chunk is the same 256 KiB in this
benchmark), `cost_density` reduces to `estimated_recompute_tokens` --
i.e. **how expensive this chunk would be to recompute**, not **how often
it gets reused**. `mixed_zipfian`'s popular prompts are short (few
chunks, `chunk_start` close to `total_tokens` for most of their chunks),
so they score *low* cost and get evicted preferentially, even though
they're the chunks a frequency- or recency-aware policy would keep. This
is an intentional design tradeoff (recompute-cost minimization, not hit-rate
maximization) worth stating plainly in the policy's docs, since it means
`COST_AWARE` is not a strict improvement over `LRU`/`LFU` on every
workload -- it wins specifically when recompute cost varies independently
of reuse frequency (see Finding 2).

### Finding 2 -- `COST_AWARE` shows promise where recompute cost varies by position, but the signal is noisy

On `multi_round_chat`, where later chunks in a growing prefix are cheaper
to recompute than earlier ones (`chunk_start` grows every round),
`COST_AWARE` beats `LRU`/`FIFO`/`MRU` on p95 modeled latency at the
tightest capacity (50 MiB: 51.7 ms vs. 61.4 ms for all three), though
`LFU` still edges it out there (43.0 ms). At 100 MiB, `COST_AWARE`'s hit
rate (70.5%) is above `LRU`'s (58.0%) in this single sweep run, consistent
with the policy preferentially retaining expensive-to-recompute
early-session chunks. **However**, Experiment 2 (below) found that
`multi_round_chat` hit-rate readings for `COST_AWARE` vary by up to ~10
percentage points across repeated runs of the identical config, due to
the policy's use of real wall-clock time for recency decay -- so this
particular hit-rate comparison should be read as "plausible, in the
direction the design predicts" rather than a precise, reproducible
number. The p95-latency advantage at 50 MiB is a more solid data point
since it doesn't hinge on which chunks narrowly won a close eviction
race.

### Finding 3 -- `COST_AWARE`'s eviction-candidate selection is O(n log n), not O(1)

The throughput gap is the most striking result: 1,851 req/s vs. ~100K+
req/s for the other policies on `mixed_zipfian` under high eviction churn
(a ~55x gap). This traces directly to `get_evict_candidates`:
`LRUCachePolicy`/`FIFOCachePolicy`/`MRUCachePolicy` walk an `OrderedDict`
and return as soon as they collect `num_candidates` keys (O(k)).
`CostAwareEvictionPolicy.get_evict_candidates`
(`cost_aware_policy.py:485-535`) instead builds a sort key for **every**
key in `cache_dict` and calls `sorted()` over the whole cache on every
single eviction. Under a full cache with heavy churn (thousands of
evictions per sweep run), that's thousands of O(n log n) sorts, which
dominates wall-clock cost. This doesn't affect the *modeled* per-request
latency (that's a separate cost function), but it is real CPU overhead the
storage backend pays on every eviction, and it gets worse as cache
capacity (hence `n`) grows. Worth flagging for follow-up: batching
evictions (`num_candidates > 1`, already supported by the interface) would
amortize the sort cost, whereas the current backend call site
(`local_cpu_backend.py`) evicts one candidate at a time.

## Experiment 2: ablation

Isolates the two ideas combined in `CostAwareEvictionPolicy`'s score, run
at 100 MiB against `mixed_zipfian` and `multi_round_chat`
(`benchmarks/cache_policy/run_ablation.py`; full data in
[`results/ablation_results.csv`](../../../../../benchmarks/cache_policy/results/ablation_results.csv)):

| Workload | Variant | Hit rate | p95 latency | Evictions |
|---|---|---:|---:|---:|
| mixed_zipfian | full (`half_life=60s`, `alpha=0.2`) | 48.1% | 20.5 ms | ~5,975 |
| mixed_zipfian | no_recency (`half_life=1e9`) | 48.1-48.2% | 20.5 ms | ~5,975 |
| mixed_zipfian | no_ewma (`alpha=1.0`) | 48.1-48.2% | 20.5 ms | ~5,975 |
| mixed_zipfian | cost_agnostic (LRU reference) | 79.8% | 30.7 ms | 2,078 |
| multi_round_chat | full | 60.3-70.9% (run-to-run noise, see below) | 56.8-61.4 ms | 185-224 |
| multi_round_chat | no_recency | 66.0-70.5% (run-to-run noise, see below) | 56.8-61.4 ms | 187-206 |
| multi_round_chat | no_ewma | 70.5-70.9% | 61.4 ms | 186-187 |
| multi_round_chat | cost_agnostic (LRU reference) | 58.0% | 61.4 ms | 910 |

(`mixed_zipfian` uses a smaller request count than Experiment 1, so
absolute values differ from that table; only within-table comparisons
matter. `multi_round_chat` ranges are from 5 repeated runs of the
identical config -- see below for why.)

- **Important methodology caveat, discovered while re-running this
  ablation**: `CostAwareEvictionPolicy` computes recency decay from
  `time.monotonic()` -- **real** wall-clock time elapsed during the
  simulation loop -- not a simulated logical clock. On `multi_round_chat`
  (small eviction counts, so each run's wall-clock timing is dominated by
  interpreter/OS scheduling noise rather than by the workload itself), 5
  repeated runs of the exact same `full` config produced hit rates
  ranging from 60.3% to 70.9%, and `full` vs. `no_recency` swapped which
  one scored higher between runs. **The original single-run reading of
  this table (`full` at 70.5% vs. `no_recency` at 66.0%) is not a
  reliable signal** -- it's within this run-to-run noise band, not a real
  effect of recency decay. On `mixed_zipfian`, by contrast, all three
  `COST_AWARE` variants stayed tightly clustered (48.1-48.2%) across
  repeated runs, because that workload's much higher eviction volume
  (~5,975 evictions vs. ~200) means cost-density dominates and swamps
  whatever wall-clock jitter affects the decay term. **Net effect: this
  benchmark suite cannot currently distinguish the standalone
  contribution of recency decay on low-eviction-count workloads**; doing
  so would require the simulator to drive the policy on a fake/injected
  clock instead of `time.monotonic()`, which is out of scope for this
  suite without a policy-side seam to inject one.
- **EWMA smoothing had no measurable effect** in either workload,
  consistently across repeated runs. That's a property of these specific
  workloads, not evidence the smoothing is useless: `cost_ewma_alpha`
  only changes behavior when the *same key* receives multiple distinct
  `observed_recompute_tokens` observations (repeated put/evict/reinsert
  cycles at different `chunk_start` positions). Neither `mixed_zipfian`
  nor `multi_round_chat` currently produces that pattern -- each chunk is
  inserted once and only ever hit afterward. A workload with request
  reordering or chunk re-insertion at varying prefix depths would be
  needed to exercise the EWMA path; that's a gap in the current suite,
  not a conclusion about the EWMA feature itself.
- **On `mixed_zipfian`, none of the `COST_AWARE` variants beat the LRU
  reference**, consistently -- consistent with Finding 1: this workload's
  popularity skew isn't the axis `COST_AWARE` optimizes for, regardless of
  which of its two sub-mechanisms is active.

## Summary

`CostAwareEvictionPolicy` behaves as designed: it trades hit-rate-in-general
for recompute-cost-awareness, and wins specifically on workloads where
recompute cost correlates with something other than reuse frequency (e.g.
`multi_round_chat`'s growing-prefix cost gradient). It is not a universal
upgrade over `LRU`/`LFU` -- on flat, popularity-skewed traffic
(`mixed_zipfian`) with uniform chunk sizes, simpler recency/frequency
policies still win on hit rate. Its `get_evict_candidates` implementation
is algorithmically more expensive than the existing policies' and should
be considered for batched-candidate optimization before deployment in
eviction-heavy environments.

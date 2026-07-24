# Cache-policy performance benchmarks

Performance test suite for `lmcache/v1/storage_backend/cache_policy/` --
compares `CostAwareEvictionPolicy` against the existing `LRU`, `LFU`,
`FIFO`, and `MRU` policies under synthetic prefix-cache workloads.

This suite is **CPU-only and requires no GPU or running model**. It does
not measure real inference latency; it replays synthetic request traces
through the real policy objects (the same `update_on_hit` /
`update_on_put_with_metadata` / `get_evict_candidates` /
`update_on_force_evict` calls the storage backend makes -- see
`lmcache/v1/storage_backend/local_cpu_backend.py`) and scores each request
with a modeled latency cost function. See
[`docs/design/v1/storage_backend/cache_policy/cost-aware-policy-eval.md`](../../docs/design/v1/storage_backend/cache_policy/cost-aware-policy-eval.md)
for the full write-up, methodology limitations, and results.

## Layout

- `lmcache/tools/cache_policy_bench/workloads.py` -- synthetic request
  generators (`repetitive_short`, `novel_long`, `mixed_zipfian`,
  `multi_round_chat`).
- `lmcache/tools/cache_policy_bench/cost_model.py` -- the modeled
  hit/miss latency function.
- `lmcache/tools/cache_policy_bench/runner.py` -- the simulation loop,
  sweep driver, and CSV/JSON writers.
- `tests/benchmarks/test_cache_policy_bench.py` -- pytest-benchmark tests.
- `benchmarks/cache_policy/plot_results.py` -- renders charts from a
  sweep CSV.
- `benchmarks/cache_policy/run_ablation.py` -- isolates the two ideas in
  `CostAwareEvictionPolicy`'s score (EWMA cost smoothing vs. recency
  decay).
- `benchmarks/cache_policy/results/` -- checked-in sample CSV/JSON plus
  the charts referenced by the evaluation doc. Nightly CI runs write
  fresh output under `results/nightly/` (uploaded as a workflow
  artifact, not committed).

## Running it

### Fast smoke benchmarks (what CI runs on every PR)

```bash
pytest tests/benchmarks/test_cache_policy_bench.py -m "not slow" --benchmark-only
```

Each case is one (policy, workload) pair at a small fixed cache size and a
small request count -- fast enough to run on every PR
(`.github/workflows/test.yml`), and asserts the run doesn't crash and
produces sane metrics (hit rate in `[0,1]`, etc). It is a regression guard,
not a correctness check -- correctness lives in `tests/v1/test_cache_policy.py`.

### Full parameter sweep (nightly)

```bash
pytest tests/benchmarks/test_cache_policy_bench.py -m slow -v
```

or directly via the CLI, which also writes CSV/JSON:

```bash
python -m lmcache.tools.cache_policy_bench.runner --sweep \
    -o benchmarks/cache_policy/results/local
```

This sweeps all five policies across all four workloads and three cache
sizes (50 / 100 / 200 MiB by default -- pass `--cache-sizes-mib` to
override). It runs on `.github/workflows/cache_policy_benchmark_nightly.yml`
(schedule + `workflow_dispatch`) and uploads results as a workflow artifact.

### Charts

```bash
python benchmarks/cache_policy/plot_results.py \
    -i benchmarks/cache_policy/results/sweep_results.csv \
    -o benchmarks/cache_policy/results/charts
```

### Ablation study

```bash
python benchmarks/cache_policy/run_ablation.py \
    -o benchmarks/cache_policy/results
```

Isolates `CostAwareEvictionPolicy`'s two combined ideas: pure cost-density
ranking with recency decay disabled (`no_recency`), unsmoothed cost
observations (`no_ewma`), and the full policy, against an `LRU` reference
(`cost_agnostic`).

## Metrics collected

Each `BenchResult` row (see `runner.py`) reports, per (policy, workload,
cache-size) combination:

- `token_hit_rate` -- fraction of tokens served from the prefix cache.
- `latency_mean_seconds` / `latency_p50_seconds` / `latency_p95_seconds` /
  `latency_p99_seconds` -- **modeled**, not measured, request latency (see
  the evaluation doc for why and how).
- `requests_per_second` / `tokens_per_second` -- simulator throughput
  (wall-clock time of the Python simulation loop itself, a CPU-cost proxy
  for the policy's own bookkeeping overhead -- not model throughput).
- `eviction_count` -- number of chunks evicted during the run.
- `rss_delta_bytes` -- process RSS delta during the run (via `psutil`), a
  coarse CPU-memory proxy. There is no GPU utilization metric: this suite
  never touches a GPU.

## Sample output

`results/sample/` contains a small pre-generated `sweep_results.csv` and
`sweep_results.json` (from a `--quick` run) so you can see the schema
without running anything.

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
- `benchmarks/cache_policy/run_ablation.py` -- isolates the ideas in
  `CostAwareEvictionPolicy`'s score (EWMA cost smoothing, recency decay,
  frequency weighting).
- `benchmarks/cache_policy/robustness_sweep.py` -- checks that a policy
  change generalizes rather than just fixing one benchmark reading: a
  direct cost-density sanity check plus a Zipf-skew-strength sweep. See
  the "robustness sweep" section of the evaluation doc for why this
  exists.
- `lmcache/tools/cache_policy_bench/sharegpt_workload.py` -- adapts the
  real ShareGPT conversation corpus (via the existing
  `benchmarks/multi_round_qa/` download/preprocess pipeline) into the same
  `Request` shape the synthetic generators produce -- a real, not
  synthetic, data source for the same simulator.
- `benchmarks/cache_policy/stats.py` -- dependency-free percentile-bootstrap
  confidence interval helper.
- `benchmarks/cache_policy/real_dataset_eval.py` -- statistically robust
  real-data evaluation: repeated bootstrap-resampled runs with confidence
  intervals, swept across corpus scale and cache size.
- `tests/benchmarks/test_cache_policy_bench_real_data.py` -- edge-case /
  adversarial stress tests on real data (near-empty cache, capacity-cliff
  monotonicity, pathologically long conversations, high concurrent
  fan-out). Opt-in only -- see "Real-data (ShareGPT) testing" below.
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

Isolates `CostAwareEvictionPolicy`'s combined ideas: pure cost-density
ranking with recency decay disabled (`no_recency`), unsmoothed cost
observations (`no_ewma`), and the full policy, against an `LRU` reference
(`cost_agnostic`).

### Robustness sweep

```bash
python benchmarks/cache_policy/robustness_sweep.py \
    -o benchmarks/cache_policy/results
```

Verifies a policy-scoring change generalizes: a direct two-chunk check
that the cost-density term still discriminates by cost once other terms
(frequency, recency) are held constant, plus a Zipf-skew-strength sweep
(`zipf_s` from mild to extreme popularity concentration) so a hit-rate
improvement isn't just an artifact of the one skew value the standard
sweep happens to use.

## Real-data (ShareGPT) testing

Everything above is synthetic. This tier replays the same simulator
against a real corpus of ~35K real multi-turn ShareGPT conversations
(human/GPT turns with real token-length distributions), reusing the
existing download/preprocess pipeline in `benchmarks/multi_round_qa/`
rather than reimplementing dataset fetching.

### 1. Prepare the corpus (one-time, ~650 MB download)

```bash
curl -L -o benchmarks/multi_round_qa/ShareGPT_V3_unfiltered_cleaned_split.json \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
cd benchmarks/multi_round_qa
python data_preprocessing.py --parse 1.0 --trace ShareGPT_V3_unfiltered_cleaned_split.json
cd ../..
```

(`prepare_sharegpt_data.sh` does the same thing via `wget`, if available on
your system.) This produces `benchmarks/multi_round_qa/ShareGPT.json`
(~230 MB at `--parse 1.0`, ~35K valid conversations after the script's own
validity filtering). Both files are large and git-ignored -- see
`.gitignore`. The tokenizer download requires network access to
HuggingFace; unauthenticated requests work but are rate-limited (set
`HF_TOKEN` for faster/higher-limit downloads).

### 2. Statistically robust evaluation (bootstrap CI + scale sweep)

```bash
python benchmarks/cache_policy/real_dataset_eval.py \
    --sharegpt-path benchmarks/multi_round_qa/ShareGPT.json \
    -o benchmarks/cache_policy/results/real_data
```

Runs every (policy, corpus-scale, cache-size) cell `--repeats` times (default
6), each with a fresh bootstrap resample of the conversation corpus
(`--scales`, default `500 2000 5000` conversations), and reports mean +
95% CI per cell via `benchmarks/cache_policy/stats.py::bootstrap_ci` --
not single-run point estimates. This directly addresses the wall-clock-
jitter problem noted in the evaluation doc's methodology section
(`CostAwareEvictionPolicy` uses real `time.monotonic()` for recency decay).
Writes both the raw per-repeat rows and the aggregated-with-CI table as
JSON (and CSV, git-ignored). `COST_AWARE` is significantly slower per run
than the other policies (see Finding 3 in the evaluation doc) --
budget more time as `--scales`/`--repeats` grow.

### 3. Edge-case / stress tests

```bash
LMCACHE_SHAREGPT_PATH=benchmarks/multi_round_qa/ShareGPT.json \
    pytest tests/benchmarks/test_cache_policy_bench_real_data.py -v
```

Without `LMCACHE_SHAREGPT_PATH` set, every test in that file is skipped --
**this tier is not wired into any CI workflow** (large download + tokenizer
fetch is not something to run on every PR or every nightly build). It is
local/manual-reproduction only. Covers: a far-too-small cache (thrash,
no crash), hit-rate monotonicity across a cache-size "capacity cliff",
replaying only the longest real conversations, and a direct comparison of
low vs. high concurrent conversation fan-out at a fixed cache size.

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

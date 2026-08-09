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
with a modeled latency cost function.

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
- `benchmarks/cache_policy/run_admission_control_ablation.py` -- isolates
  `AdmissionControlledPolicy`'s one tunable parameter, `halve_every` (the
  frequency sketch's decay window).
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
  real-data evaluation: repeated subsample-without-replacement runs
  (paired across policies by seed) with confidence intervals, swept
  across corpus scale and cache size -- see "Real-data (ShareGPT)
  testing" below for why this is not a classical bootstrap.
- `tests/benchmarks/test_cache_policy_bench_real_data.py` -- edge-case /
  adversarial stress tests on real data (near-empty cache, capacity-cliff
  monotonicity, pathologically long conversations, high concurrent
  fan-out). Opt-in only -- see "Real-data (ShareGPT) testing" below.
- `benchmarks/cache_policy/experiments/` -- non-production candidate
  improvements (score rebalancing, TinyLFU-style admission control,
  two-tier hierarchical caching) and `compare_directions.py`, the harness
  that ran all of them against both the synthetic suite and real data to
  find which is worth building for real. See "Direction-finding
  experiment" in the evaluation doc for the result (admission control
  won, clearly).
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
`HF_TOKEN` for faster/higher-limit downloads). `data_preprocessing.py`
tokenizes "gpt" turns with `--model` (default
`mistralai/Mistral-7B-Instruct-v0.2`).

**Corpus provenance for the committed `results/real_data/` and
`results/admission_control/` artifacts** -- the exact `ShareGPT.json`
used to produce them:

- Downloaded from the URL in the `curl` command above.
- Processed with `python data_preprocessing.py --parse 1.0 --trace
  ShareGPT_V3_unfiltered_cleaned_split.json` (tokenizer:
  `mistralai/Mistral-7B-Instruct-v0.2`, the script's default).
- 35,399 conversations after filtering.
- Expected SHA-256:
  `ba3f477c3ee8f2d5ce7411c4bdbf869bbd895c869a3fdad574d55165e7f546d5`
  (verify with `sha256sum benchmarks/multi_round_qa/ShareGPT.json`). A
  different corpus snapshot will not reproduce the committed numbers
  exactly, since ShareGPT is not versioned upstream.

### 2. Statistically robust evaluation (repeated subsampling + scale sweep)

```bash
python benchmarks/cache_policy/real_dataset_eval.py \
    --sharegpt-path benchmarks/multi_round_qa/ShareGPT.json \
    -o benchmarks/cache_policy/results/real_data
```

Runs every (policy, corpus-scale, cache-size) cell `--repeats` times (default
6). Each repeat draws a conversation subsample without replacement (see
`real_dataset_eval.py`'s module docstring); policies are paired because
every policy receives the same subsample seed at a given repeat. The
resulting intervals (via
`benchmarks/cache_policy/stats.py::bootstrap_ci`/`paired_bootstrap_ci_diff`)
describe sensitivity to the selected corpus subset -- they are not a
classical full-corpus bootstrap, and should not be described as one. 
Benchmark runs use a deterministic logical clock for reproducible recency tracking. 
Writes both the raw per-repeat rows
and the aggregated-with-CI table as JSON (and CSV, git-ignored).
`COST_AWARE` is significantly slower per run than the other policies (see
Finding 3 in the evaluation doc) -- budget more time as
`--scales`/`--repeats` grow.

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

## Direction-finding experiment

`experiments/` holds four non-production candidate improvements (score
rebalancing, TinyLFU-style admission control, two-tier hierarchical
caching, plus the existing baselines) and a harness that ran all of them
through both the synthetic suite and real data to find which one is
actually worth building. See "Direction-finding experiment" in the
evaluation doc for the result: TinyLFU-style admission control won
clearly and now ships as a real class,
`lmcache/v1/storage_backend/cache_policy/admission_control.py`
(`AdmissionControlledPolicy`, selectable via
`get_cache_policy("ADMISSION_<INNER>")`, e.g. `"ADMISSION_LRU"`)
for its design and a report comparing both directions explored. The
`experiments/` contains the scripts used to compare the candidate policy directions.
To reproduce the comparison:

```bash
# Synthetic leg (fast, no corpus needed)
python benchmarks/cache_policy/experiments/compare_directions.py \
    --synthetic -o benchmarks/cache_policy/results/experiments

# Real-data leg (needs the ShareGPT corpus prepared above; time-boxed --
# COST_AWARE-derived directions are slow, see Finding 3)
python benchmarks/cache_policy/experiments/compare_directions.py \
    --real-data --sharegpt-path benchmarks/multi_round_qa/ShareGPT.json \
    --scales 500 2000 --cache-sizes-mib 100 --repeats 4 \
    -o benchmarks/cache_policy/results/experiments

# Chart
python benchmarks/cache_policy/experiments/plot_comparison.py \
    --synthetic benchmarks/cache_policy/results/experiments/synthetic_comparison.json \
    --real-data benchmarks/cache_policy/results/experiments/real_data_comparison.json \
    -o benchmarks/cache_policy/results/charts/direction_comparison.png
```

## Full `AdmissionControlledPolicy` evaluation

Since `AdmissionControlledPolicy` is a real, registered policy
(`get_cache_policy("ADMISSION_<INNER>")`), the *existing* CLI-driven
scripts above evaluate it directly -- just pass `ADMISSION_*` names, no
new tooling needed for the synthetic sweep, robustness sweep, or
real-data validation. Only the `halve_every` ablation needed a new small
script, `run_admission_control_ablation.py`. See "Experiments 1-4" in
[`admission-control-policy.md`](../../docs/design/v1/storage_backend/cache_policy/admission-control-policy.md)
See the final report for the complete analysis and discussion of limitations.

```bash
# Synthetic sweep (reuses runner.py --sweep with an expanded --policies list)
python -m lmcache.tools.cache_policy_bench.runner --sweep \
    --policies LRU LFU FIFO MRU COST_AWARE ADMISSION_LRU ADMISSION_LFU ADMISSION_FIFO ADMISSION_MRU ADMISSION_COST_AWARE \
    -o benchmarks/cache_policy/results/admission_control
python benchmarks/cache_policy/plot_results.py \
    -i benchmarks/cache_policy/results/admission_control/sweep_results.csv \
    -o benchmarks/cache_policy/results/charts/admission_control

# halve_every ablation (new script)
python benchmarks/cache_policy/run_admission_control_ablation.py \
    -o benchmarks/cache_policy/results/admission_control

# Zipf-skew robustness (reuses robustness_sweep.py; ADMISSION_LRU already in its POLICIES list)
python benchmarks/cache_policy/robustness_sweep.py \
    -o benchmarks/cache_policy/results/admission_control

# Full real-data statistical validation (reuses real_dataset_eval.py --policies)
# -- writes to results/real_data/, the single canonical ShareGPT results
# directory; do not write a second copy under results/admission_control/.
python benchmarks/cache_policy/real_dataset_eval.py \
    --sharegpt-path benchmarks/multi_round_qa/ShareGPT.json \
    --policies LRU LFU COST_AWARE ADMISSION_LRU ADMISSION_COST_AWARE \
    --scales 500 2000 5000 --cache-sizes-mib 50 100 200 --repeats 6 \
    -o benchmarks/cache_policy/results/real_data

# Stress tests (ADMISSION_LRU already in POLICIES lists in both files)
LMCACHE_SHAREGPT_PATH=benchmarks/multi_round_qa/ShareGPT.json \
    pytest tests/benchmarks/test_cache_policy_bench_real_data.py -v
pytest tests/benchmarks/test_cache_policy_bench.py -k freeze -v
```

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

## Report and reproducibility

`report/cache_policy_evaluation_report.pdf` is the final submitted report.
**It is not generated by this repository** -- there is deliberately no
script or documented command here that (re)builds the PDF, so there is
never any ambiguity about which document is authoritative. What *is*
reproducible from the repository, exactly, is every quantitative figure
and table used in that report: each one is rendered by
[`report/generate_figures.py`](report/generate_figures.py) from a
checked-in JSON result file, mapped one-to-one in
[`report/FIGURE_SOURCES.md`](report/FIGURE_SOURCES.md).

### Fast path: rebuild the figures from committed results

```bash
./benchmarks/cache_policy/reproduce_figures.sh
```

This regenerates
[`results/admission_control/admission_vs_lfu_paired.json`](results/admission_control/admission_vs_lfu_paired.json)
(the committed ADMISSION_LRU-vs-LFU statistical comparison, computed by
[`compare_admission_vs_lfu.py`](compare_admission_vs_lfu.py) from
`multiseed_sweep_raw.json`) and then every PNG under `report/figures/`,
using only already-committed JSON -- no sweeps are re-run, so it's fast
and deterministic. It needs `lmcache` importable (`PYTHONPATH=.` against
a repo checkout under the conda env from `environment.yml`, or the
`Dockerfile` image); `requirements-figures.txt` alone is not sufficient,
since the freeze-illustration figure does a small one-off replay through
the real simulator.

**Using Docker instead of conda?** See
[`DOCKER.md`](DOCKER.md) for the Docker-only install + reproduction
walkthrough (build the image, run the deterministic check, run the
tests, rebuild the figures) -- no local Python setup required.

### Full path: rerun the experiments from scratch

Only needed if you want the committed JSON itself to reflect a fresh
benchmark run rather than trusting what's checked in. These can take a
while and are not bundled into one script:

```bash
PYTHONPATH=. python benchmarks/cache_policy/main_sweep_multiseed.py \
    -o benchmarks/cache_policy/results/admission_control

PYTHONPATH=. python benchmarks/cache_policy/run_admission_control_ablation.py \
    -o benchmarks/cache_policy/results/admission_control

PYTHONPATH=. python benchmarks/cache_policy/real_dataset_eval.py \
    --sharegpt-path benchmarks/multi_round_qa/ShareGPT.json \
    -o benchmarks/cache_policy/results/real_data
```

(See "Full `AdmissionControlledPolicy` evaluation" above for the
complete set of sweep/ablation/robustness/real-data commands, including
policy lists.) Re-run `./reproduce_figures.sh` afterward to render the
figures from the newly-written JSON.

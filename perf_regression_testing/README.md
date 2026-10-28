# Performance regression test (PRT)

PRT runs LMCache end to end every night on a real serving setup and fails
when a metric has shifted to a worse level than its own history. A weekly
suite runs vLLM's two offloading connectors and Mooncake on the same setup
as a reference; those runs never fail the build.

```
suite.py --suite nightly|weekly            (one suite at a time: a lock file)
  -> LMCache: this checkout (Buildkite builds dev)
  -> images: vLLM nightly, pinned for a week; LMCache built FROM it
             (a failed build ends the night, exit 2)
  -> for each (workload, deployment) pair of the suite:
       4 idle GPUs -> cache-layer services -> vLLM -> AIPerf (960 s window)
       -> score, check validity
       -> test each metric against this pair's own history
  -> report.json + report.md
exit 0 pass | 1 regressed (nightly only) | 2 invalid run / no images / host setup | 75 lock wait timed out
```

## What a run measures

Workload `agentic`: a replay of 393 Claude Code sessions
(`semianalysisai/cc-traces-weka-062126-256k`, pinned revision), closed loop.
A model with a 1M context replays the same sessions uncapped
(`semianalysisai/cc-traces-weka-062126`, requests up to 996,579 tokens).
Each model is a frontier open-weight release on H200, served with its vLLM
recipe's flags and replaced when a newer one appears. DeepSeek-V4.1-Flash-1017
(MLA with compressed and sliding-window KV) runs the recipe's single-node TEP
layout (TP8 + EP on 8 x H200) with DSpark speculative decoding, on the uncapped
trace at 393 concurrent sessions (all of them) and a 1500 GiB DRAM budget.
Every attention cache is whole on each TP rank, so the GPU KV is one rank's
pool, below the sessions' context. Two weekly deployments keep other models and
the 256k trace, and add disk:
`vllm_lmcache_ssm` serves Qwen3.8-Flash-Next-FP8 (Gated DeltaNet + sparse GQA,
MTP) at 230 on 4 x H200 (TP4 + EP) with 800 GiB DRAM + 1500 GiB disk, and
`vllm_lmcache_swa_l2` serves Trinity-Large-Thinking-FP8-Block (hybrid
sliding-window, TP8 + EP) at 340 with 1500 GiB DRAM + 6000 GiB disk. The
competitors run the same stock vLLM image, model and budget as `vllm_lmcache`;
only the cache layer differs:

| deployment | suite | cache layer |
|---|---|---|
| `vllm_lmcache` | nightly, gates | LMCache MP server, L1 |
| `vllm_native` | weekly | vLLM `OffloadingConnector`, CPU tier |
| `vllm_simple` | weekly | vLLM `SimpleCPUOffloadConnector`, CPU |
| `vllm_mooncake` | weekly | Mooncake store, 4 clients, CPU pool |
| `vllm_lmcache_ssm` | weekly | LMCache, L1 + `fs_native` L2, on the hybrid linear-attention model |
| `vllm_lmcache_swa_l2` | weekly | LMCache, L1 + `fs_native` L2, on the hybrid sliding-window model |

Tested metrics: decode p10/p50 (tok/s per request) and TTFT p50/p95.
Recorded only: request count, errors, preemptions, prompt tokens by source
(and the cache layer's share), cache hit share (prompt tokens not computed)
against the ideal one (what an unbounded prefix cache would serve the same
prompts, to the token), input and output token throughput (tok/s over
the window, from vLLM's counters), spec decode accept length, L2 disk used,
the servers' block I/O during the run, the GPU KV cache size in tokens, and
the host's memory and scratch free space.

## How a regression is decided

`runner/baseline.py`, no per-metric thresholds: for each metric, runs of the
same config (`config_hash`) are tested for a latest stretch that ranks worse
than what came before (max Mann-Whitney statistic over all splits,
permutation p-value). The one setting is `[regression].false_alarm_rate`
(0.033 per gated run, about one a month). Consequences:

- a single bad night never fails the gate;
- a new config needs about 10 runs before it can fail;
- a confirmed shift starts a new baseline, so it fails the gate once.

## Host setup (once)

Host paths come from the environment:

| variable | what |
|---|---|
| `PRT_ROOT` | required: PRT state, history, caches, 6 TB of L2 scratch |
| `PRT_HF_CACHE` | an `HF_HOME` holding the model, default `/root/.cache/huggingface` |
| `PRT_RDMA_DEVICES` | comma-separated RDMA devices, needed by the weekly Mooncake |

```bash
# Models, at their [models] revisions. Runs are offline; a suite downloads a
# missing one first (runner/hf.py), so this only saves that time. Under
# $PRT_ROOT, a suite also deletes models no longer in [models].
HF_HOME=${PRT_HF_CACHE:-/root/.cache/huggingface} hf download \
    deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170
HF_HOME=${PRT_HF_CACHE:-/root/.cache/huggingface} hf download \
    arcee-ai/Trinity-Large-Thinking-FP8-Block \
    --revision 6412c1ad1588c664977b0f830b80a20273173318
HF_HOME=${PRT_HF_CACHE:-/root/.cache/huggingface} hf download \
    Qwen/Qwen3.8-Flash-Next-FP8 \
    --revision 236dfdf285828023ca3bcd3f37366c58a3469b13

# Traces, verified against the pinned sha256 and dir_digest. A suite builds
# a missing one first, in the serving image.
python3.11 perf_regression_testing/workloads/agentic/fetch_trace.py \
    build $PRT_ROOT/work/weka-062126-256k-dir
python3.11 perf_regression_testing/workloads/agentic/fetch_trace.py \
    build $PRT_ROOT/work/weka-062126-dir
```

Needs docker with the NVIDIA runtime, Python >= 3.11 (standard
library only; `fetch_trace.py build` needs `huggingface_hub`), and for
Mooncake RDMA devices and `gh`. A missing snapshot or variable exits 2.

## Running

```bash
python3.11 -m perf_regression_testing.suite --suite nightly   # or weekly
```

Reports go to `$PRT_ROOT/prt-suites/<suite>_<UTC time>/`, the history to
`$PRT_ROOT/prt-results/history.jsonl`. Buildkite runs the same command from
`.buildkite/perf_regression_testing/buildkite-pipeline.yml`, with schedules
setting `PRT_SUITE=nightly` or `weekly`; reports go to
`$PRT_ROOT/prt-suites/buildkite-<build number>-<suite>/` and are attached to
the build.
A manual build with `PRT_DEPLOYMENTS=vllm_simple` (comma-separated) in its
env runs only those of the suite's deployments (`--deployments`).
With `PRT_RESULTS_REPO` set, each suite also appends its runs to that repo's
`data/runs.jsonl` (`publish.py`), which a static GitHub Pages site renders.
Publishing is best effort and never changes the suite's exit code.

## Layout

| path | what |
|---|---|
| `config.toml` | every setting, with the reason for each non-default |
| `suite.py` | one suite: lock, images, runs, check, report |
| `publish.py` | append a suite's runs to the results repo's `data/runs.jsonl` |
| `runner/` | images, one run, scoring, sampler, regression check |
| `docker/` | serving, LMCache, Mooncake and AIPerf images |
| `aiperf_plugins/` | AIPerf trace framing and live-history replay |
| `workloads/agentic/` | trace fetch and verification |
| `tests/` | `python -m pytest perf_regression_testing/tests` (not collected by a repo-root `pytest`) |

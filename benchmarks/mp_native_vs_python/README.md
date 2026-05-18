# Native MP Benchmarks

This folder contains small Python-vs-native MP benchmark helpers for the
native C++ LMCache MP server work. The scripts write JSON reports and create
the output parent directory when `--output` points at a missing folder.

## Controller Latency

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/controller_latency.py \
    --request ping \
    --clients 1 \
    --iterations 100 \
    --output /tmp/lmcache-native-benchmarks/controller_latency.json
```

This benchmark compares Python and native MP controller-envelope latency. Use
`--request ping`, `--request noop`, `--request lookup-miss`, or
`--request lookup-fs-l2-partial`; the lookup modes send real token-key `LOOKUP`
requests that do not require registered CUDA KV cache. The filesystem-L2 mode
seeds one of two chunk metadata files before each Python/native server run, so
the path includes L2 metadata checks but not KV byte movement. Use `--clients N`
to run concurrent clients; each client sends `--iterations` requests. The
report includes mean, p50, p95, p99, raw latency samples, aggregate
`requests_per_s`, and best-effort `/proc` resource deltas for the MP server
process. It does not exercise KV byte movement.

## vLLM Reuse

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --compare-python \
    --steady-state-warmup-rounds 1 \
    --steady-state-rounds 2 \
    --output /tmp/lmcache-native-benchmarks/vllm_native_vs_python.json
```

This harness starts one MP server and runs one writer vLLM process followed by
reader vLLM processes that reuse the same prompt. Native summaries include MP
cache hits, misses, derived `cache_hit_rate`, retrieves, unsupported-request
counts, optional clean-stderr checks, TTFT when vLLM exposes request stats,
throughput summaries, and MP server `/proc` resource deltas. `--compare-python`
also captures per-worker MP traces for both server modes and adds
`mp_request_latency_ms` to the report, with client-observed mean, p50, p95,
p99, and raw latency samples for real vLLM `STORE`, `LOOKUP`, `RETRIEVE`, and
other MP request types seen in the worker trace.

Use `--mp-trace-output path.jsonl` to capture metadata-only JSONL rows for the
real MP requests emitted by the vLLM worker processes. The trace records
request types, request ids, token/key ranges, block ids, KV wrapper shapes,
layout hints, and response summaries; it intentionally does not persist
reusable CUDA IPC handles. Add `--require-mp-trace-lifecycle` to fail the smoke
unless the captured traffic includes the expected real vLLM
`REGISTER_KV_CACHE`/`STORE`/`LOOKUP`/`QUERY_PREFETCH_STATUS`/`RETRIEVE`
lifecycle and matching layerwise/layout hints.

Native runs can also add `--require-kvcache-checksum-match`. The harness then
keeps per-worker traces enabled, queries native `/kvcache/check` inside each
vLLM worker while its KV cache is still registered, and fails if a reader
`RETRIEVE` checksum differs from the writer `STORE` checksum for the same block
range. When `--mp-trace-output` is also set, the trace includes a
`vllm_kvcache_checksum_match` row containing the writer and reader checksum
responses; `tools/mp_trace_replay.py` validates that row so the real-vLLM
checksum evidence can be checked later even though CUDA IPC handles are not
reusable across processes. These trace summaries are also included in
`--compare-python` reports, even when `--mp-trace-output` is not requested.

The current reports are focused validation artifacts, not a full production
benchmark matrix. Broader workload sweeps can build on the same controller,
vLLM steady-state, trace-latency, and resource-delta fields.

Native C++ MP also has an optional single-round-trip lookup extension. Set
`lmcache.mp.lookup_with_result=true` in `kv_transfer_config.extra_config`, or
set `LMCACHE_MP_LOOKUP_WITH_RESULT=1`, to make the scheduler send
`LOOKUP_WITH_RESULT` instead of `LOOKUP` followed by `QUERY_PREFETCH_STATUS`.
This preserves the default Python-compatible protocol path unless explicitly
enabled. The first Qwen3-8B `long_doc_qa` experiment with this option preserved
output equality but regressed TTFT, so the saved goal-verifier artifacts keep
the default lookup/status path.

## Goal Optimization Artifact Audit

```bash
python3 benchmarks/mp_native_vs_python/verify_goal_optimization.py
```

For a machine-readable report:

```bash
python3 benchmarks/mp_native_vs_python/verify_goal_optimization.py \
  --json-output /tmp/goal-optimization-audit.json
```

This verifier reads the saved Qwen3-8B and Qwen3-14B `long_doc_qa` artifact
directories recorded in `../GOAL_OPTIMIZATION.md`. It checks the hard 2x TTFT
targets, Python/native response equality, successful cache-hit query rows,
native status counters, bounded native memory use, query-round parity, and the
saved Nsight Systems report. By default it also requires
`/tmp/lmcache-long-doc-qa-native-perf-stat.txt` to contain saved `perf stat`
output for cycles, context switches, CPU migrations, and cache misses. Pass
`--perf-stat-output path` to use a different file or
`--allow-missing-perf-stat` for a partial audit. It also requires saved
topology artifacts from `nvidia-smi topo -m` and `numactl --hardware`; use
`--nvidia-smi-topo-output path`, `--numactl-hardware-output path`, or
`--allow-missing-topology` when running a partial audit on a machine without
those captures. The default audit also checks the Qwen3-8B adapter-timing
diagnostic artifact at
`/tmp/lmcache-long-doc-qa-qwen3-8b-native-adaptertiming-1779048612`; use
`--qwen3-8b-adapter-timing-dir path` for a different artifact or
`--allow-missing-adapter-timing` for a partial audit. It also checks the
Qwen3-8B and Qwen3-14B vLLM-only control artifacts at
`/tmp/lmcache-long-doc-qa-qwen3-8b-vllmonly-1779049198` and
`/tmp/lmcache-long-doc-qa-qwen3-14b-vllmonly-1779052865`; use
`--qwen3-8b-vllm-only-dir path`, `--qwen3-14b-vllm-only-dir path`, or
`--allow-missing-vllm-only` when needed.
It also checks the Qwen3-8B lookup/status timing artifact at
`/tmp/lmcache-long-doc-qa-qwen3-8b-native-lookuptiming-1779049672`; use
`--qwen3-8b-lookup-timing-dir path` or `--allow-missing-lookup-timing` for a
partial audit.

The verifier is expected to exit non-zero for the current artifacts because
Gate 3 is not met for either target model and Gate 2 is still not met for
Qwen3-8B. It should pass only after native C++ MP is no slower than Python MP
on TTFT, reaches the `<=35.237ms` and `<=37.645ms` TTFT targets for Qwen3-8B
and Qwen3-14B, and preserves query-round parity under the same benchmark contract,
with all required OS/CUDA evidence present. The JSON
report includes top-level aggregate
`gates`, per-model `gates`, and a top-level `completion` object so automation
can distinguish a completed goal from the current blocked state and read the
allowed next decisions. Model reports also include `native_cuda_gpu_hot_cache`,
`native_transfer_lock_timing`, and `native_request_type_latency` so readers can
see whether the saved native `/status` artifact carried the hot-cache,
transfer-lock, and request-type timing schemas. It also records
`native_request_type_queue_wait` when saved artifacts carry worker queue-wait
metrics, plus `native_lookup_result_fast_path_count` when saved artifacts carry
the completed lookup-result fast-path counter. These fields help separate
execution time from time spent waiting behind earlier native tasks. The report
also records zero-retrieve TTFT floors derived
from saved query timing and native retrieve counters, so readers can see
whether the remaining TTFT gap is still reachable by native retrieve
optimization. Per-model reports include Python/native vLLM log markers for
eager mode, disabled CUDA graphs, and inference-time Triton JIT warnings, which
are shared path evidence rather than native-server evidence. The top-level
`evidence.adapter_timing` report records whether the diagnostic artifact is
non-layerwise, response-identical, and below the configured retrieve-completion
wait threshold. The `evidence.vllm_only_controls` report records the
no-LMCache control TTFT, output equality, and absence of `kv_transfer_config`
for both target models.
The `evidence.lookup_timing` report records lookup submit/status timing,
query cache-hit lookup totals, retrieve completion timing, response equality,
and non-layerwise registration for the lookup-timing diagnostic artifact.

## Five-Run Python-vs-Native Gate

`five_run_compare.py` aggregates saved benchmark artifacts and applies the
active `../GOAL.md` rule: native C++ MP must beat Python MP on every comparable
measured unit after at least five runs per side.

```bash
python3 benchmarks/mp_native_vs_python/five_run_compare.py \
  --controller-report '/tmp/controller-run-*.json' \
  --vllm-report '/tmp/vllm-smoke-run-*.json' \
  --long-doc-run python:Qwen3-8B:/tmp/qwen3-8b-python-run-1 \
  --long-doc-run native:Qwen3-8B:/tmp/qwen3-8b-native-run-1 \
  --min-runs 5 \
  --json-output /tmp/lmcache-native-five-run-summary.json
```

The helper does not start servers or run benchmarks. It reads the JSON reports
from `controller_latency.py` and `vllm_native_smoke.py`, plus long-doc artifact
directories containing `bench.stdout`, `warmup_round.csv`, `query_round.csv`,
and `responses.txt`. It exits non-zero when a metric is incomplete or failing.
Use `--no-fail` only when producing an exploratory report from known-failing
artifacts.

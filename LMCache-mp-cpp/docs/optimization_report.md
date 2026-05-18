# Native MP Optimization Report

Date: 2026-05-18

## Objective

Follow `GOAL_OPTIMIZATION.md`: make native C++ LMCache MP at least 2x faster
than Python MP on cache-hit TTFT for the same vLLM `long_doc_qa` workload,
while keeping generated outputs byte-identical.

## Current Artifacts

Python MP baselines:

- Qwen3-8B: `/tmp/lmcache-long-doc-qa-qwen3-8b-python-1779034941`
- Qwen3-14B: `/tmp/lmcache-long-doc-qa-qwen3-14b-python-1779035109`

Native MP optimized runs with explicit `--cuda-gpu-hot-cache`,
stream-ordered CUDA future waits, and the completed lookup-result fast path:

- Qwen3-8B: `/tmp/lmcache-long-doc-qa-qwen3-8b-native-faststatus-nolog-1779050718`
- Qwen3-14B: `/tmp/lmcache-long-doc-qa-qwen3-14b-native-faststatus-nolog-1779050813`

Qwen3-8B connector-timing diagnostic run:

- `/tmp/lmcache-long-doc-qa-qwen3-8b-native-adaptertiming-1779048612`

Qwen3-8B lookup/status timing diagnostic run:

- `/tmp/lmcache-long-doc-qa-qwen3-8b-native-lookuptiming-1779049672`

Qwen3-8B vLLM-only control run:

- `/tmp/lmcache-long-doc-qa-qwen3-8b-vllmonly-1779049198`

Qwen3-8B native lookup-with-result experiment:

- `/tmp/lmcache-long-doc-qa-qwen3-8b-native-lookupresult-1779052255`

OS/CUDA profiler artifacts:

- Nsight Systems: `/tmp/lmcache-native-roundtrip-nsys.nsys-rep`
- `perf stat`: `/tmp/lmcache-long-doc-qa-native-perf-stat.txt`
- GPU topology: `/tmp/lmcache-native-topology-nvidia-smi-topo.txt`
- NUMA topology: `/tmp/lmcache-native-topology-numactl-hardware.txt`
- Perf-backed native Qwen3-8B rerun:
  `/tmp/lmcache-long-doc-qa-qwen3-8b-native-perf-1779040972`

All runs used:

- vLLM `0.21.0`
- `uv run --python 3.12`
- `--document-length 512 --num-documents 2 --repeat-count 2`
- `--repeat-mode tile --output-len 16 --max-inflight-requests 1`
- `--sleep-time-after-warmup 1 --completions --json-output`
- `--max-model-len 2048 --enforce-eager --no-enable-prefix-caching`
- LMCache MP `--l1-size-gb 2.0 --eviction-policy LRU --chunk-size 32`

## Results

| Model | Server | Query TTFT / Prompt | Query Round / Prompt | Warmup Round / Prompt |
|---|---|---:|---:|---:|
| Qwen3-8B | Python MP | 0.070475s | 0.243756s | 0.215865s |
| Qwen3-8B | Native MP optimized | 0.071207s | 0.237265s | 0.224246s |
| Qwen3-14B | Python MP | 0.075290s | 0.345624s | 0.346290s |
| Qwen3-14B | Native MP optimized | 0.069478s | 0.339667s | 0.340703s |

Per-prompt query TTFT from the preserved `query_round.csv` files:

| Model | Server | Prompt 0 | Prompt 1 | Prompt 2 | Prompt 3 | Mean |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | Python MP | 0.122131s | 0.054062s | 0.053541s | 0.052165s | 0.070475s |
| Qwen3-8B | Native MP optimized | 0.119482s | 0.055261s | 0.054861s | 0.055223s | 0.071207s |
| Qwen3-14B | Python MP | 0.125290s | 0.059764s | 0.058168s | 0.057937s | 0.075290s |
| Qwen3-14B | Native MP optimized | 0.123087s | 0.052307s | 0.050558s | 0.051959s | 0.069478s |

Correctness:

- Qwen3-8B Python/native `responses.txt` are byte-identical.
- Qwen3-14B Python/native `responses.txt` are byte-identical.
- Both native optimized runs completed with `7 STORE`, `4 RETRIEVE`, and
  `11 LOOKUP`.
- Both native optimized runs had `0` unsupported requests and `0`
  transfer-lock failures.
- Both native optimized runs served all `11` completed lookup-result status/hit
  checks through the native fast path.

## CUDA And OS Evidence

Native CUDA transfer moved from CPU-staged `cudaMemcpy` to a native CUDA
block-transfer kernel plus an explicit same-GPU hot chunk cache.

Nsight Systems focused profile:

- Report: `/tmp/lmcache-native-roundtrip-nsys.nsys-rep`
- Command: focused `test_native_cuda_binary_round_trips_pytorch_cuda_ipc_store_retrieve`
  with `LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-nsys-wrapper.sh`
- CUDA GPU summary: 3 native `TransferKernel` launches and 6 small H2D memops
  for kernel metadata.
- Kernel time: 12.384us total GPU time across 3 launches.
- CUDA GPU memcpy time: 6.240us total, all host-to-device metadata copies.
- There were no GPU D2H/H2D payload memcpy operations on the hot path.

Native status counters for the flagged optimized runs:

| Model | Retrieve Total | Store Total | Copy Total | Cache Total | Open Tensors | CUDA memcpy calls | CUDA kernel calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 7.627ms | 963.371ms | 123.414ms | 459.466ms | 4.873ms | 0 | 112 |
| Qwen3-14B | 8.033ms | 1559.678ms | 193.214ms | 776.627ms | 5.468ms | 0 | 112 |

Worker queue-wait counters from the same native status snapshots:

| Model | LOOKUP Queue Wait Total | LOOKUP Queue Wait Max | STORE Queue Wait Total | RETRIEVE Queue Wait Total |
|---|---:|---:|---:|---:|
| Qwen3-8B | 298.735ms | 180.412ms | 1.035ms | 0.672ms |
| Qwen3-14B | 235.002ms | 233.863ms | 1.203ms | 0.308ms |

I also tested `--max-workers 2` on Qwen3-8B to see whether a wider fixed
worker pool could reduce LOOKUP queueing. It regressed query TTFT to
`0.088754s` and query-round time to `0.258613s`, with output still
byte-identical, so the current report keeps the single-worker artifacts as the
best native evidence.

The connector-timing diagnostic run used the same Qwen3-8B benchmark contract
with `LMCACHE_MP_CONNECTOR_TIMING=1`. It preserved byte-identical output and
measured query TTFT `0.071712s`. vLLM registered `use_layerwise=false`, so this
run exercised the non-layerwise `LMCacheMPWorkerAdapter.get_finished()`
retrieve completion path. The four retrieve completion logs reported:

| Prompt retrieve | `query_us` | `result_us` |
|---|---:|---:|
| 0 | 26 | 5 |
| 1 | 16 | 2 |
| 2 | 25 | 2 |
| 3 | 17 | 1 |

Total connector-side retrieve completion wait was `94us` across all four query
retrieves. The tens-of-milliseconds TTFT floor is therefore not hidden in the
Python connector's retrieve completion check.

The lookup/status timing diagnostic run used the same Qwen3-8B benchmark
contract with `LMCACHE_MP_CONNECTOR_TIMING=1` and preserved byte-identical
output. It measured query TTFT `0.071300s` and query round `0.235560s`.
The log contained all `11` lookup submit rows and `11` lookup status rows.
For the four query cache-hit rows, lookup submit totaled `12.007ms`, lookup
status totaled `11.817ms`, and retrieve completion totaled `115us`.

This shows a real per-query coordination cost of about `5.956ms/prompt`, but it
does not close the hard target gap: subtracting that cost from the observed
native TTFT would still leave Qwen3-8B far above `35ms`. The same diagnostic
also shows one `222.020ms` lookup submit wait on a `1002`-token non-cache-hit
lookup before the query cache-hit rows, which matches the native queue-wait
evidence but does not explain the repeated-hit TTFT floor by itself.

I added an optional append-only native `LOOKUP_WITH_RESULT` protocol extension
to test whether removing the status-poll round trip improves the end-to-end
goal metric. The vLLM connector enables it with
`lmcache.mp.lookup_with_result=true` or `LMCACHE_MP_LOOKUP_WITH_RESULT=1`. The
Qwen3-8B run at
`/tmp/lmcache-long-doc-qa-qwen3-8b-native-lookupresult-1779052255` preserved
byte-identical output and kept `0` unsupported requests and `0` transfer-lock
failures, but regressed query TTFT to `0.084431s` and query round to
`0.290588s`. Therefore the canonical artifact remains the faster
fast-status-native run, and single-round-trip lookup is not a finish path for
Gate 2 or Gate 3.

The Qwen3-8B vLLM-only control used the same model, prompt shape,
`--enforce-eager`, `--no-enable-prefix-caching`, completions API, and output
contract, but no LMCache connector. It measured query TTFT `0.052862s`, query
round `0.225080s`, and byte-identical output. Per-prompt TTFT was:

| Prompt | TTFT |
|---|---:|
| 0 | 0.101243s |
| 1 | 0.036839s |
| 2 | 0.037408s |
| 3 | 0.035960s |

The fastest vLLM-only prompt remains above the `0.035237s` Qwen3-8B hard target.
This is diagnostic context for shared OpenAI/vLLM/model latency; the primary
speedup comparator remains warm-cache native LMCache versus warm-cache Python
LMCache.

The Qwen3-14B vLLM-only control used the same no-LMCache contract. The artifact
at `/tmp/lmcache-long-doc-qa-qwen3-14b-vllmonly-1779052865` measured query
TTFT `0.068589s`, query round `0.342607s`, and byte-identical output.
Per-prompt TTFT was:

| Prompt | TTFT |
|---|---:|
| 0 | 0.110802s |
| 1 | 0.054465s |
| 2 | 0.055083s |
| 3 | 0.054006s |

The fastest vLLM-only prompt remains above the `0.037645s` Qwen3-14B hard target.
This gives the same diagnostic context for both requested target models.

The strongest CUDA-side diagnostic was a Qwen3-8B vLLM-only run without
`--enforce-eager`, allowing vLLM to use torch compile and CUDA graphs. The
artifact at `/tmp/lmcache-long-doc-qa-qwen3-8b-vllmonly-cudagraph-1779053232`
measured query TTFT `0.047651s`, query round `0.191460s`, and byte-identical
output. The fastest prompt was `0.038978s`, still above the `0.035237s` hard
target. This confirms CUDA graphs help, but this contract-changing lever is not
enough by itself and is not eligible for the current eager-mode verifier.

Local scheduler/eBPF feasibility is limited without privileged setup:
`taskset` and `numactl` are available, but `chrt -f 1 true` fails with
`Operation not permitted`, `ulimit -r` is `0`, `bpftrace` reports that it must
run as root, and `/proc/sys/kernel/perf_event_paranoid` is `2`. That makes CPU
affinity and NUMA binding immediately testable, while realtime priority,
eBPF scheduler telemetry, and eBPF-based hinting require root, cgroup policy,
or `sched_ext` setup before they can be used as benchmark evidence.

I tested the immediately available affinity/NUMA feature in isolation by
wrapping both native MP and `vllm serve` with
`taskset -c 0-31,64-95 numactl --cpunodebind=0 --membind=0`:

| Model | Artifact | Query TTFT | Query Round | Result |
|---|---|---:|---:|---|
| Qwen3-8B | `/tmp/lmcache-long-doc-qa-qwen3-8b-native-numa-affinity-1779054163` | `0.070302s` | `0.237878s` | byte-identical output; slight TTFT improvement |
| Qwen3-14B | `/tmp/lmcache-long-doc-qa-qwen3-14b-native-numa-affinity-1779054310` | `0.073281s` | `0.343231s` | byte-identical output; TTFT regression |

Both runs preserved native status correctness (`7 STORE`, `4 RETRIEVE`,
`11 LOOKUP`, `0` unsupported requests, and `0` transfer-lock failures). The
feature-specific verifier output at
`/tmp/goal-optimization-affinity-feature2.json` reports `Gate 2=true` for both
models, but `Gate 3=false` for both models, with only `1.002x` Qwen3-8B and
`1.027x` Qwen3-14B speedup versus warm-cache Python LMCache. The effect is
useful deployment tuning evidence, but it is too small and inconsistent to be
the Gate 3 finish path.

I then built a pipeline-overhead audit before continuing with more individual
features:

- Artifact: `/tmp/goal-optimization-pipeline-overhead.json`
- Qwen3-8B native TTFT: `71.207ms`
- Qwen3-8B vLLM-only control TTFT: `52.862ms`
- Qwen3-8B native retrieve transfer: `1.907ms/prompt`
- Qwen3-8B lookup submit plus status diagnostic: `5.956ms/prompt`
- Qwen3-14B native TTFT: `69.478ms`
- Qwen3-14B vLLM-only control TTFT: `68.589ms`
- Qwen3-14B native retrieve transfer: `2.008ms/prompt`

The biggest measured bucket is therefore the shared vLLM/client/model
first-token path, not native CUDA retrieve. A retrieve-only or NUMA-only patch
cannot close the revised 2x TTFT target. The next useful work is to instrument
the vLLM first-token path more deeply, or explicitly change the benchmark
contract for exact-shape warmup/CUDA graphs and rerun both Python and native
baselines.

Current native `/status` and Prometheus output also expose transfer-lock wait
and hold total/max metrics plus per-request-type latency and worker queue-wait
summaries for LOOKUP, STORE, RETRIEVE, and FREE_LOOKUP_LOCKS. The saved
long-doc QA status artifacts predate some of those schemas, so the verifier
records whether the saved artifact contains those fields instead of requiring
them retroactively.

Perf-backed native Qwen3-8B rerun:

- Artifact directory:
  `/tmp/lmcache-long-doc-qa-qwen3-8b-native-perf-1779040972`
- Summary: `query_ttft_per_prompt=0.073480s`,
  `query_round_time_per_prompt=0.244257s`.
- Native status: `7 STORE`, `4 RETRIEVE`, `11 LOOKUP`, `0` unsupported,
  `0` transfer-lock failures, `112` CUDA kernel calls, `0` CUDA memcpy calls.
- `perf stat` command used
  `/usr/lib/linux-tools/6.17.0-23-generic/perf` because `/usr/bin/perf`
  reports missing tools for the running `6.6.0` kernel.
- Captured counters for the long-doc QA client process: `13,432,640,235`
  cycles, `0` context switches, `0` CPU migrations, and `4,352,639` cache
  misses over `7.354511871s` elapsed. System-wide perf was blocked by
  `perf_event_paranoid=2`.

Per-prompt retrieve overhead is therefore about `1.9-2.0ms` for each model.
Even a perfect zero-cost native retrieve would only reduce mean TTFT from
`71.2ms` to `69.3ms` on Qwen3-8B, and from `69.5ms` to `67.5ms` on Qwen3-14B.
The fastest prompt in each native query round still has a conservative
zero-retrieve floor of `52.9ms` for Qwen3-8B and `48.4ms` for Qwen3-14B after
subtracting the maximum native retrieve time
observed in the run. The Gate 3 target requires `<=35.237ms` and `<=37.645ms`,
respectively, so the missing time is not in native MP retrieve anymore.

Host topology captured during the run:

- GPU: NVIDIA H100 PCIe
- GPU CPU affinity: `0-31,64-95`
- GPU NUMA affinity: node `0`
- NUMA node 0 free memory was over `700GB`
- `nvidia-smi topo -m` reported the GPU on the host PCIe system path

vLLM log evidence shows the remaining TTFT path is shared with Python MP:

- `--enforce-eager` disables torch compile and CUDA graphs.
- vLLM reports `Cudagraph is disabled under eager mode`.
- vLLM reported Triton JIT during inference for `_compute_slot_mapping_kernel`.
- Both Python and native runs use the same vLLM model path, prompt shape, and
  OpenAI-compatible HTTP benchmark client.
- `verify_goal_optimization.py` records these vLLM log markers for each saved
  Python/native artifact as machine-readable shared-path evidence.

Source-path audit:

- `benchmarks/long_doc_qa/long_doc_qa.py` measures TTFT from client request
  start until the first streamed completion chunk with content. This includes
  OpenAI HTTP handling, vLLM scheduling, KV connector work, first-token model
  execution, and response streaming.
- `lmcache/integration/vllm/lmcache_mp_connector.py` submits native cache-hit
  retrieves in `start_load_kv`. The current benchmark uses `use_layerwise=false`,
  so retrieve completion is observed through `get_finished()`; the optional
  `wait_for_layer_load` hook uses `result_on_current_stream()` when layerwise
  mode is enabled.
- `lmcache/integration/vllm/vllm_multi_process_adapter.py` sends the actual
  `RETRIEVE` request and tracks the retrieve future. Fresh timing logs show
  completed retrieve future checks take microseconds. After that future
  completes, the remaining first-token work is normal vLLM/model execution.
- Fresh lookup/status timing logs show query cache-hit lookup/status costs are
  milliseconds, not microseconds, but still too small to explain the gap from
  `71.2ms` TTFT to the `35ms` hard target.
- Native C++ MP can reduce the retrieve future cost, and the optimized runs do
  reduce it to about `1.9-2.0ms/prompt`. The current native warm-cache TTFT is
  still only `0.99x` of Python warm-cache TTFT on Qwen3-8B and `1.08x` on
  Qwen3-14B, so the required `2x` relative improvement is not present.

## Gate Audit

| Requirement | Evidence | Status |
|---|---|---|
| Gate 1: native at least 3x faster than old native | 8B: `0.520s -> 0.0712s`; 14B: `0.543s -> 0.0695s` | Passed |
| Gate 2: native no slower than Python | 14B passes; 8B remains slightly slower than Python TTFT | Not met |
| Gate 3: native at least 2x faster than Python TTFT | Required `<=0.035237s` and `<=0.037645s`; measured `0.0712s` and `0.0695s` | Not met |
| Gate 4: native query-round no worse than Python | 8B `0.2373s` vs `0.2438s`; 14B `0.3397s` vs `0.3456s` | Passed |
| Output equality | Python/native response files match byte-for-byte for both models | Passed |
| No unsupported/lock failures | Native status has `0` for both counters | Passed |
| Default-off fast path | Same-GPU hot cache requires `--cuda-gpu-hot-cache` | Passed |
| Bounded memory observability | `/status` reports `cuda_gpu_hot_cache.entries` and `.bytes` | Implemented |

Warm-cache relative score:

| Model | Overall Native/Python Speedup | Steady-State Speedup Excluding First Query |
|---|---:|---:|
| Qwen3-8B | `0.99x` | `0.97x` |
| Qwen3-14B | `1.08x` | `1.14x` |

## Completion Audit

The active goal is not complete.

Reproducible audit command:

```bash
python3 benchmarks/mp_native_vs_python/verify_goal_optimization.py
```

Current result: exit code `1`, with the hard 2x TTFT requirements failing and
the required profiler artifacts present.

Machine-readable audit:

```bash
python3 benchmarks/mp_native_vs_python/verify_goal_optimization.py \
  --json-output /tmp/goal-optimization-audit.json
```

Current JSON artifact: `/tmp/goal-optimization-audit.json`, with
`passed=false`, profiler evidence present, Gate 4 passing for both target
models, Qwen3-14B passing Gate 2, Qwen3-8B still failing Gate 2, and both
models failing Gate 3. The same JSON now records adapter timing, lookup/status
timing, and vLLM-only control evidence for both target models.

| `GOAL_OPTIMIZATION.md` requirement | Artifact evidence | Audit result |
|---|---|---|
| 2x faster native TTFT than Python for Qwen3-8B | Python `0.070475s`; native `0.071207s`; hard target `<=0.035237s` | Missing |
| 2x faster native TTFT than Python for Qwen3-14B | Python `0.075290s`; native `0.069478s`; hard target `<=0.037645s` | Missing |
| Query-round no worse than Python or explained lower bound | 8B native `0.237265s` vs Python `0.243756s`; 14B native `0.339667s` vs Python `0.345624s` | Passed |
| Byte-identical generated output | `cmp` succeeds for Python/native `responses.txt` on both models | Passed |
| Matching request success/count semantics | each query CSV has 4 successful query rows; native status has `7 STORE`, `4 RETRIEVE`, `11 LOOKUP` | Passed |
| No unsupported requests | native status `metrics.unsupported_count == 0` for both models | Passed |
| No transfer-lock failures | native status `metrics.transfer_lock_failure_count == 0` for both models | Passed |
| Bounded native memory | native status reports DRAM bytes below 2GiB capacity for both models and no disk spill | Passed |
| CUDA profiler evidence | Nsight report captured and verifier checks `/tmp/lmcache-native-roundtrip-nsys.nsys-rep` exists | Passed |
| OS perf-counter evidence | Saved `perf stat` artifact includes cycles, context switches, CPU migrations, and cache misses for a Qwen3-8B native benchmark-window rerun; system-wide perf is blocked by `perf_event_paranoid=2`, so this is per-process user-space evidence | Partial |
| Qwen3-8B lookup/status timing evidence | Diagnostic artifact records `23.824ms` total query cache-hit lookup/status time and `115us` retrieve completion time across 4 query prompts | Passed |
| Default-off optimized path | same-GPU hot cache requires explicit `--cuda-gpu-hot-cache` | Passed |

## Conclusion

The native MP bottleneck that caused the original 7x slowdown is fixed. Native
cache-hit retrieval is now about `1.9-2.0ms` per prompt. Under the current
`GOAL_OPTIMIZATION.md` benchmark contract, Gate 3 is still not met because
native warm-cache TTFT is roughly parity with Python warm-cache TTFT rather
than `2x` faster. Reaching the required relative improvement needs a new
native-controlled mechanism or an explicit benchmark-contract change followed
by fresh Python/native baselines.

# CacheBlend CI Verification Inventory

Date: 2026-06-04, Asia/Taipei.

This document is an evidence inventory for PR
<https://github.com/LMCache/LMCache/pull/3255>. It separates the current
LMCache/vLLM direction from the repo harnesses, and it states exactly what each
test path proves. It intentionally does not treat a generic LMCache MP smoke as
proof that CacheBlend works end to end.

## Sources Checked

Local repo sources:

- `.buildkite/k3_tests/README.md`
- `.buildkite/k3_tests/blend/BK_WEB_SETUP.md`
- `.buildkite/k3_tests/blend/pipeline.yml`
- `.buildkite/k3_tests/blend/run.sh`
- `.buildkite/k3_tests/blend/scripts/run-blend-test.sh`
- `.buildkite/k3_tests/{unit,correctness,integration,multiprocess,comprehensive,sglang}/pipeline.yml`
- `docs/source/getting_started/quickstart.rst`
- `docs/source/mp/index.rst`
- `docs/source/mp/configuration.rst`
- `docs/source/mp/architecture.rst`
- `lmcache/v1/multiprocess/server.py`
- `lmcache/v1/multiprocess/modules/blend.py`
- `lmcache/v1/mp_observability/subscribers/metrics/cb_server.py`

Online sources checked:

- LMCache MP mode docs: <https://docs.lmcache.ai/mp/index.html>
- LMCache MP configuration docs: <https://docs.lmcache.ai/mp/configuration.html>
- LMCache quickstart: <https://docs.lmcache.ai/getting_started/quickstart.html>
- LMCache dynamic connector docs:
  <https://docs.lmcache.ai/api_reference/dynamic_connector.html>
- LMCache Kubernetes operator docs: <https://docs.lmcache.ai/mp/operator.html>
- LMCache TensorRT-LLM docs:
  <https://docs.lmcache.ai/integrations/tensorrt_llm.html>
- LMCache CLI server docs: <https://docs.lmcache.ai/cli/server.html>
- vLLM `LMCacheMPConnector` API docs:
  <https://docs.vllm.ai/en/v0.19.0/api/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector/>
- CacheBlend paper: <https://arxiv.org/abs/2405.16444>

## Current Direction

LMCache MP mode is the documented recommended deployment shape for vLLM. LMCache
runs as a standalone service and vLLM attaches through `LMCacheMPConnector`.
The documented advantages are process isolation, shared cache across multiple
vLLM instances, independent CPU/GPU scaling, L1 plus L2 storage, and built-in
observability.

The documented server entrypoint direction is:

```bash
lmcache server
```

The legacy ZMQ-only entrypoint remains:

```bash
python -m lmcache.v1.multiprocess.server
```

Both entrypoints share `--engine-type`. For CacheBlend, the current documented
server path is the MP server with:

```bash
--engine-type blend
```

In current repo code, that appends `BlendModule` in
`lmcache/v1/multiprocess/server.py`. There is no current
`lmcache/v1/multiprocess/blend_server_v2.py` file in this checkout.
The online MP docs still describe `blend_server_v2` as a legacy entrypoint in
some rendered pages, so this is a documented drift to treat carefully: current
repo state wins for this PR's executable harness.

For vLLM connector resolution, the docs explicitly distinguish upstream and
LMCache-shipped connectors:

- `LMCacheMPConnector` defaults to vLLM's built-in connector.
- On vLLM versions that support dynamic connector loading, the LMCache-shipped
  implementation should be selected with:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both"
}
```

That matters for CacheBlend because the K3 script uses a prefiller connector
named `LMCacheMPCBConnector`. A public vLLM image may not know that connector
unless the image or `kv_connector_module_path` registers it.

## LMCache Service And Direction Inventory

This section interprets "all services/directions" as the deployable service
surfaces, engine integrations, and storage directions that matter for deciding
what the tests prove.

| Surface | Direction / role | Evidence source | What proves it works | What does not prove it |
| --- | --- | --- | --- | --- |
| vLLM in-process | LMCache runs inside the vLLM process through `LMCacheConnectorV1` or dynamic connector loading. | `docs/source/getting_started/quickstart.rst`, LMCache dynamic connector docs. | A vLLM process serves repeated-prefix requests and logs LMCache store/retrieve from inside the engine. | MP server health, K3 MP tests, or CacheBlend proxy readiness. |
| vLLM MP service | `lmcache server` runs as a standalone ZMQ plus HTTP service; vLLM attaches via `LMCacheMPConnector`. | `docs/source/mp/index.rst`, `docs/source/cli/server.rst`, LMCache MP docs. | LMCache server starts, vLLM connector registers KV caches, OpenAI requests complete, MP metrics/logs show lookup/store/retrieve. | In-process `LMCacheConnectorV1` behavior or unit-only subscriber tests. |
| CacheBlend MP | MP server uses `--engine-type blend`; `BlendModule` adds non-prefix CB operations and metrics. | `docs/source/mp/configuration.rst`, `docs/source/mp/architecture.rst`, `lmcache/v1/multiprocess/modules/blend.py`, CacheBlend paper. | The full prefiller/decoder/proxy workload reaches `shuffle_doc_qa.py`, emits CB store/lookup/retrieve/final events, and exposes raw `lmcache_blend_*` metrics. | Standard prefix-cache hits, decoder-only MP connector startup, or generic MP metrics. |
| SGLang integration | SGLang uses LMCache MP and SGLang-specific cache/radix behavior. | `docs/source/getting_started/quickstart.rst`, `.buildkite/k3_tests/sglang/*`. | SGLang correctness run proves output is unchanged and LMCache is exercised; performance run proves lower TTFT for that SGLang path. | vLLM connector behavior, CacheBlend prefiller connector behavior, or TensorRT-LLM behavior. |
| TensorRT-LLM integration | Online docs describe in-process and MP connector modes through TRT-LLM's KV Cache Connector API. Local recipe pages still mark TRT-LLM support as in progress for specific models. | Online TensorRT-LLM docs, `lmcache/integration/tensorrt_llm/*`, `docs/source/getting_started/quickstart.rst`, recipe docs. | A TRT-LLM run with the LMCache connector completes lookup/retrieve/store and logs/metrics cache hits. | vLLM or SGLang tests; current K3 pipelines do not cover TRT-LLM. |
| Kubernetes operator | `LMCacheEngine` CR reconciles DaemonSet, Service, ConfigMap, resources, and optional ServiceMonitor for MP servers. | `docs/source/mp/operator.rst`, online operator docs. | Operator install/reconcile test creates expected K8s objects and vLLM consumes the generated connection ConfigMap. | Plain `lmcache server` local startup or Buildkite job scheduling. |
| L2 storage adapters | MP server can back L1 with persistent L2 adapters: `nixl_store`, `nixl_store_dynamic`, `fs`, `dax`, and native/remote adapters. | `docs/source/mp/l2_storage.rst`, `docs/source/mp/configuration.rst`, design docs under `docs/design/v1/distributed/l2_adapters/`. | L2-specific tests/benchmarks show store, prefetch/load, eviction, and restart/persist behavior for the configured adapter. | L1-only MP tests or CacheBlend tests without an L2 adapter. |
| Observability service | EventBus subscribers expose Prometheus metrics, logs, and optional traces for MP and CB events. | `docs/source/mp/observability.rst`, `lmcache/v1/mp_observability/*`. | Raw Prometheus scrape, subscriber tests, and, for E2E, service logs showing the relevant event path was exercised. | Presence of metric definitions alone. |
| CLI service tools | `lmcache describe`, `ping`, `kvcache`, `bench server`, and `bench engine` talk to LMCache HTTP/server or engine endpoints. | `docs/source/cli/*.rst`, `lmcache/cli/*`. | CLI commands run against a live endpoint and report expected status/metrics. | Full vLLM/SGLang/TRT or CacheBlend semantics unless the CLI workload explicitly drives them. |

## What CacheBlend Is Testing

The CacheBlend paper frames the problem as non-prefix KV reuse for RAG-like
inputs where multiple reused text chunks are not always the prompt prefix.
CacheBlend reuses precomputed KV caches and selectively recomputes a subset of
tokens so the composed cache can approach full-prefill quality with lower TTFT.

That is a different claim from standard prefix caching. A valid CacheBlend test
must prove non-prefix reuse through the CacheBlend path, not only that ordinary
LMCache MP store/retrieve works.

## Repo Harness Inventory

| Harness | Local entrypoint | Image / GPU | What it actually proves | What it does not prove |
| --- | --- | --- | --- | --- |
| Unit | `.buildkite/k3_tests/unit/run.sh` | `lmcache/ci-base:latest`, 1 GPU | Python/CUDA unit coverage, including observability subscriber unit tests and targeted multiprocess tests. | Full vLLM service behavior, full CacheBlend benchmark, K3 service orchestration. |
| Correctness | `.buildkite/k3_tests/correctness/run.sh` | `lmcache/ci-base:latest`, 1 GPU | LMCache output equivalence against base vLLM on the configured dataset flow. | CacheBlend full prefill/prefill-decoder proxy path, Prometheus CB metric emission. |
| Integration CPU/Disk | `.buildkite/k3_tests/integration/run.sh cpu|disk` | `lmcache/ci-base:latest`, 1 GPU | Direct vLLM plus LMCache API behavior for CPU and disk backends. | Multiprocess K3 topology and CacheBlend. |
| Multiprocess | `.buildkite/k3_tests/multiprocess/run.sh <test>` | `lmcache/ci-base:latest`, 1 or 2 GPUs; CPU-only variants for SHM/pickle | MP service behavior across vLLM bench, long-doc QA, L2, deadlock, lm_eval, cache stats, HTTP API, restart/fault cases, and CPU E2E transfer modes. | CacheBlend-specific prefiller `LMCacheMPCBConnector`, blend proxy, non-prefix CB benchmark. |
| Comprehensive | `.buildkite/k3_tests/comprehensive/run.sh <config>` | `lmcache/ci-base:latest`, 1 or 2 GPUs | Parallel broader benchmark configs: PD, P2P, v3 variants, local CPU, local disk, async, multi-device, layerwise. | The dedicated CacheBlend K3 image and CB connector path. |
| SGLang | `.buildkite/k3_tests/sglang/run.sh correctness|perf` | `lmcache/ci-base:latest`, 1 GPU | SGLang plus LMCache correctness and TTFT performance checks. | vLLM CacheBlend and vLLM MP connector behavior. |
| Blend / CacheBlend | `.buildkite/k3_tests/blend/run.sh` -> `scripts/run-blend-test.sh` | `tensormesh/cacheblend:latest`, 2 GPUs | Intended full CacheBlend service workload: blend server, prefiller, decoder, proxy, `shuffle_doc_qa.py`, log error scan, and branch-local `lmcache_blend_*` metrics scrape. | It still depends on the image/runtime knowing the prefiller `LMCacheMPCBConnector`. |

Coverage gaps by service surface:

- No current K3 pipeline in this checkout proves TensorRT-LLM integration.
- No current K3 pipeline in this checkout proves Kubernetes operator reconcile
  behavior.
- L2 is covered only where a specific harness config enables it
  (`long_doc_qa_l2`, comprehensive L2 configs, or adapter-specific tests).
- CacheBlend is covered only by the dedicated Blend harness, not by generic MP,
  integration, correctness, or unit jobs.

## CacheBlend K3 Harness Details

The maintained repo path for the preconfigured image workflow is:

```bash
bash .buildkite/k3_tests/blend/run.sh
```

Buildkite configuration:

- Queue: `k8s`
- Image: `tensormesh/cacheblend:latest`
- GPU limit: `nvidia.com/gpu: "2"`
- Timeout: 90 minutes
- Artifact paths: `logs_*/*.log`
- GitHub trigger: PR labels `blend` or `full`, or branch `dev`

Default workload from `scripts/run-blend-test.sh`:

```text
MODEL=openai/gpt-oss-20b
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.5
BENCHMARK_TIMEOUT_SEC=4800
SHUFFLE_NUM_DOCUMENTS=3
SHUFFLE_DOCUMENT_LENGTH=1000
SHUFFLE_OUTPUT_LEN=200
LMCACHE_MP_PORT=6566
SERVICE_PORT=10001
PREFILLER_PORT=8100
DECODER_PORT=8200
TELEMETRY_PORT=5768
```

The script starts:

1. LMCache blend server.
2. Prefiller vLLM with `LMCacheMPCBConnector`.
3. Decoder vLLM with `LMCacheMPConnector`.
4. CacheBlend proxy.
5. `benchmarks/multi_doc_qa/shuffle_doc_qa.py`.
6. A log scan for `error`, `traceback`, or `fatal`.

The success line is:

```text
[PASS] Blend integration test completed successfully.
```

## Gaps Found Before This Branch Update

The K3 CacheBlend script is the right source of truth for the preconfigured
image workflow, but it is not yet proven against this PR until a real K3 run or
equivalent controlled run succeeds.

Gaps found before the harness update in this branch:

- The script invokes:

```bash
python -m lmcache.v1.multiprocess.blend_server_v2
```

  That module is absent in this checkout. The current documented/code path is
  `lmcache server --engine-type blend` or
  `python -m lmcache.v1.multiprocess.server --engine-type blend`.

- The prefiller uses:

```json
{"kv_connector":"LMCacheMPCBConnector","kv_role":"kv_both"}
```

  In a reconstructed public vLLM environment this failed with
  `Unsupported connector type: LMCacheMPCBConnector`. That does not prove the
  private `tensormesh/cacheblend:latest` image is broken, but it does prove the
  generic public vLLM image is not an equivalent substitute without connector
  registration.

- The K3 Blend script wrote service and benchmark logs, but it did not scrape
  `/metrics`. This branch now adds a raw `lmcache_blend_*` Prometheus scrape
  under `logs_${BUILD_ID}/build_${BUILD_ID}_blend_metrics.log`.

## CacheBlend Metrics To Require

Internal OTel instrument names in code use dots:

```text
lmcache_blend.lookup_requests
lmcache_blend.lookup_requested_tokens
lmcache_blend.lookup_hit_tokens
lmcache_blend.lookup_fingerprint_hits
lmcache_blend.lookup_storage_hits
lmcache_blend.lookup_stale_chunks
lmcache_blend.lookup_no_gpu_context_errors
lmcache_blend.retrieve_requests
lmcache_blend.retrieve_chunks
lmcache_blend.retrieve_failures
lmcache_blend.store_pre_computed_requests
lmcache_blend.store_pre_computed_chunks
lmcache_blend.store_pre_computed_failures
lmcache_blend.store_final_requests
lmcache_blend.store_final_chunks
lmcache_blend.store_final_failures
lmcache_blend.fingerprints_registered
lmcache_blend.chunks_evicted
```

Prometheus output normalizes these to underscore names and counters should have
`_total` suffixes. The evidence scrape should include raw lines matching:

```bash
curl -fsS http://127.0.0.1:${PROMETHEUS_PORT:-9090}/metrics \
  | grep -E '^(# HELP |# TYPE |lmcache_blend_)'
```

Expected Prometheus metric prefixes include:

```text
lmcache_blend_lookup_requests_total
lmcache_blend_lookup_requested_tokens_total
lmcache_blend_lookup_hit_tokens_total
lmcache_blend_lookup_fingerprint_hits_total
lmcache_blend_lookup_storage_hits_total
lmcache_blend_lookup_stale_chunks_total
lmcache_blend_lookup_no_gpu_context_errors_total
lmcache_blend_retrieve_requests_total
lmcache_blend_retrieve_chunks_total
lmcache_blend_retrieve_failures_total
lmcache_blend_store_pre_computed_requests_total
lmcache_blend_store_pre_computed_chunks_total
lmcache_blend_store_pre_computed_failures_total
lmcache_blend_store_final_requests_total
lmcache_blend_store_final_chunks_total
lmcache_blend_store_final_failures_total
lmcache_blend_fingerprints_registered_total
lmcache_blend_chunks_evicted_total
```

## Acceptance Bar For PR #3255

A real CacheBlend evidence comment should include all of the following:

- Full PR link: <https://github.com/LMCache/LMCache/pull/3255>
- Exact branch, commit, image, GPU type/count, command, and env.
- Raw logs showing the LMCache blend server startup.
- Raw logs showing prefiller vLLM startup and successful readiness.
- Raw logs showing decoder vLLM startup and successful readiness.
- Raw proxy startup/request logs.
- Raw `shuffle_doc_qa.py` benchmark output.
- Raw final pass/fail line.
- Raw `lmcache_blend_*` Prometheus metric lines from `/metrics`.
- Explicit statement whether the run used the K3 `tensormesh/cacheblend:latest`
  image or a reconstructed equivalent.

Do not claim full CacheBlend verification from:

- Generic MP server startup alone.
- Decoder-only `LMCacheMPConnector` startup.
- Native extension build success.
- Unit tests for observability subscribers.
- A Modal CPU/GPU metrics smoke that does not exercise
  `LMCacheMPCBConnector`, the CacheBlend proxy, and `shuffle_doc_qa.py`.

## Next Implementation Target

The Blend K3 path in this branch has been updated to align with the current
documented server direction and collect metrics:

1. The default server module is now `lmcache.v1.multiprocess.server` with
   `--engine-type blend`; it can still be overridden with
   `BLEND_SERVER_MODULE` if the image requires a private module.
2. The server now uses current `--l1-size-gb` spelling and binds
   `--prometheus-port`.
3. The decoder uses explicit dynamic connector loading by default:
   `DECODER_KV_CONNECTOR_MODULE_PATH=lmcache.integration.vllm.lmcache_mp_connector`.
4. The prefiller connector remains `LMCacheMPCBConnector` by default, but
   `PREFILLER_KV_CONNECTOR` and `PREFILLER_KV_CONNECTOR_MODULE_PATH` are now
   env-configurable because this checkout does not contain an LMCache-shipped
   `LMCacheMPCBConnector` module path.
5. A best-effort metrics scrape is preserved under
   `logs_${BUILD_ID}/build_${BUILD_ID}_blend_metrics.log` before cleanup.
6. Post the raw logs and metrics to PR #3255 only after the workload reaches
   the benchmark and either passes or fails with complete evidence.

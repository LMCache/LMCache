# CacheBlend V2 end-to-end benchmark

This Buildkite job is the production-style CacheBlend V2 E2E path for PRs that touch the vLLM MP adapter, `BlendEngineV2`, or the CacheBlend precomputed/final KV flow.

It is intentionally stricter than the focused pytest/H100 verifier: pytest proves protocol and server behavior in isolation; this job proves live OpenAI-compatible workload traffic through vLLM, the CacheBlend proxy, and the LMCache blend server.

## Topology

```text
shuffle_doc_qa benchmark client
  -> CacheBlend proxy (:${SERVICE_PORT:-10001})
    -> prefiller vLLM (:${PREFILLER_PORT:-8100})
      -> LMCache blend server ZMQ (:${LMCACHE_MP_PORT:-6566})
    -> decoder vLLM (:${DECODER_PORT:-8200})
      -> same LMCache blend server ZMQ (:${LMCACHE_MP_PORT:-6566})
```

The LMCache server also exposes the documented HTTP frontend on `${LMCACHE_HTTP_PORT:-8080}` for health/status artifacts.

## Run from Buildkite

Upload the existing pipeline:

```bash
buildkite-agent pipeline upload .buildkite/k3_tests/blend/pipeline.yml
```

The job uses `tensormesh/cacheblend:latest` with two GPUs and runs:

```bash
bash .buildkite/k3_tests/blend/run.sh
```

## Run directly on a 2-GPU worker/container

```bash
cd /workspace/LMCache

export HF_TOKEN=<token>
export BUILDKITE_BUILD_ID=manual-cacheblend-pr3333
export MODEL=openai/gpt-oss-20b
export LMCACHE_SERVER_ENTRYPOINT=cli
export LMCACHE_L1_SIZE_GB=70
export LMCACHE_MP_PORT=6566
export LMCACHE_HTTP_PORT=8080
export SERVICE_PORT=10001
export PREFILLER_PORT=8100
export DECODER_PORT=8200
export TELEMETRY_PORT=5768
export TENSOR_PARALLEL=1
export MAX_MODEL_LEN=16384
export GPU_MEM_UTIL=0.5
export SHUFFLE_NUM_DOCUMENTS=3
export SHUFFLE_DOCUMENT_LENGTH=1000
export SHUFFLE_OUTPUT_LEN=200
export BENCHMARK_TIMEOUT_SEC=4800

bash .buildkite/k3_tests/blend/run.sh
```

## Run on Modal 2x H100

The checked-in Modal runner uses `gpu="H100:2"`, injects the Hugging Face token
with `modal.Secret.from_name("hf-token")`, runs the same `run.sh`, and copies
the evidence bundle to a Modal Volume named `lmcache-cacheblend-v2-e2e-artifacts`.

```bash
modal secret create hf-token HF_TOKEN=<token>

# Fast confidence pass.
modal run .buildkite/k3_tests/blend/modal_h100_e2e.py \
  --mode smoke \
  --commit <pr-head-sha>

# Maintainer-facing full pass.
modal run .buildkite/k3_tests/blend/modal_h100_e2e.py \
  --mode full \
  --commit <pr-head-sha>
```

Smoke mode uses:

```bash
MAX_MODEL_LEN=2048
LMCACHE_L1_SIZE_GB=20
SHUFFLE_NUM_DOCUMENTS=1
SHUFFLE_DOCUMENT_LENGTH=512
SHUFFLE_OUTPUT_LEN=64
BENCHMARK_TIMEOUT_SEC=2400
```

Full mode uses:

```bash
MAX_MODEL_LEN=16384
LMCACHE_L1_SIZE_GB=70
SHUFFLE_NUM_DOCUMENTS=3
SHUFFLE_DOCUMENT_LENGTH=1000
SHUFFLE_OUTPUT_LEN=200
BENCHMARK_TIMEOUT_SEC=4800
```

The default Modal image is `tensormesh/cacheblend:latest`, matching the
Buildkite job expectation that `/opt/venv` already contains the baseline vLLM
environment. If the image is mirrored or renamed, set `LMCACHE_MODAL_IMAGE` when
launching the Modal run.

If Modal cannot pull the private Buildkite image, use a public CUDA image and
bootstrap `/opt/venv` during the Modal image build. For an ungated public-model
smoke when no `hf-token` secret exists, disable the secret requirement and pass a
public model explicitly:

```bash
LMCACHE_MODAL_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu22.04 \
LMCACHE_MODAL_BOOTSTRAP_VLLM=1 \
LMCACHE_MODAL_REQUIRE_HF_SECRET=0 \
modal run .buildkite/k3_tests/blend/modal_h100_e2e.py \
  --mode smoke \
  --model facebook/opt-125m \
  --commit <pr-head-sha>
```

`run.sh` installs the current LMCache wheel in `/workspace/.venv`, then `scripts/run-blend-test.sh` launches:

- `lmcache server --engine-type blend` (preferred documented MP server path)
- prefiller vLLM with `LMCacheMPConnector`
- decoder vLLM with `LMCacheMPConnector`
- the local CacheBlend proxy
- `benchmarks/multi_doc_qa/shuffle_doc_qa.py`

Both vLLM processes pass an explicit CacheBlend MP connector config:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://localhost",
    "lmcache.mp.port": 6566,
    "lmcache.mp.cacheblend": true
  }
}
```

`kv_connector_module_path` forces vLLM to use the LMCache-shipped connector instead of a vendored connector. `lmcache.mp.cacheblend=true` is the MP CacheBlend V2 adapter switch; do not rely on the older in-process `LMCACHE_ENABLE_BLENDING` variable for this path.

## Legacy server fallback

The script defaults to the current documented server entrypoint:

```bash
lmcache server --engine-type blend
```

For compatibility debugging only, the legacy direct module entrypoint is still available:

```bash
export LMCACHE_SERVER_ENTRYPOINT=legacy
bash .buildkite/k3_tests/blend/run.sh
```

That launches `python -m lmcache.v1.multiprocess.blend_server_v2`. Do not use the legacy path as the preferred production recipe unless diagnosing a CLI regression.

## Required pass/fail gates

The E2E is considered production-actionable only if all of these are true:

- LMCache blend server starts and logs `LMCache cache blend v2 server is running...` or `BlendEngineV2` evidence.
- The orchestration log proves the documented CLI path: `lmcache server --engine-type blend`.
- Prefiller and decoder vLLM servers both become ready.
- Adapter logs show `enable_cacheblend=True`.
- Logs show CacheBlend V2 protocol activity, not only ordinary MP operations:
  - `CB_REGISTER_KV_CACHE` or `Registered CB KV cache`
  - `CB_STORE_PRE_COMPUTED` or `Stored pre-computed doc`
  - `CB_LOOKUP_PRE_COMPUTED_V2`
  - `CB_RETRIEVE_PRE_COMPUTED_V2` or `Retrieved pre-computed`
  - `CB_STORE_FINAL` or final-store evidence
- Logs show a non-empty CacheBlend match/hit, e.g. `Retrieved pre-computed for N match results` with `N > 0` or nonzero blend hit metrics.
- Proxy/telemetry logs show request-level prefiller save, proxy-to-decoder forwarding, and decoder completion.
- Benchmark exits `0` before `BENCHMARK_TIMEOUT_SEC`.
- Logs contain no traceback/fatal/runtime failure, HTTP 5xx, engine crash, CUDA/NCCL failure, or telemetry timeout.

`scripts/validate-blend-logs.sh` enforces these gates after the benchmark. A workload that starts the blend server but emits no CB lookup/retrieve/hit evidence must fail; that does not prove non-prefix CacheBlend reuse.

Run the GPU-free validator fixture suite before changing the validator rules:

```bash
bash .buildkite/k3_tests/blend/scripts/validate-blend-logs.sh --self-test
```

## Artifacts

Buildkite uploads:

- `logs_*/*.log`
- `logs_*/*.json`
- `logs_*/*.txt`

Important files:

- `build_<id>_blend.log` — orchestration log
- `build_<id>_blend_server.log` — LMCache server log and status output
- `build_<id>_prefiller_<port>.log` — prefiller vLLM log
- `build_<id>_decoder_<port>.log` — decoder vLLM log
- `build_<id>_proxy.log` — proxy/telemetry log
- `build_<id>_benchmark.log` — benchmark stdout/stderr
- `lmcache-status-final.json` — final HTTP status snapshot when available
- `versions.txt` — git SHA and package versions
- `nvidia-smi.txt` — GPU/runtime evidence

## Stronger benchmark follow-up

For maintainer-facing performance evidence, the official benchmark CLI can be run against the same `lmcache server --engine-type blend` + vLLM topology:

```bash
lmcache bench engine \
  --engine-url http://localhost:10001 \
  --workload long-doc-permutator \
  --lmcache-url http://localhost:8080 \
  --ldp-num-contexts 4 \
  --ldp-context-length 8000 \
  --ldp-num-permutations 24 \
  --ldp-num-inflight-requests 2 \
  --no-interactive \
  --json bench_summary.json
```

Use this for TTFT/throughput comparisons. The Buildkite smoke remains useful as a merge gate because it validates the exact CacheBlend V2 protocol path and artifacts in CI.

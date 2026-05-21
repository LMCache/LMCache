# Buildkite Web UI: Blend (CacheBlend)

**Purpose**: production-style CacheBlend V2 E2E proof for the disaggregated path:

```text
shuffle_doc_qa client
  -> CacheBlend proxy
    -> prefiller vLLM
      -> lmcache server --engine-type blend
    -> decoder vLLM
      -> same LMCache blend server
```

## Steps

Paste `buildkite-pipeline.yml` into the Buildkite pipeline Steps editor.

Set `HF_TOKEN` in the pasted env block or in **Pipeline Settings → Environment Variables**. The token is needed for gated model access. Optional: set `VLLM_WHEEL_URL` if testing a specific vLLM wheel.

The upload step runs `pipeline.yml` on the `k8s` queue. The real job uses:

- image: `tensormesh/cacheblend:latest`
- GPUs: `nvidia.com/gpu: "2"`
- command: `bash .buildkite/k3_tests/blend/run.sh`
- HF cache: host `/data/huggingface` mounted at `/root/.cache/huggingface`

Artifacts uploaded:

- `logs_*/*.log`
- `logs_*/*.json`
- `logs_*/*.txt`

## GitHub trigger

Use this filter:

```text
build.pull_request.labels includes "blend" || build.pull_request.labels includes "full" || build.branch == 'dev'
```

Recommended settings:

- Rebuild on PR label change: Yes
- Skip queued intermediate builds: Yes
- Cancel running intermediate builds: Yes

Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes). Changes under `.buildkite/` always run. Add the `force-ci` label to bypass.

## Strict validator rules

The job runs `scripts/validate-blend-logs.sh` after the benchmark. The validator fails unless logs prove real CacheBlend V2 traffic, not just ordinary MP traffic:

- documented server path: `lmcache server --engine-type blend`
- `BlendEngineV2` / blend-server startup evidence
- adapter startup with `enable_cacheblend=True`
- `CB_REGISTER_KV_CACHE`
- `CB_STORE_PRE_COMPUTED`
- `CB_LOOKUP_PRE_COMPUTED_V2`
- `CB_RETRIEVE_PRE_COMPUTED_V2`
- `CB_STORE_FINAL`
- non-empty CacheBlend hit/match evidence (`N > 0` or nonzero hit metrics)
- request-level prefiller save → proxy forwarding → decoder completion
- benchmark exit `0`
- no traceback, runtime error, CUDA/NCCL fatal, HTTP 5xx, engine death, or process-death evidence

Run the validator fixtures locally before changing rules:

```bash
bash .buildkite/k3_tests/blend/scripts/validate-blend-logs.sh --self-test
```

## Modal H100 local replication

The same E2E can be launched from a developer machine on Modal:

```bash
modal secret create hf-token HF_TOKEN=<token>

modal run .buildkite/k3_tests/blend/modal_h100_e2e.py \
  --mode smoke \
  --commit <pr-head-sha>

modal run .buildkite/k3_tests/blend/modal_h100_e2e.py \
  --mode full \
  --commit <pr-head-sha>
```

If Modal cannot pull the private Buildkite image, use a public CUDA image and bootstrap `/opt/venv` at Modal image-build time:

```bash
LMCACHE_MODAL_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu22.04 \
LMCACHE_MODAL_BOOTSTRAP_VLLM=1 \
modal run .buildkite/k3_tests/blend/modal_h100_e2e.py \
  --mode smoke \
  --commit <pr-head-sha>
```

Use CUDA 12.9.x for the public fallback image to match the current vLLM nightly PyTorch/cu129 wheels. The Modal bootstrap also installs CUDA 13 cudart runtime libs because current NIXL EP imports may require `libcudart.so.13`.

For an ungated public-model smoke when the `hf-token` Modal secret is unavailable:

```bash
LMCACHE_MODAL_IMAGE=nvidia/cuda:12.9.2-devel-ubuntu22.04 \
LMCACHE_MODAL_BOOTSTRAP_VLLM=1 \
LMCACHE_MODAL_REQUIRE_HF_SECRET=0 \
modal run .buildkite/k3_tests/blend/modal_h100_e2e.py \
  --mode smoke \
  --model facebook/opt-125m \
  --commit <pr-head-sha>
```

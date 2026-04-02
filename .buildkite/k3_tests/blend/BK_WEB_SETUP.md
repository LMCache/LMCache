# Buildkite Web UI: Blend (CacheBlend)

**Steps**: paste `buildkite-pipeline.yml` (set `HF_TOKEN`; optional `VLLM_WHEEL_URL`). It uploads `pipeline.yml` → 2×GPU job on `k8s`, image `tensormesh/cacheblend:latest`: `run.sh` → `setup-blend-env.sh` → `scripts/run-blend-test.sh`. HF cache: host `/data/huggingface` → `/root/.cache/huggingface`.

**GitHub trigger**: `build.pull_request.labels includes "blend" || build.pull_request.labels includes "full" || build.branch == 'dev'` — rebuild on label change: Yes; skip queued / cancel running: Yes.
 
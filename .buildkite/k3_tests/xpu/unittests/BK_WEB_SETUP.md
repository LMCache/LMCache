# Buildkite Web UI Setup: XPU Smoke Test

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "xpu" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

The scheduled nightly builds an XPU CI candidate from the current
`vllm/vllm-openai-xpu:nightly` digest. Its Dockerfile installs DPC++ using
`.buildkite/k3_tests/xpu/install_xpu_dpcpp_compiler.sh`, but does not install
LMCache. Both XPU pipelines install LMCache from their checked-out source on
each run. The candidate is pushed so both can test the same immutable image.
Only when UT and MP pass does the nightly record its digest on
`buildkite_latest_tested_vllm`; failures keep the previous pin.
The image keeps the upstream `vllm serve` entrypoint. `BASH_ENV` loads oneAPI
in noninteractive Bash jobs (including those launched by Buildkite); a
non-Bash command must source `/opt/intel/oneapi/setvars.sh` separately if it
needs the compiler environment.

The verifier triggers the existing `unit-tests-xpu` and `xpu-mp-test` Buildkite
pipelines. Configure `BUILDKITE_API_TOKEN` as a secret with `read_builds` and
`write_builds` scopes. The existing
`DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` must be able to push to the public
`lmcache/vllm-openai-xpu-ci` repository; XPU nodes must be able to pull it.
GitHub Actions must be able to update `buildkite_latest_tested_vllm`.

### Trigger strategy

The XPU pipeline is intentionally lightweight, so it is label/branch gated:

| Condition | Result |
|-----------|--------|
| PR label includes `full` | upload the XPU pipeline |
| branch is `dev` | upload the XPU pipeline |
| any docs/asset-only change | path filter skips upload |
| any change under `.buildkite/` | path filter forces upload |

The path filter treats the following as trivial for the k3 test harness:

- `*.md`, `LICENSE*`, `NOTICE*`
- `.gitignore`, `.gitattributes`, `.editorconfig`, `.mailmap`, `CODEOWNERS`
- anything under `docs/`, `asset/`, or `.github/`

If you need the XPU pipeline to run for a docs/asset-only PR, add the
`force-ci` label.


## Required host setup

Before creating the pipeline, prepare the machine that will run the `intel-xpu` queue:

1. Run [setup-cluster.sh](../../k3_harness/setup-cluster.sh) to install K3s, the GPU Operator, and the shared host volumes.
2. Run [install-agent-stack.sh](../../k3_harness/install-agent-stack.sh) with a Buildkite agent token and a GitHub token.


## Buildkite UI snippet

If you want to create the pipeline manually, paste this into the Steps editor:

```yaml
agents:
  queue: "intel-xpu"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/xpu/unittests/pipeline.yml
```

## What this pipeline does

- Runs the XPU smoke test on the `intel-xpu` queue
- Uses the latest Buildkite-verified XPU `image@sha256:...` from
  `tested_runtimes.jsonl`; until the first pin, UT uses
  `vllm/vllm-openai-xpu:v0.26.0` and MP retains its previous public ECR image.
- Installs LMCache from source via `setup-lmcache-only-env.sh`
- Verifies `torch.xpu.is_available()` inside the job pod

## TODO

- Refine the XPU path filter if additional XPU-only subtrees need to be excluded

- Refactor unit tests targeting multiple devices.

- Enable more `xpu` tests within current CI architecture.

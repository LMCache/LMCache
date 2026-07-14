# Buildkite Web UI Setup: XPU Smoke Test

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "xpu"`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This pipeline now has a single step: it runs the XPU smoke test directly in a
prebuilt public image and installs LMCache from source inside the job pod.

With this trigger filter, the pipeline only starts when the PR has the `xpu`
label. If the label is missing, the XPU pipeline is not triggered.

## Required host setup

Before creating the pipeline, prepare the machine that will run the `xpu` queue:

1. Run [setup-cluster.sh](../../k3_harness/setup-cluster.sh) to install K3s, the GPU Operator, and the shared host volumes.
2. Run [install-agent-stack.sh](../../k3_harness/install-agent-stack.sh) with a Buildkite agent token and a GitHub token.
3. Make sure the host can reach `public.ecr.aws` so the pod can pull the public XPU image.

## Queue and image notes

- The pipeline uses the `xpu` queue.
- The test pod runs `setup-lmcache-only-env.sh`, which installs LMCache from source on top of the public XPU base image.
- The test pod pulls `public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:ee0da84ab9e04ac7610e28580af62c365e898389-xpu` directly from ECR Public.
- The XPU test pod requests `deviceclass.resource.kubernetes.io/gpu.intel.com` through DRA.

## Buildkite UI snippet

If you want to create the pipeline manually, paste this into the Steps editor:

```yaml
agents:
  queue: "xpu"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/xpu/pipeline.yml
```

## What this pipeline does

- Runs the XPU smoke test on the `xpu` queue
- Uses the prebuilt public XPU image from ECR Public
- Installs LMCache from source via `setup-lmcache-only-env.sh`
- Verifies `torch.xpu.is_available()` inside the job pod

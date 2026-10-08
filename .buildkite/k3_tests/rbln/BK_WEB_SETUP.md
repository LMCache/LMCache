# Buildkite Web UI Setup: RBLN MP Hardware Smoke

This directory backs the `rbln-mp-test` pipeline on the `rbln-queue` queue.
It runs a focused LMCache multiprocess (MP) smoke on a real Rebellions NPU.
The shared K3 harness is NVIDIA-specific and is not reused here.

## Where the job runs

`rbln-queue` is served by [agent-stack-k8s](https://github.com/buildkite/agent-stack-k8s)
in the `buildkite` namespace of the LMCache CI cluster (managed by ArgoCD as
`lmcache.ci.buildkite`). Every job is a pod on the `npu2` node group:

```
Buildkite job ──> agent-stack-k8s ──> pod (buildkite ns, node-group=npu2)
                                       ├─ checkout container (git-ssh-credentials)
                                       └─ container-0: rbln-serve image (Harbor), uid 0
                                            └─ NPU: DRA claim from ResourceClaimTemplate npu1
```

- **NPU.** The `npu1` ResourceClaimTemplate allocates one device of
  DeviceClass `npu.rebellions.ai`. `rbln-container-toolkit` injects it as
  `/dev/rbln0` (plus `/dev/rblnfs0`, `/dev/rsd0`) and mounts the
  driver-matched runtime libraries (`librbln-thunk`, `librbln-ccl`, ...)
  under `/usr/local/lib/rbln`. Each job gets its own NPU, so no
  `RBLN_VISIBLE_DEVICES` binding or concurrency group is needed. The node has
  two NPUs and the controller runs at most two jobs (`max-in-flight: 2`).
- **Image.** `harbor.k8s.rebellions.in/rebellions-sw/rbln-serve:0.12.0rc0-ubuntu24.04-py3.12`
  comes from the public `rebellions-sw` Harbor project, so no pull secret is
  needed. Its `/opt/venv` ships a matched `torch-rbln` / `rebel-compiler`
  pair (0.12.0rc1, `torch` 2.13.0+cpu) plus gcc/g++ and ninja, so the job
  needs no RBLN Portal credentials. The image has no `curl` or `uv`; `run.sh`
  uses `pip` and a Python health probe. Harbor has no `latest` tag, so bump
  the pinned tag deliberately; the image label
  `ai.rebellions.rblness.versions` lists the bundled RBLN package versions.
- **User.** The image runs as uid 2000, which cannot write `/opt/venv`. The
  step sets `runAsUser: 0` so LMCache installs into the image's environment;
  the pod is discarded after the job.

## Pipeline settings

| Pipeline slug | Steps editor source | Uploaded definition |
|---------------|---------------------|---------------------|
| `rbln-mp-test` | `buildkite-pipeline.yml` | `.buildkite/k3_tests/rbln/pipeline.yml` |

Paste this into the `rbln-mp-test` pipeline's Steps editor:

```yaml
agents:
  queue: "rbln-queue"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/rbln/pipeline.yml
```

The pod needs network egress to PyPI for the LMCache build, common, and test
requirements.

## What `run.sh` does

1. Checks that `torch.rbln` sees an NPU and runs a tensor op on `rbln:0`.
2. Installs the LMCache build, common, and test requirements with `pip`; the
   image's `torch==<ver>+cpu` already satisfies the unpinned `torch`
   requirement.
3. Checks the RBLN runtime again, then builds LMCache with
   `BUILD_WITH_RBLN=1` (common native extensions only; RBLN has no compiled
   transfer kernel) and `--no-deps`, so the image's RBLN pair stays in place.
4. Runs `tests/v1/platform/devices/rbln`, including the real-NPU HND and MLA
   store/retrieve round trips in `test_rbln_device_transfer.py`.
5. Starts `lmcache server` with `LMCACHE_DEVICE_BACKEND=rbln`, waits for
   `/healthcheck`, and checks for a clean shutdown.

Runtime versions, `pip freeze`, pytest output, and the server log are
uploaded from `rbln-ci-artifacts/smoke/`.

Optional overrides (Buildkite build environment):

| Variable | Default | Purpose |
|----------|---------|---------|
| `RBLN_CI_PYTHON` | `python3` | Interpreter with torch-rbln installed |
| `LMCACHE_DEVICE_BACKEND` | unset | Explicit backend override; normal runs exercise auto-detection |
| `TEST_SELECTOR` | unset | Pass a pytest `-k` selector to the RBLN tests |
| `RBLN_CI_ZMQ_PORT` | `6555` | MP server ZMQ port |
| `RBLN_CI_HTTP_PORT` | `7555` | MP server HTTP port |

## GitHub trigger settings

- Filter: `build.pull_request.labels includes "rbln" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

| Condition | Result |
|-----------|--------|
| PR label includes `rbln` or `full` | upload the RBLN pipeline |
| branch is `dev` | upload the RBLN pipeline |
| docs/asset/example-only change | path filter skips upload |
| change only under another suite's `.buildkite/k3_tests/<suite>/` | path filter skips upload |
| change under `.buildkite/k3_tests/rbln/` or `common_scripts/` | path filter forces upload |

Add the `force-ci` label when a PR the filter skips still needs the RBLN lane.

Not covered yet: vLLM-RBLN model serving with the LMCache MP connector (the
`rbln-serve` image already ships vLLM-RBLN), multi-NPU tests (the `npu2`
template), and transfer-performance gates.

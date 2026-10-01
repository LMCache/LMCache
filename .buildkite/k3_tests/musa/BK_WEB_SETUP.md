# Buildkite Web UI Setup: MUSA Unit and Hardware Smoke

This directory backs two MUSA pipelines on the maintainer-provisioned
`MooreThreads` queue. `unit-tests-musa` runs the broad MUSA-compatible unit
suite, mirroring `unit-tests-xpu`, while `musa-mp-test` runs the focused
LMCache hardware/server smoke. The agent launches both in the pinned TorchMUSA
image; the shared K3 harness is NVIDIA-specific and is not reused here.

## Pipeline settings

Configure the two Buildkite pipelines as follows:

| Pipeline slug | Steps editor source | Uploaded definition |
|---------------|---------------------|---------------------|
| `unit-tests-musa` | `buildkite-unit-tests-pipeline.yml` | `.buildkite/k3_tests/musa/unit-pipeline.yml` |
| `musa-mp-test` | `buildkite-pipeline.yml` | `.buildkite/k3_tests/musa/pipeline.yml` |

Both upload steps and both dynamically uploaded test steps explicitly target
the existing `MooreThreads` queue. They share one global concurrency group, so
the unit and smoke jobs cannot contend for the same device. This repository
change reuses that queue; it does not create, rename, or replace it.

Both uploaded pipelines bind their job to the first MUSA device:

```text
MUSA_VISIBLE_DEVICES=0
```

No Buildkite UI environment variable is required. If the agent maps a
different device, change this value in both `pipeline.yml` and
`unit-pipeline.yml` to that explicit device index. Do not expose more than one
GPU to either lane.

Optional debugging overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LMCACHE_DEVICE_BACKEND` | unset | Explicit backend override for debugging; normal tests exercise auto-detection |
| `TEST_SELECTOR` | unset | Pass a pytest `-k` selector to the selected suite |
| `MUSA_CI_IMAGE` | pinned MUSA full-test image | Override the pre-provisioned TorchMUSA container image |
| `MUSA_CI_PYTHON` | `python3` | Override the Python executable inside the MUSA image |
| `MUSA_CI_ZMQ_PORT` | `6555` | Override the MP server ZMQ port |
| `MUSA_CI_HTTP_PORT` | `7555` | Override the MP server HTTP port |

## GitHub trigger settings

- Filter: `build.pull_request.labels includes "musa" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

### Trigger strategy

| Condition | Result |
|-----------|--------|
| PR label includes `musa` | upload the MUSA pipeline |
| PR label includes `full` | upload the MUSA pipeline |
| branch is `dev` | upload the MUSA pipeline |
| any docs/asset-only change | path filter skips upload |
| any change under `.buildkite/` | path filter forces upload |

The path filter treats Markdown, licenses, repository metadata, and files below
`docs/`, `asset/`, and `.github/` as trivial. Add the `force-ci` label when a
docs-only PR still needs the MUSA lane.

## Required agent setup

The self-hosted agent must provide:

1. A Linux MUSA host with a working driver and SDK.
2. Docker access and permission to use the MUSA devices from containers.
3. Access to the pinned
   `sh-harbor.mthreads.com/ai-kv/kuae-lmcache-vllm:20260819-kuae-ssd-e2e-full-tests-musa-aiter-ipc`
   image, or an equivalent image configured through `MUSA_CI_IMAGE`.
4. A compatible, pinned `torch`, `torch_musa`, `libmusart.so`, compiler, pip,
   and curl inside that image.
5. Access to the configured Python package index so the job can synchronize
   the current LMCache build and test requirements.
6. Device index `0` available to the pipeline's explicit
   `MUSA_VISIBLE_DEVICES=0` binding.

The wrapper mounts the current Buildkite checkout read-only and copies it to an
ephemeral container working directory, so tests exercise the PR rather than
the copy baked into the image without leaving root-owned build files on the
agent. The script directly uses the pre-provisioned Python environment and
does not create a venv. It installs the repository's current build, common,
and test requirements without requesting upgrades; the image's installed
TorchMUSA build satisfies the unpinned `torch` requirement. LMCache itself is
rebuilt from that working copy with
`BUILD_WITH_MUSA=1`, `BUILD_MOONCAKE=0`, and `--no-deps`, so installation
cannot replace the image's pinned TorchMUSA stack.

The Docker invocation uses the maintainer-validated `--ipc=host` and
`--network=host` flags. It explicitly sets `--entrypoint /bin/bash` because the
image already has `/bin/bash` as its entrypoint; appending another `bash`
causes the image to misinterpret it as a script. The wrapper uses `python3`,
which resolves to the verified `/usr/bin/python3` interpreter in this image.
The MooreThreads agent must retain the same host-side Docker/MUSA device
integration used to validate that command; an environment variable alone
cannot add device nodes to an otherwise unconfigured Docker daemon.

## Buildkite UI snippets

Paste this into the new `unit-tests-musa` pipeline's Steps editor:

```yaml
agents:
  queue: "MooreThreads"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/musa/unit-pipeline.yml
```

Keep this in the existing `musa-mp-test` pipeline's Steps editor:

```yaml
agents:
  queue: "MooreThreads"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/musa/pipeline.yml
```

The dynamically uploaded unit and smoke steps also target `MooreThreads` and
use a shared global concurrency limit of one, so two jobs cannot contend for
the same device even though their GitHub status contexts are independent.

## What these pipelines validate

- `torch_musa` imports and real MUSA hardware is visible.
- A small tensor allocation and matrix multiplication completes on `musa:0`.
- `libmusart.so` loads successfully.
- LMCache builds its common native extensions with the MUSA build profile.
- LMCache selects the `musa` device backend.
- The MUSA-compatible unit-test allowlist passes, excluding benchmark tests,
  CUDA/XPU/SGLang markers, and known optional Triton/NIXL modules.
- Focused MUSA connector, pin-memory, real-device block-transfer, and MP tests
  pass.
- The LMCache MP server reaches `/healthcheck` and terminates cleanly.
- Runtime versions, `pip freeze`, pytest output, early-failure diagnostics, and
  server logs are uploaded from `musa-ci-artifacts/`.

The initial PR gate does not run vLLM/SGLang model serving, GDS/MuFile,
MUSA memory/event IPC, multi-GPU tests, or native-transfer performance gates.
Add those as separate lanes only after this basic smoke test is stable.

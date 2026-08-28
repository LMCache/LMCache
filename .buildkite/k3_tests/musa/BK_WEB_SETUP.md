# Buildkite Web UI Setup: MUSA Unit and Hardware Smoke

This lane runs a serialized MUSA unit-test step followed by the focused LMCache
hardware/server smoke on the maintainer-provisioned `MooreThreads` queue. It
deliberately uses bare-metal execution: the shared K3 harness is
NVIDIA-specific and must not be reused until the MUSA queue has a supported
Kubernetes device plugin and runtime image.

## Pipeline settings

**Steps editor**: paste the contents of `buildkite-pipeline.yml`. Its upload
command must point to `.buildkite/k3_tests/musa/pipeline.yml`.

Both the upload step and the dynamically uploaded test steps explicitly target
the existing `MooreThreads` queue. This repository change reuses that queue; it
does not create, rename, or replace it.

Configure this environment variable on the pipeline or agent:

```text
MUSA_VISIBLE_DEVICES=0
```

The unit and smoke steps fail immediately when this value is absent. If the
agent maps a different device, use that explicit device index instead. Do not
expose more than one GPU to this lane.

Optional debugging overrides:

| Variable | Default | Purpose |
|----------|---------|---------|
| `TEST_SELECTOR` | unset | Pass a pytest `-k` selector to the focused suite |
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
2. A compatible, pinned `torch` and `torch_musa` installation in the agent's
   default Python environment.
3. `libmusart.so` on the dynamic linker search path.
4. Python with `venv`/`pip`, a C++ compiler, Ninja, and curl. `uv` is optional.
5. Access to the configured Python package index, or an equivalent internal
   dependency mirror.
6. An explicit `MUSA_VISIBLE_DEVICES` value exposing one device.

The test script creates a temporary virtual environment with
`--system-site-packages`, using `uv` when available and the standard
`python -m venv` plus `pip` otherwise. This preserves the agent's pinned
TorchMUSA stack. The script checks the MUSA runtime both before and after
dependency installation. LMCache is installed with `BUILD_WITH_MUSA=1`,
`BUILD_MOONCAKE=0`, and `--no-deps` so the editable install cannot replace that
stack with upstream PyTorch.

## Buildkite UI snippet

Paste this into the existing pipeline's Steps editor:

```yaml
agents:
  queue: "MooreThreads"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/musa/pipeline.yml
```

The dynamically uploaded unit and smoke steps also target `MooreThreads`, use a
shared global concurrency limit of one, and make the smoke depend on the unit
step, so two jobs cannot contend for the same device.

## What this pipeline validates

- `torch_musa` imports and real MUSA hardware is visible.
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

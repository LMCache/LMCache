# K3s CI Harness

Single-node K3s + NVIDIA GPU Operator for running LMCache CI on any GPU machine.

## Prerequisites

- Linux host with NVIDIA GPU(s) and driver installed
- Docker (for building the CI base image)
- Root access

## Setup (one-time)

```bash
# 1. Edit config.env with your Buildkite org and queue name
vim .buildkite/k3_harness/config.env

# 2. Install K3s + GPU Operator + build CI base image
.buildkite/k3_harness/setup-cluster.sh

# 3. Verify GPUs work in pods
.buildkite/k3_harness/smoke-test.sh

# 4. Connect to Buildkite
.buildkite/k3_harness/install-agent-stack.sh <BUILDKITE_AGENT_TOKEN>
```

`setup-cluster.sh` does everything: installs K3s, Helm, GPU Operator, builds the CI base image from `ci-base.Dockerfile`, imports it into K3s containerd locally, and creates host volume directories. Safe to re-run.

## Configuration

`config.env` has only the two values that change per-site:

| Variable | Default | What |
|----------|---------|------|
| `BUILDKITE_ORG` | `lmcache` | Buildkite organization slug |
| `BUILDKITE_QUEUE` | `k8s` | Queue name that pipelines target |

Everything else (image name, data paths, kubeconfig) uses sensible defaults.

## How Buildkite integration works (what needs to be done in the Buildkite Web UI)

This is different from the bare-metal agent setup where you install a Buildkite agent binary on your own machine, register it with a queue created on the Web UI, and manage it as a systemd service. With agent-stack-k8s, there are **no persistent agents on the machine**.

Instead, agent-stack-k8s runs a controller pod in K8s that polls Buildkite for jobs matching the configured queue tag. When a job appears, it creates a K8s pod to run it. When the job finishes, the pod is deleted.

**What you need from the Buildkite web UI**: just an agent token.

- Go to your org's **Agents** page → "Reveal Agent Token" (or create a new one)
- If your org uses [Clusters](https://buildkite.com/docs/clusters/overview), create a cluster + queue there and get the token from the cluster settings
- Pass the token to `install-agent-stack.sh`

You do **not** need to create a queue in the UI. The queue appears automatically when agent-stack-k8s connects with the queue tag from `config.env`. Pipeline steps target it with:

```yaml
agents:
  queue: "k8s"   # matches BUILDKITE_QUEUE in config.env
```

## Per-job environment

Every CI job sources `setup-env.sh` to install vLLM + LMCache:

```yaml
command: |
  source .buildkite/k3_harness/setup-env.sh
  bash .buildkite/scripts/my-test.sh
```

This installs:
1. **vLLM** from nightly wheel (~2 min)
2. **LMCache** from PR source (~30s)

Each pod has its own ephemeral filesystem that is cleanly erased after being shut down — no shared pip/uv cache, no cross-pod contention.

## Shared volumes

Mounted into every pod via `hostPath`:

| Host | Container | What |
|------|-----------|------|
| `/data/huggingface` | `/root/.cache/huggingface` | Model weights (read-heavy, write-once) |
| `/data/datasets` | `/root/correctness` | Test datasets (read-only after download) |

## GPU allocation

Request GPUs per pipeline step:

```yaml
plugins:
  - kubernetes:
      podSpec:
        containers:
          - resources:
              limits:
                nvidia.com/gpu: "2"  # 1 or 2
```

K8s device plugin handles atomic allocation. No `pick-free-gpu.sh`.

## CI base image

`ci-base.Dockerfile` builds an image with CUDA + Python 3.12 + uv + build deps (no vLLM/LMCache). `setup-cluster.sh` builds it automatically, auto-detects your GPU's compute capability for `TORCH_CUDA_ARCH_LIST`, and imports it into K3s containerd — no registry needed.

To rebuild after changing `requirements/*.txt`:
```bash
REBUILD_IMAGE=1 .buildkite/k3_harness/setup-cluster.sh
```

## Files

```
k3_harness/
├── config.env             # Buildkite org + queue (edit this)
├── ci-base.Dockerfile     # CI base image definition
├── setup-cluster.sh       # One-time: K3s + GPU Operator + base image
├── install-agent-stack.sh # One-time: Buildkite agent (needs token)
├── values.yaml            # Reference Helm values (documentation only)
├── setup-env.sh           # Per-job: install vLLM + LMCache
├── smoke-test.sh          # Verify: GPU pod runs in K3s
├── teardown.sh            # Remove everything (preserves /data/*)
└── README.md
```

## Teardown

```bash
.buildkite/k3_harness/teardown.sh
# Removes K3s, GPU Operator, agent-stack. Preserves /data/*
```

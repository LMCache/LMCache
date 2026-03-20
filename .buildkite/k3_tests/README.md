# K8s Test Pipelines

Each subdirectory under `k3_tests/` is a self-contained test with these files:

| File | Purpose |
|------|---------|
| `run.sh` | Test script (sources `k3_harness/setup-env.sh`, then runs tests) |
| `pipeline.yml` | K8s pod spec — GPU count, volumes, timeouts |
| `buildkite-pipeline.yml` | What to paste into the Buildkite UI "Steps" editor |
| `BK_WEB_SETUP.md` | Full Buildkite UI setup instructions: env vars, trigger filters, recommendations |

## Buildkite Web UI Setup

### Prerequisites

Before creating pipelines, make sure a queue named `k8s` exists in your Buildkite cluster. Go to **Organization Settings → Default cluster → Queues → New Queue** and create it. The queue needs no configuration and no agents — agent-stack-k8s creates ephemeral pod-based agents automatically when jobs arrive.

### Per-pipeline setup

For each test directory, create a pipeline in the Buildkite UI.
Each directory has a `BK_WEB_SETUP.md` with the exact settings — env vars, GitHub trigger filters, and recommendations for that test. The short version:

1. Go to your org → **New Pipeline**
2. In the **Steps** editor, paste the contents of that test's `buildkite-pipeline.yml`:
   ```yaml
   agents:
     queue: "k8s"

   env:
     HF_TOKEN: "<your HuggingFace token>"

   steps:
     - label: ":pipeline: Upload pipeline"
       command: buildkite-agent pipeline upload .buildkite/k3_tests/<test-name>/pipeline.yml
   ```
   The `agents.queue` must match the queue you created above. This routes the initial upload step to agent-stack-k8s, which checks out the repo and uploads the real pipeline definition. Each subsequent step in `pipeline.yml` also targets the same queue.
3. `HF_TOKEN` is needed for gated model access (e.g., Llama, Qwen). Set it in the `env` block as shown above, or under **Pipeline Settings → Environment Variables** in the UI — both work
4. Under **GitHub Settings**, configure trigger filters per the test's `BK_WEB_SETUP.md`
5. Save — jobs will run on the K8s queue automatically

### Trigger strategy

Not all tests should run on every push. The general pattern:

| Test weight | When to trigger | Example filter condition |
|-------------|----------------|------------------------|
| Lightweight (1 GPU, <20 min) | Every push / every PR | *(no filter)* |
| Heavy (multi-GPU, >30 min) | PR label or main branch only | `build.pull_request.labels includes "full" \|\| build.branch == 'dev'` |

Set **"Rebuild on PR label change"** to `Yes` for label-triggered pipelines so adding a label to an existing PR kicks off the build.

## Adding a New Test

1. Create a directory: `.buildkite/k3_tests/<test-name>/`

2. Write a `run.sh` that sources the shared environment setup:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   cd "$(cd "$(dirname "$0")/../../.." && pwd)"
   source .buildkite/k3_harness/setup-env.sh
   # ... your test commands ...
   ```

3. Write a `pipeline.yml`. Set the GPU limit to what your test needs:
   ```yaml
   steps:
     - label: ":test_tube: My Test"
       command: .buildkite/k3_tests/<test-name>/run.sh
       timeout_in_minutes: 30
       agents: { queue: "k8s" }
       plugins:
         - kubernetes:
             podSpec:
               containers:
                 - name: container-0
                   image: lmcache/ci-base:latest
                   imagePullPolicy: Never  # local image, imported into K3s containerd
                   resources:
                     limits:
                       nvidia.com/gpu: "1"
                   volumeMounts:
                     - { name: hf-cache, mountPath: /root/.cache/huggingface }
               volumes:
                 - { name: hf-cache, hostPath: { path: /data/huggingface, type: DirectoryOrCreate } }
   ```

4. Write a `buildkite-pipeline.yml` (the upload step for the Buildkite UI):
   ```yaml
   agents:
     queue: "k8s"

   steps:
     - label: ":pipeline: Upload pipeline"
       command: buildkite-agent pipeline upload .buildkite/k3_tests/<test-name>/pipeline.yml
   ```

5. Write a `BK_WEB_SETUP.md` documenting the Buildkite UI settings for this test (env vars, trigger filters, recommendations). Use an existing test's `BK_WEB_SETUP.md` as a template.

6. `chmod +x` your `run.sh` and create the pipeline in the Buildkite UI.

### Optional: datasets volume

If your test needs pre-downloaded data (e.g., ShareGPT), add the datasets volume:
```yaml
volumeMounts:
  - { name: datasets, mountPath: /root/correctness }
volumes:
  - { name: datasets, hostPath: { path: /data/datasets, type: DirectoryOrCreate } }
```

### Optional: Docker-in-Docker

If your test runs Docker containers inside the pod:
```yaml
securityContext:
  privileged: true
volumeMounts:
  - { name: docker-sock, mountPath: /var/run/docker.sock }
volumes:
  - { name: docker-sock, hostPath: { path: /var/run/docker.sock } }
```

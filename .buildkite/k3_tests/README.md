# K8s Test Pipelines

Each subdirectory under `k3_tests/` is a self-contained test with these files:

| File | Purpose |
|------|---------|
| `run.sh` | Test script (sources `k3_harness/setup-env.sh`, then runs tests) |
| `pipeline.yml` | K8s pod spec — GPU count, volumes, timeouts |
| `buildkite-pipeline.yml` | What to paste into the Buildkite UI "Steps" editor (runs path filter and PR check gate, then uploads `pipeline.yml`) |
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
       command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/<test-name>/pipeline.yml
   ```
   The `agents.queue` must match the queue you created above. This routes the upload
   step to agent-stack-k8s, which checks out the repo and runs the path filter.
   Builds that are not skipped upload the real `pipeline.yml` after passing the
   PR check gate (for PR builds). Each subsequent step targets the same queue.
3. `HF_TOKEN` is needed for gated model access (e.g., Llama, Qwen). Set it in the `env` block as shown above, or under **Pipeline Settings → Environment Variables** in the UI — both work
4. Under **GitHub Settings**, configure trigger filters per the test's `BK_WEB_SETUP.md`
5. Save — jobs will run on the K8s queue automatically

### Path-based skip (auto-pass on docs-only changes)

The upload step in `buildkite-pipeline.yml` runs `common_scripts/upload-pipeline.sh`
instead of `buildkite-agent pipeline upload` directly. The wrapper
(`common_scripts/path-filter.sh`) inspects the changed files in the build and:

- **Skips** the build (exits 0 without uploading `pipeline.yml` → build is
  green with just the upload step) when the changed files are not relevant to
  the uploaded test pipeline. This includes docs-only / trivial changes such
  as `*.md`, `LICENSE*`, `NOTICE*`, `.gitignore`, `.gitattributes`,
  `.editorconfig`, `.mailmap`, `CODEOWNERS`, or anything under `docs/` or
  `asset/`, or `.github/`. (`.github/` is trivial here because k3 tests run on Buildkite,
  not GitHub Actions, so workflow / CODEOWNERS / template changes do not
  affect them.)

The filter intentionally uses a conservative blacklist instead of a whitelist.
In other words, only files that are explicitly known to be irrelevant are
skipped. Anything else is treated as potentially functional and therefore keeps
that pipeline enabled.

Known-irrelevant patterns include docs-only changes (`*.md`, `docs/**`,
`asset/**`, `.github/**`, `LICENSE*`, `NOTICE*`, `.gitignore`,
`.gitattributes`, `.editorconfig`, `.mailmap`, `CODEOWNERS`) and the K3 docs
pages under `.buildkite/k3_tests/`.

This policy is safer than a whitelist because it avoids false negatives: when a
change may affect behavior, even if it is not 100% confirmed, the job still runs.


Detection:
- PR builds diff against `origin/${BUILDKITE_PULL_REQUEST_BASE_BRANCH}`
  (default `main`) using the merge-base.
- Push builds diff `HEAD~1..HEAD`.
- Scheduled builds (`BUILDKITE_SOURCE=schedule`) are never skipped.
- If the script can't determine the changed files (shallow clone with no
  parent, missing base branch, etc.) it falls back to "do not skip".

To bypass the skip and force a full run, add the **`force-ci`** label to the
PR on GitHub. Buildkite picks up PR labels automatically; when the filter
sees `force-ci` it runs the full pipeline regardless of which files changed.


### PR check gate (Code Quality and DCO)

For PR builds using `common_scripts/upload-pipeline.sh`, label and path skips
run first. If the build is not skipped, `common_scripts/wait-for-required-checks.sh`
waits for Code Quality and DCO on the build's original PR head commit before
uploading the test pipeline. Non-PR builds do not wait for these checks.
The `force-ci` label bypasses skip rules but still requires the check gate.

- `Check code quality` must report `success` or `neutral`; `skipped` is accepted
  only when `Detect changes` succeeded in the same Check Suite.
- `DCO` must report `success`.
- Missing or running checks wait. Failed checks, persistent API errors,
  timeouts, a closed PR, or a changed PR head stop the upload.

The upload agent needs Bash, curl, jq and Git. Provide `GITHUB_TOKEN` or
`GH_TOKEN` in the **initial upload step** with read access to checks, commit
statuses and pull requests in the base repository. The gate script is read
from the PR's target branch in that repository and must exist there.

The default wait is 30 minutes (`REQUIRED_CHECKS_TIMEOUT_SECONDS=1800`). Polling
starts at 10 seconds (`REQUIRED_CHECKS_POLL_SECONDS=10`) and backs off to 60
seconds. Set these environment variables on the upload step to override them.

After fixing and rerunning a failed fast check, retry the Buildkite upload step
or build for the current PR head. A completed gate does not restart automatically
when a check is rerun; builds for an older head must be replaced with a new build.

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

4. Write a `buildkite-pipeline.yml` (the snippet pasted into the Buildkite UI's Steps
   editor). Use `common_scripts/upload-pipeline.sh` for path-based skip and the
   PR check gate:
   ```yaml
   agents:
     queue: "k8s"

   steps:
     - label: ":pipeline: Upload pipeline"
       command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/<test-name>/pipeline.yml
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

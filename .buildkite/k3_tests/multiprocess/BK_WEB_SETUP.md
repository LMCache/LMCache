# Buildkite Web UI Setup: Multiprocess Tests

## Create Pipeline

1. Go to **Pipelines → New Pipeline**
2. Name: `K3s Multiprocess Tests` (or similar)
3. Repository: `git@github.com:LMCache/LMCache.git`
4. In the **Steps** editor, paste:

```yaml
agents:
  queue: "k8s"

steps:
  - label: ":pipeline: Upload pipeline"
    command: buildkite-agent pipeline upload .buildkite/k3_tests/multiprocess/pipeline.yml
```

## Environment Variables

Add these under **Pipeline Settings → Environment Variables**:

| Variable | Value | Why |
|----------|-------|-----|
| `HF_TOKEN` | *(your HuggingFace token)* | Gated model access (Qwen3-14B) |

## GitHub Trigger Settings

Under **Pipeline Settings → GitHub**:

| Setting | Value | Why |
|---------|-------|-----|
| Build pull requests | Yes | |
| Build branches | Yes | Runs on pushes to main/dev |
| Build pull request forks | Yes | Community contributors |
| Rebuild on PR label change | Yes | Trigger on "mp" / "full" label |
| Filter condition | `build.pull_request.labels includes "mp" \|\| build.pull_request.labels includes "full" \|\| build.branch == 'dev'` | Only on label or dev branch |
| Publish commit status | Yes | Show status check on GitHub |
| Skip queued branch builds | Yes | Only latest push matters |
| Cancel running branch builds | Yes | Save GPU time |

## Recommendations

- **Trigger**: PR labels `mp` or `full`, or push to `dev`/`main`. This test needs 2 GPUs, runs Docker-in-Docker, and takes ~45 minutes — too heavy for every push.
- **Timeout**: 60 minutes in `pipeline.yml` (Docker image pulls can be slow on first run).
- **Privileged pod**: This test requires `securityContext.privileged: true` because it runs Docker containers inside the K8s pod. This is already configured in `pipeline.yml`.

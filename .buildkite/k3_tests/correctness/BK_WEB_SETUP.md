# Buildkite Web UI Setup: Correctness Tests

## Create Pipeline

1. Go to **Pipelines → New Pipeline**
2. Name: `K3s Correctness Tests` (or similar)
3. Repository: `git@github.com:LMCache/LMCache.git`
4. In the **Steps** editor, paste:

```yaml
agents:
  queue: "k8s"

steps:
  - label: ":pipeline: Upload pipeline"
    command: buildkite-agent pipeline upload .buildkite/k3_tests/correctness/pipeline.yml
```

## Environment Variables

Add these under **Pipeline Settings → Environment Variables**:

| Variable | Value | Why |
|----------|-------|-----|
| `HF_TOKEN` | *(your HuggingFace token)* | Gated model access (Qwen2.5-14B tokenizer) |

> `BUILD_ID` is set automatically from `BUILDKITE_BUILD_ID`.

## GitHub Trigger Settings

Under **Pipeline Settings → GitHub**:

| Setting | Value | Why |
|---------|-------|-----|
| Build pull requests | Yes | |
| Build branches | Yes | |
| Build pull request forks | Yes | Community contributors |
| Filter condition | *(none — leave empty)* | Run on every push/PR |
| Publish commit status | Yes | Required status check |
| Skip queued branch builds | Yes | Only latest push matters |
| Cancel running branch builds | Yes | Save GPU time |

## Recommendations

- **Trigger**: Every push and every PR. This is a single-GPU, ~15-minute test — lightweight enough to be a required status check.
- **Required check**: Consider making this a required GitHub status check for merging PRs.

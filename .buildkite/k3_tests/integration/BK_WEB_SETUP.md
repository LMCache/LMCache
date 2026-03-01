# Buildkite Web UI Setup: Integration Tests

## Create Pipeline

1. Go to **Pipelines → New Pipeline**
2. Name: `K3s Integration Tests` (or similar)
3. Repository: `git@github.com:LMCache/LMCache.git`
4. In the **Steps** editor, paste:

```yaml
steps:
  - label: ":pipeline: Upload pipeline"
    command: buildkite-agent pipeline upload .buildkite/k3_tests/integration/pipeline.yml
```

## Environment Variables

Add these under **Pipeline Settings → Environment Variables**:

| Variable | Value | Why |
|----------|-------|-----|
| `HF_TOKEN` | *(your HuggingFace token)* | Gated model access |

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

- **Trigger**: Every push and every PR. Single-GPU, ~20-minute test — lightweight enough to run on all PRs.
- **Required check**: Consider making this a required GitHub status check alongside correctness.

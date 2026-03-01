# Buildkite Web UI Setup: Comprehensive Tests

## Create Pipeline

1. Go to **Pipelines → New Pipeline**
2. Name: `K3s Comprehensive Tests` (or similar)
3. Repository: `git@github.com:LMCache/LMCache.git`
4. In the **Steps** editor, paste:

```yaml
steps:
  - label: ":pipeline: Upload pipeline"
    command: buildkite-agent pipeline upload .buildkite/k3_tests/comprehensive/pipeline.yml
```

## Environment Variables

Add these under **Pipeline Settings → Environment Variables**:

| Variable | Value | Why |
|----------|-------|-----|
| `HF_TOKEN` | *(your HuggingFace token)* | Gated Llama model access (tokenizer download) |

> `BUILD_ID` and `CUDA_VISIBLE_DEVICES` are set automatically by Buildkite and the K8s device plugin.

## GitHub Trigger Settings

Under **Pipeline Settings → GitHub**:

| Setting | Value | Why |
|---------|-------|-----|
| Build pull requests | Yes | |
| Build branches | Yes | Runs on pushes to main/dev |
| Build pull request forks | Yes | Community contributors |
| Rebuild on PR label change | Yes | Trigger on "full" label |
| Filter condition | `build.pull_request.labels includes "full" \|\| build.branch == 'dev'` | Only run on "full" label or dev branch |
| Publish commit status | Yes | Show status check on GitHub |
| Skip queued branch builds | Yes | Only latest push matters |
| Cancel running branch builds | Yes | Save GPU time on force-pushes |

## Recommendations

- **Trigger**: PR label `full` or push to `dev`/`main`. This test runs 10 parallel GPU steps (7x 1-GPU + 3x 2-GPU) — too heavy for every push.
- **Timeout**: Individual steps have 30-minute timeouts in `pipeline.yml`.
- **No schedule needed**: The label-based trigger is sufficient. Add a nightly schedule only if you want regression testing on `dev` even without PRs.

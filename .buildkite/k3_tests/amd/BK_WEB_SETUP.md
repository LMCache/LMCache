# Buildkite Web UI Setup: AMD Unit Tests

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "amd" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This pipeline runs the bare-metal ROCm unit-test flow while making PR admission
opt-in so AMD capacity is only consumed when requested.

### Trigger strategy

| Condition | Result |
|-----------|--------|
| PR label includes `amd` | upload the AMD pipeline |
| PR label includes `full` | upload the AMD pipeline |
| branch is `dev` | upload the AMD pipeline |
| any docs/asset-only change | path filter skips upload |
| any change under `.buildkite/` | path filter forces upload |

The path filter treats the following as trivial for the AMD lane:

- `*.md`, `LICENSE*`, `NOTICE*`
- `.gitignore`, `.gitattributes`, `.editorconfig`, `.mailmap`, `CODEOWNERS`
- anything under `docs/`, `asset/`, or `.github/`

If you need the AMD pipeline to run for a docs/asset-only PR, add the
`force-ci` label.

## Buildkite UI snippet

If you want to create or update the pipeline manually, paste this into the
Steps editor:

```yaml
steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/amd/pipeline.yml
```

Keep the existing AMD runner / queue assignment in the Buildkite pipeline
configuration; this repository change does not alter host provisioning.

## What this pipeline does

- Runs the AMD bare-metal unit-test workflow in `.buildkite/k3_tests/amd/pipeline.yml`
- Admits PRs carrying the `amd` or `full` label
- Reuses the shared path filter so docs-only PRs skip the AMD lane cleanly

# Buildkite Web UI Setup: MooreThreads Smoke Test

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "mthread" || build.pull_request.base_branch == "mbl/mthread-ci-test-dev"`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This pipeline is intentionally minimal. It validates that the dedicated
`MooreThreads` queue, upload wrapper, and repo-hosted pipeline wiring are all
working by running a single hello step.

### Trigger strategy

| Condition | Result |
|-----------|--------|
| PR label includes `mthread` | upload the MooreThreads pipeline |
| PR base branch is `mbl/mthread-ci-test-dev` | upload the MooreThreads pipeline |
| any docs/asset-only change | path filter skips upload |
| any change under `.buildkite/` | path filter forces upload |

The path filter treats the following as trivial for this lane:

- `*.md`, `LICENSE*`, `NOTICE*`
- `.gitignore`, `.gitattributes`, `.editorconfig`, `.mailmap`, `CODEOWNERS`
- anything under `docs/`, `asset/`, or `.github/`

If you need the pipeline to run for a docs/asset-only PR, add the `force-ci`
label.

## Buildkite UI snippet

If you want to create or update the pipeline manually, paste this into the
Steps editor:

```yaml
agents:
  queue: "MooreThreads"

steps:
- label: ":pipeline: Upload pipeline"
  command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/mthread/pipeline.yml
```

Keep the existing MooreThreads runner / queue assignment in the Buildkite
pipeline configuration; this repository change only adds the repo-side files.

## What this pipeline does

- Routes the upload step through the `MooreThreads` queue
- Reuses the shared path filter in `common_scripts/upload-pipeline.sh`
- Uploads `.buildkite/k3_tests/mthread/pipeline.yml`
- Runs a single hello smoke test on the MooreThreads agent

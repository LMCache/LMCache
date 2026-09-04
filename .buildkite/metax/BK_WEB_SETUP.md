# Buildkite Web UI Setup: MetaX MACA Smoke + Unit Tests

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "metax" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This is the same trigger convention already used by the other single-agent
bare-metal lanes (AMD, XPU, MUSA) and the sglang lane. "Skip queued / cancel
running branch builds" matters more here than on the K8s-backed lanes: this
queue has exactly one dedicated agent and one GPU, so a superseded build left
running just delays feedback on the newest commit without adding any value --
there's no second job it could run in parallel with.

### Trigger strategy

| Condition | Result |
|-----------|--------|
| PR label includes `metax` | upload the MetaX pipeline |
| PR label includes `full` | upload the MetaX pipeline |
| branch is `dev` | upload the MetaX pipeline |
| any docs/asset-only change | path filter skips upload |
| any change under `.buildkite/` | path filter forces upload |

The path filter treats the following as trivial (see
`.buildkite/k3_tests/common_scripts/path-filter.sh`):

- `*.md`, `LICENSE*`, `NOTICE*`
- `.gitignore`, `.gitattributes`, `.editorconfig`, `.mailmap`, `CODEOWNERS`
- anything under `docs/`, `asset/`, or `.github/`

If you need the MetaX pipeline to run for a docs/asset-only PR, add the
`force-ci` label.

Label-gating (rather than running on every PR unconditionally, as the
sglang lane does) matches the majority convention here because this queue's
capacity is a single shared GPU/host, same constraint as AMD/XPU/MUSA.
Starting conservative (opt-in via label, or automatic on `dev`) is the
safer default; revisit as this queue's real usage patterns become clear.

## Required host setup

The `metax-maca` queue's agent must additionally have passwordless sudo for
removing its own workspace directory (see "What this pipeline does" below,
and the end-of-run cleanup in `run-unit-tests.sh` / `run-smoke-tests.sh`),
matching what the main CUDA `pipeline.yml` step already requires on its own
bare-metal queue.

## Buildkite UI snippet

```yaml
steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/metax/pipeline.yml
```

Keep the existing MetaX runner / queue selection ("metax-maca") in the
Buildkite pipeline configuration.

## What this pipeline does

- PR builds run `run-smoke-tests.sh`: a fast, small subset of `tests/v1/`
  covering core compute/platform/native-extension paths.
- Builds on `dev` (post-merge) run `run-unit-tests.sh`: pytest's default
  collection (not scoped to `tests/v1/` -- covers the whole `tests/` tree
  except what's explicitly ignored/deselected), matching the scope of the
  main CUDA "Unit Tests" step.
- Both scripts delete the entire workspace (`sudo rm -rf`) after the test run
  completes, matching the main `pipeline.yml`'s own convention, so the next
  build starts from a fresh checkout rather than accumulating build
  artifacts on this queue's limited (tens-of-GB) disk. A defensive cleanup
  of the venv/coverage/cache paths also runs at the *start* of each script,
  in case a previous build was cancelled (see "Skip queued / cancel running
  branch builds" above) before reaching its own end-of-run cleanup.

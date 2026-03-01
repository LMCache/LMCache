# Buildkite Web UI Setup: Integration Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: *(none — runs on every push/PR)*
- Skip queued / cancel running branch builds: Yes

Lightweight (1 GPU) — good candidate for a required GitHub status check.

## Nightly Scheduled Build

Create a **Scheduled Build** on this pipeline for regular health checks:

- **Schedule**: daily (e.g. `0 1 * * *` — 1am UTC)
- **Branch**: `dev`

No extra env vars needed — integration tests don't produce baselines.

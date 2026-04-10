# Buildkite Web UI Setup: Integration Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: *(none — runs on every push/PR)*
- Skip queued / cancel running branch builds: Yes

Lightweight (1 GPU) — good candidate for a required GitHub status check.

> **One-time UI update required**: re-paste `buildkite-pipeline.yml` into the
> Steps editor after merging. Builds whose only changes are
> docs/`*.md`/`LICENSE`/`.github/**` then auto-pass via the
> [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Changes under `.buildkite/` always run. Set `K3_PATH_FILTER_DISABLE=1` to
> bypass.
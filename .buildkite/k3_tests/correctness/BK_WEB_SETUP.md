# Buildkite Web UI Setup: Correctness Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: *(none — runs on every push/PR)*
- Skip queued / cancel running branch builds: Yes

Lightweight (1 GPU) — good candidate for a required GitHub status check.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes)
> baked into `pipeline.yml`. Changes under `.buildkite/` always run. Set
> `K3_PATH_FILTER_DISABLE=1` to bypass. The filter lives in the repo, not
> in the Buildkite UI, so editing it never requires re-pasting anything.

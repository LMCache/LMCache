# Buildkite Web UI Setup: Integration Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: *(none — runs on every push/PR)*
- Skip queued / cancel running branch builds: Yes

This pipeline reports the required `buildkite/k3-integration-test` status.
Keep its initial upload job enabled even for tests-only PRs.

The upload job uses the [path filter](../README.md#required-checks-and-tests-only-changes)
to skip test steps for changes confined to `tests/` and trivial files such as
docs/`*.md`/`LICENSE`/`.github/**`. It exits successfully so Buildkite reports a
passing required status. Relevant runtime and CI changes still run the tests.
Add `force-ci` to bypass the path filter.

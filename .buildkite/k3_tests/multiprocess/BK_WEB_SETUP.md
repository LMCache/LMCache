# Buildkite Web UI Setup: Multiprocess Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "mp" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This pipeline reports the required `buildkite/k3-multiprocess-test` status.
Keep adding `full` when auto-merge is enabled, including on tests-only PRs;
alternatively, add `mp` to start it earlier. Suppressing the build entirely
leaves the required status missing and blocks merging.

The initial upload job uses the [path filter](../README.md#required-checks-and-tests-only-changes)
to skip test steps for changes confined to `tests/` and trivial files such as
docs/`*.md`/`LICENSE`/`.github/**`. It exits successfully so Buildkite reports a
passing required status without starting the GPU tests. Relevant runtime and
CI changes still run the tests. Add `force-ci` alongside `mp` or `full` to
bypass the path filter.

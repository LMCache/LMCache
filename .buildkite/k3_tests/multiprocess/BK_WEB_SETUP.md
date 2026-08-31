# Buildkite Web UI Setup: Multiprocess Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "mp" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

Heavy test (2 GPUs, Docker-in-Docker, ~45 min) — run on `"mp"`/`"full"` label or dev push, not every PR.

**`dsv4_flash_tp` (DeepSeek-V4-Flash, 4 GPUs, 40+ min)** is the only 4-GPU step
here, so any build that includes it holds a whole 4-GPU node and pushes every
other build's 1- and 2-GPU steps behind it. It is off by default and runs when
either of these holds:

- the PR carries the **`dsv4`** label, alongside the `mp`/`full` label that
  gates this pipeline — for a change to the hybrid-KV-group / slot-compression
  path
- the build was started with **`RUN_DSV4_TEST=true`** in "New Build" env

The `dsv4` label does not exist in the repo yet — it has to be created once
(Issues → Labels) before the label path can be used, and applying it needs
triage permission on LMCache/LMCache, which a PR author working from a fork
does not have. The `RUN_DSV4_TEST=true` path needs neither.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Changes under `.buildkite/` always run. Add `force-ci` label to the PR to
> bypass.

## No periodic run

`dsv4_flash_tp` has no schedule of its own — it runs when a PR opts in or when
someone starts a build with `RUN_DSV4_TEST=true`. A 4-GPU node for 40+ minutes
is too expensive to spend on a timer nobody is watching, so a regression in
DeepSeek-V4-Flash support surfaces at the next opt-in build rather than the
next morning.

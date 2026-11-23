# Buildkite Web UI Setup: Unit Tests

**Steps editor**: the upload step for `.buildkite/k3_tests/unit/pipeline.yml`.

**GitHub settings**:
- Filter builds using a conditional: `build.pull_request.id != null && build.pull_request.draft != true`
- Build when pull request is opened or updated: Yes
- Build when pull request becomes ready for review: Yes
- Skip when pull request has existing build for commit and branch: **No**

**Builds settings**: Cancel Intermediate Builds: Yes, so a new push cancels the
running build of the previous commit.

`buildkite/k3-unit-tests` is a required check on `dev`. It builds every commit
of a PR that is ready for review, in parallel with the GitHub Actions checks,
and never builds draft commits. Clicking "Ready for review" builds the current
commit. Converting the PR back to draft cancels its running build
(`.github/workflows/pr_draft_cancel_buildkite.yml`); that is why "Skip ...
existing build" must be off, or the next ready-for-review on the same commit
would leave the required check canceled.

Why ready-for-review commits, in parallel (PR CI baseline, 2026-10-17 to
2026-11-15, 805 GPU-h of unit tests):
- 46% of those GPU-hours (372 GPU-h) ran on draft commits; the filter above
  removes them without delaying any ready PR.
- Starting only after Code Quality passed would save another ~25 GPU-h but
  delay the GPU start by 14 min (p50) / 160 min (p90), the GitHub runner queue.
- Starting only after test.yml and cpu_device.yml passed would save ~80 GPU-h
  in total beyond the filter but delay the check by 72 min (p50) / 298 min
  (p90).

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Add `force-ci` label to the PR to bypass.

# Buildkite Web UI Setup: Unit Tests

**Steps editor**: unchanged — the upload step for `.buildkite/k3_tests/unit/pipeline.yml`.

**GitHub settings**:
- Filter builds using a conditional: `build.pull_request.draft != true`
- Build when pull request is opened or updated: Yes
- Build when pull request becomes ready for review: Yes
- Skip when pull request has existing build for commit and branch: **No**
- Skip queued / cancel running branch builds: Yes

`buildkite/k3-unit-tests` is a required check on `dev`. Each non-draft PR
commit gets one build, and its GPU step runs only after the GitHub Layer 2
checks pass:

```
PR ready for review / new commit on a ready PR
  GitHub Actions  Code Quality: pre-commit -> test.yml + cpu_device.yml
  Buildkite       Wait for GitHub Layer 2 (CPU pod) -> wait -> Unit Tests (1 GPU)
```

The wait step (`common_scripts/wait_for_github_l2.py`) polls the newest
"Code Quality" run of the PR's head commit in which `Check code quality` ran,
i.e. a ready-for-review run. It passes when that run succeeds and fails when it
fails or is cancelled, so the check stays pending until the GPU step finishes
and the GPU is never used for a commit whose CPU checks failed. It polls through
the GitHub API, so it works the same for PRs from forks. Non-PR builds skip the
wait.

Why each setting:
- The draft filter keeps draft pushes off the GPU queue; drafts only run the
  Python 3.12 CPU tests on GitHub.
- "Ready for review" starts the build when a draft leaves draft without a new
  commit.
- "Skip ... existing build for commit" must be off: converting a PR back to
  draft cancels its Code Quality run, which fails the waiting build, and the
  next ready-for-review on the same commit has to build again.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes)
> before the wait step is uploaded. Add `force-ci` label to the PR to bypass.

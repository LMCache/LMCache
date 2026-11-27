# Buildkite Web UI Setup: Token-dropping GPU Tests

Create an independent `k3-token-dropping-gpu` pipeline using the same repository
and agents as `k3-unit-tests`. Paste `buildkite-pipeline.yml` into its Steps editor.

**GitHub settings**:
- Filter: `build.pull_request.id != null && build.pull_request.draft != true`
- Build when pull request is opened or updated: Yes
- Build when pull request becomes ready for review: Yes
- Skip when pull request has existing build for commit and branch: No
- Cancel Intermediate Builds: Yes

The upload step checks the changed files before requesting a GPU. It runs for
GPU worker/view changes (including the worker's token-dropping dependencies)
and changes to this suite, independent of PR labels. Other ready PRs run only
the upload step and exit without allocating a GPU. Unknown change sets run
the suite.
PRs predating the lane must rebase before using it; an upload-only success is
not GPU test evidence.

The test step reuses the K3 GPU image and environment setup, explicitly pins
vLLM 0.25.1, checks CUDA availability, and rejects skipped or missing test cases.
The GPU view suite is always included; the worker suite is added when the
worker or its test file exists.

Keep the existing branch protection unchanged. For a GPU-dependent fast-lane PR,
verify the actual GPU test step passes on its current HEAD before merging or
enabling auto-merge. During rollout, validate a relevant PR, an unrelated PR
and a GPU-dependent PR with the fast-lane label.

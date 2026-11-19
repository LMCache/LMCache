# Buildkite Web UI Setup: K3 GPU Unit Tests

In the [GitHub settings for `k3-unit-tests`](https://buildkite.com/lmcache/k3-unit-tests/settings/github), configure pull request triggers as follows:

| Setting | Value |
| --- | --- |
| Build when pull request is opened or updated | Yes |
| Build when pull request becomes ready for review | Yes |
| Skip when pull request has existing build for commit and branch | No |
| Filter builds using a conditional | `build.pull_request.draft != true` |

Enable the Ready trigger before enabling the filter. Turning off duplicate
suppression lets a PR build the same commit when marked Ready, even if its
draft build already ran the upload step.

The webhook filter saves the upload job. The matching condition in
[`pipeline.yml`](pipeline.yml) skips the GPU step if a draft build reaches
pipeline upload before the hosted settings are updated. Ready PRs still go
through the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).

To verify the rollout, open a draft PR with a change under
`.buildkite/k3_tests/unit/`. Check that the webhook creates no build. Mark the
same commit Ready and confirm the GPU unit-test step is scheduled. A docs-only
Ready PR should still complete after the upload job.

The condition uses draft status from the triggering event. It applies to new
builds and leaves any GPU job from an earlier Ready event running.

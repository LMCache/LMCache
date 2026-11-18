# Buildkite Web UI Setup: Unit Tests

**Steps editor**: unchanged — the upload step for `.buildkite/k3_tests/unit/pipeline.yml`.

**GitHub settings**:
- Filter builds using a conditional: `build.pull_request.draft != true`
- Build when pull request is opened or updated: Yes
- Build when pull request becomes ready for review: Yes
- Skip when pull request has existing build for commit and branch: **No**
- Skip queued / cancel running branch builds: Yes

**Cluster prerequisite**: a Kubernetes secret with a GitHub token that has
**no scopes**, used only to lift the GitHub API rate limit:

```
kubectl create secret generic github-api-readonly -n buildkite \
    --from-literal=token=<GitHub token without scopes>
```

PR code, including PRs from forks, runs in the pods that read it, so never use
the repository write token from `buildkite-git-creds` here. Create the secret
before this pipeline change lands; the steps cannot start without it.

`buildkite/k3-unit-tests` is a required check on `dev`. Each non-draft PR
commit gets one build:

```
GitHub Actions  Code Quality: pre-commit (L1) -> test.yml + cpu_device.yml (L2)
Buildkite       wait for L1 (CPU pod) -> Unit Tests (1 GPU) in parallel with L2
                                         stopped as soon as an L2 job fails
```

- `github_ci_gate.py wait-l1` polls the newest "Code Quality" run of the PR
  head commit in which `Check code quality` ran (a ready-for-review run) and
  passes once that job succeeds. A first-time contributor's fork PR therefore
  reaches the GPU only after a maintainer approved its GitHub run.
- `run-gpu-with-l2-watch.sh` runs the tests next to `github_ci_gate.py
  watch-l2`. When a `Test /` or `CPU device /` job fails or is cancelled (a new
  push, or the PR going back to draft), it stops the tests and fails the step,
  releasing the GPU. When Layer 2 passes, the tests run on to their own result.
- Non-PR builds skip both. GitHub API errors are retried and never fail or stop
  the tests.

Why each setting:
- The draft filter keeps draft pushes off the GPU queue; drafts only run the
  Python 3.12 CPU tests on GitHub.
- "Ready for review" starts the build when a draft leaves draft without a new
  commit.
- "Skip ... existing build for commit" must be off: going back to draft cancels
  the GitHub run, which fails this build, and the next ready-for-review on the
  same commit has to build again.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes)
> before these steps are uploaded. Add `force-ci` label to the PR to bypass.

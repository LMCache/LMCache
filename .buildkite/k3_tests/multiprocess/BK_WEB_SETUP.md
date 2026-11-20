# Buildkite Web UI Setup: Multiprocess Tests

Multiprocess tests run in two independent, parallel Buildkite pipelines:

| Pipeline | Steps editor | Covers |
|---|---|---|
| `k3-multiprocess-smoke` (PR-smoke) | `buildkite-smoke-pipeline.yml` | 0/1-GPU representative checks from `smoke-pipeline.yml` |
| `k3-multiprocess-test` (heavy) | `buildkite-pipeline.yml` (fill in `HF_TOKEN`) | 2-GPU, Kimi, DSV4 and the full engine/transport matrices from `pipeline.yml` |

`split_manifest.yaml` lists which workload belongs to which tier.

**GitHub trigger settings** (both pipelines):
- Filter: `(build.pull_request.labels includes "mp" || build.pull_request.labels includes "full") && build.pull_request.draft != true`
- Rebuild on PR label change: Yes
- Build when pull request becomes ready for review: Yes
- Skip when pull request has existing build for commit and branch: No (converting
  a PR back to draft cancels its builds, so the next ready-for-review on the same
  commit must build again)
- Skip queued / cancel running branch builds (Skip / Cancel Intermediate Builds): Yes

Draft PRs never build, even when labeled: 123 GPU-h of multiprocess builds ran
on draft commits in the 30-day PR CI baseline (2026-10-17 to 2026-11-15).

Both pipelines start in parallel on the same labels without depending on each
other; `upload-pipeline.sh` decides which tier runs or bypasses:

| PR labels | PR-smoke | Heavy |
|---|---|---|
| `mp` or `full` (auto-merge adds `full`) | runs | bypassed (exits green), unless the PR changes shared CI files (`.buildkite/k3_tests/`, `.buildkite/k3_harness/`, `pyproject.toml`, `setup.py`, `requirements/`, Docker files) |
| + `force-ci` | runs | runs in parallel |
| + `skip expensive ci` | bypassed (exits green) | bypassed (exits green; `force-ci` overrides) |

Setting `MP_FULL_ON_PR=true` on the heavy pipeline (including manual UI/API
rebuilds on a PR) restores heavy on every labeled PR. Heavy still runs on
scheduled builds and on vLLM / SGLang pin verification.

**Rollout order (required to avoid an un-gated merge window):**
1. Create `k3-multiprocess-smoke` on Buildkite with the filter above and add
   `buildkite/k3-multiprocess-smoke` to `dev` Required Checks first.
2. Then merge this PR (which enables heavy bypass on ordinary PRs).

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Changes under `.buildkite/` always run. Add `force-ci` label to the PR to
> bypass.

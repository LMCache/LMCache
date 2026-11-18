# Buildkite Web UI Setup: Multiprocess Tests

Multiprocess tests run in two Buildkite pipelines:

| Pipeline | Steps editor | Covers |
|---|---|---|
| `k3-multiprocess-smoke` (PR-smoke) | `buildkite-smoke-pipeline.yml` | 0/1-GPU representative checks from `smoke-pipeline.yml` |
| `k3-multiprocess-test` (heavy) | `buildkite-pipeline.yml` (fill in `HF_TOKEN`) | 2-GPU, Kimi, DSV4 and the full engine/transport matrices from `pipeline.yml` |

`split_manifest.yaml` lists which workload belongs to which tier.

**GitHub trigger settings** (both pipelines):
- Filter: `build.pull_request.labels includes "mp" || build.pull_request.labels includes "full" || build.pull_request.labels includes "mp-full"`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

Both pipelines start on the same labels; `upload-pipeline.sh` then decides
which one runs:

| PR labels | PR-smoke | Heavy |
|---|---|---|
| `mp` or `full` (auto-merge adds `full`) | runs | skipped, unless the PR changes shared CI files (`.buildkite/k3_tests/`, `.buildkite/k3_harness/`, `pyproject.toml`, `setup.py`, `requirements/`, Docker files) |
| + `force-ci` | runs, then triggers heavy once smoke passes | runs once, from PR-smoke's trigger; its own webhook build exits green |
| + `mp-full` | bypassed (exits green) | runs directly |
| + `skip expensive ci` | skipped | skipped (`force-ci` overrides) |

With `force-ci`, the PR-smoke build waits for the heavy build it triggers, so
`buildkite/k3-multiprocess-smoke` stays pending until heavy finishes. Make that
check required as well, or a PR can merge while heavy is still running: heavy's
own webhook build reports green as soon as it defers.

The `mp-full` label does not exist in the repository by default; create it
before relying on it. Setting `MP_FULL_ON_PR=true` on the heavy pipeline
restores heavy on every labeled PR.

Heavy still runs on scheduled builds and on vLLM / SGLang pin verification.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Changes under `.buildkite/` always run. Add `force-ci` label to the PR to
> bypass.

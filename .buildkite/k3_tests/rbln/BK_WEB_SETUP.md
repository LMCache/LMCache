# Buildkite Web UI Setup: RBLN MP Hardware Smoke

**Steps editor**: paste contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "rbln" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

**Cluster prerequisites**: `rbln-queue` is served by agent-stack-k8s on the
`npu2` node group. The step claims one NPU from the `buildkite/npu1` DRA
ResourceClaimTemplate and runs the public Harbor `rbln-serve` image, whose
`/opt/venv` already holds a matched `torch-rbln` / `rebel-compiler` pair, so
no RBLN Portal credentials are needed. The image runs as uid 2000, which
cannot write `/opt/venv`, so the step sets `runAsUser: 0`. Harbor has no
`latest` tag; bump the pinned tag in `pipeline.yml` deliberately.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Add the `force-ci` label to a PR to bypass.

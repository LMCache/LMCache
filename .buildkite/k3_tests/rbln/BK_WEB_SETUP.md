# Buildkite Web UI Setup: RBLN MP Hardware Smoke

**Steps editor**: paste contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "rbln" || build.pull_request.labels includes "full"`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

**Cluster prerequisites**: `rbln-queue` is served by agent-stack-k8s on the
`npu2` node group. The step claims one NPU from the `buildkite/npu1` DRA
ResourceClaimTemplate and runs the public Harbor `rbln-serve` image, whose
`/opt/venv` already holds a matched `torch-rbln` / `rebel-compiler` pair, so
no RBLN Portal credentials are needed. The image runs as uid 2000, which
cannot write `/opt/venv`, so the step sets `runAsUser: 0`.

**Image selection**: `upload-pipeline.sh` sets the step's image to the last
`rbln-serve` image the nightly lane (`buildkite-nightly-pipeline.yml`)
verified, read from `tested_runtimes.jsonl` on the
`buildkite_latest_tested_vllm` branch by `resolve-pinned-vllm.sh`. Until the
first nightly pin it uses `RBLN_FALLBACK_IMAGE` there. A failed nightly keeps
the previous pin, so an `rbln-serve` regression does not reach PR builds.

```
rbln-serve nightly CI passes      -> version set gets nightly-verified
rbln-mp-nightly smoke passes      -> image@digest appended to tested_runtimes.jsonl
rbln-mp-test (PR / dev) uploads   -> runs on the newest pinned RBLN image
```

Verified images live on `harbor.mgmt.rbln.in`, which rejects anonymous pulls,
so the `rbln-queue` job pods need an image pull secret for it.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Add the `force-ci` label to a PR to bypass.

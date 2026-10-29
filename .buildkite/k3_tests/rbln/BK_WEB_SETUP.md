# Buildkite Web UI Setup: RBLN MP Hardware Smoke

The smoke runs in two Buildkite pipelines that share `pipeline.yml`:

| Pipeline | Trigger | `rbln-serve` image | On failure |
|----------|---------|--------------------|------------|
| `rbln-mp-test` | PRs and `dev` pushes | pinned tag in `pipeline.yml` | blocks the PR |
| `rbln-mp-nightly` | nightly schedule | newest `nightly-verified` image | notifies only |

The pinned tag only moves when someone bumps it, so `rbln-mp-test` cannot see
vLLM-RBLN changes that break LMCache until then. `rbln-mp-nightly` follows the
image rbln-serve's own nightly CI last verified. When it breaks and
`rbln-mp-test` does not, bump the pin and adapt LMCache.

## rbln-mp-test

**Steps editor**: paste contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "rbln" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Add the `force-ci` label to a PR to bypass.

## rbln-mp-nightly

**Steps editor**: paste contents of `buildkite-nightly-pipeline.yml`. Its
`RBLN_SERVE_NIGHTLY_VERIFIED=1` makes `upload-pipeline.sh` run
`resolve_nightly_image.py`, which asks the rbln-serve version-set API for the
latest `nightly-verified` image and passes it to `pipeline.yml` as
`RBLN_SERVE_IMAGE`, pinned by digest:

```bash
$ python3 .buildkite/k3_tests/rbln/resolve_nightly_image.py
harbor.mgmt.rbln.in/rebellions-sw/rbln-serve:0.12.1-dev24-gc43bfa4-ubi10.2@sha256:7c41dd68…
```

If the lookup fails, the upload step fails; it does not fall back to the pin.

**GitHub settings**: turn off build triggers and commit statuses. A separate
pipeline keeps nightly results off PR and `dev` statuses, and keeps
"cancel running branch builds" in `rbln-mp-test` from cancelling the nightly
`dev` build.

**Schedule** (Pipeline Settings → Schedules → New Schedule):
- Cron interval: `0 18 * * *` (UTC; 03:00 KST)
- Branch: `dev`

**Notifications**: add a failed-build notification for this pipeline, for
example a Slack notification service filtered to `rbln-mp-nightly` and failed
builds.

## Cluster prerequisites

`rbln-queue` is served by agent-stack-k8s on the `npu2` node group. The step
claims one NPU from the `buildkite/npu1` DRA ResourceClaimTemplate.

`rbln-mp-test` runs the public Harbor `rbln-serve` image, whose `/opt/venv`
already holds a matched `torch-rbln` / `rebel-compiler` pair, so no RBLN Portal
credentials are needed. The image runs as uid 2000, which cannot write
`/opt/venv`, so the step sets `runAsUser: 0`. Harbor has no `latest` tag; bump
the pinned tag in `pipeline.yml` deliberately.

`rbln-mp-nightly` pulls from `harbor.mgmt.rbln.in`, which does not allow
anonymous pulls. Give the job pods a pull secret for it, for example by adding
a `docker-registry` secret to the `imagePullSecrets` of the service account
the agent-stack-k8s job pods run as. The upload step needs `python3` and HTTPS
access to `rbln-serve.udc.rbln.in`.

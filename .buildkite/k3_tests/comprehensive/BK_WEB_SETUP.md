# Buildkite Web UI Setup: Comprehensive Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

Heavy test (10 parallel GPU steps) — run on `"full"` label or dev push, not every PR.

## Nightly Scheduled Build (rolling baselines)

Create a **Scheduled Build** on this same pipeline to upload performance baselines:

- **Schedule**: daily (e.g. `0 2 * * *` — 2am UTC)
- **Branch**: `dev`
- **Extra Environment Variables**: `NEED_UPLOAD=true`

Each config step writes a date-stamped baseline (`<feature>-YYYYMMDD.json`) as a Buildkite artifact. The finalize step (`scripts/upload-baselines.sh`) collects them all, prunes files older than 5 days, and pushes a single commit to the `benchmarks-main` branch.

PR builds automatically compare against the rolling 5-day worst-case (max latency) baseline.

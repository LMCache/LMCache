# Weekly full build

Per-PR GPU CI runs a narrower set of tests than it used to. The weekly full
build runs the whole CUDA suite once a week on the tip of `dev`, so a
regression the PR tiers miss shows up within a week, between two known
commits.

| | |
|---|---|
| Workflow | [`.github/workflows/weekly_full_build.yml`](../../../.github/workflows/weekly_full_build.yml) |
| Schedule | Manual runs first (**Actions → Weekly full build → Run workflow**, from `dev`); `schedule` (`Sunday 08:00 UTC`) is commented out until initial runs are validated |
| Gate | [`.github/scripts/weekly_full_build.py`](../../../.github/scripts/weekly_full_build.py) |
| What must run | [`manifest.json`](manifest.json) (CUDA), [`manifest-xpu.json`](manifest-xpu.json) (XPU), [`manifest-amd.json`](manifest-amd.json) (AMD), [`manifest-rbln.json`](manifest-rbln.json) (RBLN) |
| Results | The workflow run: its summary links every Buildkite build; per-device artifacts `weekly-full-build*-<n>` keep `report.json`, `summary.md` and failed-job logs for 90 days |

Sunday 08:00 UTC starts the quietest six hours of the `k8s` GPU queue in the
30-day PR CI baseline (2026-10-17 to 2026-11-15): 0.6 of 16 GPUs busy on
average, against 5 to 10 on weekdays.

## What runs

| Pipeline | Content |
|---|---|
| `k3-unit-tests` | All unit tests. Without `--maxfail=1`, so one failure does not hide the next; JUnit report in `junit-unit.xml`. |
| `k3-multiprocess-test` | Every workload of `pipeline.yml`, plus `dsv4_flash_tp`: the full-size DeepSeek-V4-Flash L1 correctness test at TP=4 on vLLM and SGLang. It left per-PR CI in #6143; only the weekly build runs it (`CI_WEEKLY_FULL=1`). |
| `k3-blend-test` | CacheBlend plugin compatibility and coordinator global match. |
| `unit-tests-xpu` | Intel XPU unit tests (`-rfEs`, JUnit report in `junit-xpu-unit.xml`). |
| `xpu-mp-test` | Full Intel XPU multiprocess matrix across `inline` and `shm` transports. |
| `amd-mp-test` | AMD ROCm unit tests (`-rfEs`), vLLM E2E, IPC probe, and ROCm 10 TP=8 stress (single bare-metal agent; queue wait can take 3-5 hours). |
| `rbln-mp-test` | RBLN MP hardware smoke. |

`dsv4_flash_tp` runs with `DSV4_FALLBACK_VLLM_VERSION=""`: it tests the pinned
vLLM as is instead of falling back to an older vLLM when the current one
cannot serve the model.

## How a run is decided

```
weekly_full_build.yml (cron or manual)
  -> push the dev commit to ci-weekly-full
  -> create one Buildkite build per pipeline, env CI_WEEKLY_FULL=1
  -> wait for all builds (up to 330 min)
  -> gate: passed only if
       every build passed
       every job passed, not soft-failed
       every pattern in manifest.json "required" matched a passed job
       no suite was skipped (path filter / skip labels)
       no workload was skipped as unsupported, unless listed in
         "expected_unsupported"
  -> summary + artifact, workflow fails otherwise
```

A job that passed only after an automatic retry still counts as passed. The
summary lists it under "Passed after a retry" so flakiness stays visible.

`CI_WEEKLY_FULL=1` changes three things on Buildkite:

- the path filter never skips (`path-filter.sh`);
- the `DSV4 full-size (4-GPU, weekly)` group runs (`multiprocess/pipeline.yml`);
- a multiprocess workload skipped as unsupported leaves a `weekly-skip-<job>`
  annotation that the gate reads (`multiprocess/run.sh`).

## When the weekly build fails

1. Open the workflow run. The summary lists each failed, missing or skipped
   job with a link to its Buildkite job.
2. Compare with the last green weekly run: the regression is in
   `<last green commit>..<this commit>` on `dev`.
3. A genuine failure stays red until it is fixed. Do not add it to a skip list
   or a retry; open an issue and link the run.
4. If a matrix combination is unsupported on purpose (an engine lacks the
   capability), add a pattern for its label to `expected_unsupported` in the
   same PR that explains why.

When a workload is added to `multiprocess/pipeline.yml` or
`xpu/multiprocess/pipeline.yml`, add it to `required` in `manifest.json` or
`manifest-xpu.json`; `tests/ci/test_weekly_full_build.py` fails otherwise.

## One-time setup (admins)

- **Branch filter.** On `k3-unit-tests`, `k3-multiprocess-test`, `k3-blend-test`,
  `unit-tests-xpu` and `xpu-mp-test`, add `!ci-weekly-full` to the branch filter
  (and ensure `amd-mp-test` / `rbln-mp-test` keep `build_branches: false` or
  add `!ci-weekly-full`), so the push of the CI-owned branch starts no
  duplicate builds. The weekly builds pass
  `ignore_pipeline_branch_filters`. Without the filter, `upload-pipeline.sh`
  still ends those push builds green before any test runs.
- **Token.** The repository secret `BUILD_KITE_API_TOKEN` needs
  `read_builds` and `write_builds` (already set for the XPU nightly) and
  `read_build_logs` to save failed-job logs. Without `read_build_logs` the run
  is still judged; the summary notes that logs were not saved.
- **Model weights.** `dsv4_flash_tp` loads about 160 GB of
  `deepseek-ai/DeepSeek-V4-Flash` from `/data/huggingface`. Pre-download it on
  both GPU nodes, or the first run spends most of its timeout downloading.
- **Blend secrets.** `k3-blend-test` needs its usual `HF_TOKEN`, `GH_TOKEN`
  and `CB_PLUGIN_REPO` in the Buildkite pipeline settings.

#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Buildkite step: run the PRT_SUITE suite and attach its report to the build,
# keeping the suite's exit code. Output stays under $PRT_ROOT/prt-suites/.
#
#   PRT_SUITE=nightly PRT_ROOT=/raid/prt bash .buildkite/perf_regression_testing/run.sh
set -uo pipefail
for v in PRT_SUITE PRT_ROOT; do
  if [[ -z "${!v:-}" ]]; then
    # Not 1 (a regression) or a soft-failed code: a broken schedule must show.
    echo "run.sh: $v is unset; set it in the Buildkite env (see buildkite-pipeline.yml)" >&2
    exit 64
  fi
done
# The time keeps a retried job from reusing the first attempt's dir.
out=$PRT_ROOT/prt-suites/buildkite-${BUILDKITE_BUILD_NUMBER:-local}-$PRT_SUITE-$(date -u +%Y%m%d-%H%M%S)
${PYTHON:-python3.11} -m perf_regression_testing.suite --suite "$PRT_SUITE" --out "$out"
rc=$?
if [[ -f "$out/report.md" ]]; then
  mkdir -p prt-report && cp "$out"/report.* prt-report/
  style=success
  [[ $rc -eq 1 ]] && style=error
  [[ $rc -ne 0 && $rc -ne 1 ]] && style=warning
  buildkite-agent annotate --style "$style" --context "prt-$PRT_SUITE" \
    < "$out/report.md"
fi
exit "$rc"

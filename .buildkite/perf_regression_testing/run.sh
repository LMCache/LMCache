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
# The suite needs Python >= 3.11 (tomllib); hosts differ in what they ship.
if [[ -z "${PYTHON:-}" ]]; then
  for p in python3.13 python3.12 python3.11 python3; do
    if "$p" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      PYTHON=$p && break
    fi
  done
fi
if [[ -z "${PYTHON:-}" ]]; then
  echo "run.sh: no Python >= 3.11 on $(hostname); set PYTHON in the Buildkite env" >&2
  exit 64
fi
out=$PRT_ROOT/prt-suites/buildkite-${BUILDKITE_BUILD_NUMBER:-local}-$PRT_SUITE-$(date -u +%Y%m%d-%H%M%S)
"$PYTHON" -m perf_regression_testing.suite --suite "$PRT_SUITE" --out "$out"
rc=$?
# The host is not reachable from Buildkite: keep every log as an artifact.
if [[ -d "$out" ]]; then
  mkdir -p prt-report
  (cd "$out" && find . \( -name '*.log' -o -name '*.json' -o -name '*.md' \) \
    -not -path '*/aiperf/*' -print0 | xargs -0 -r cp --parents -t "$OLDPWD/prt-report")
fi
if [[ -f "$out/report.md" ]]; then
  style=success
  [[ $rc -eq 1 ]] && style=error
  [[ $rc -ne 0 && $rc -ne 1 ]] && style=warning
  buildkite-agent annotate --style "$style" --context "prt-$PRT_SUITE" \
    < "$out/report.md"
fi
# Mirror the runs to the results repo, whose GitHub Pages site renders them.
# Best effort: the suite's verdict does not depend on it.
publish() {
  local dir auth rc
  dir=$(mktemp -d)
  auth="Authorization: Basic $(printf 'x-access-token:%s' "${PRT_RESULTS_TOKEN:-}" | base64 -w0)"
  git -c http.extraHeader="$auth" clone -q --depth 1 \
      "https://github.com/$PRT_RESULTS_REPO.git" "$dir" &&
    "$PYTHON" -m perf_regression_testing.publish "$out/report.json" \
      "$dir/data/runs.jsonl" --host "$(hostname -s)" \
      --build-url "${BUILDKITE_BUILD_URL:-}" &&
    # A suite with no usable images appends no lines: nothing to push.
    { git -C "$dir" diff --quiet || {
      git -C "$dir" add data/runs.jsonl &&
        git -C "$dir" -c user.name=prt-ci -c user.email=prt-ci@users.noreply.github.com \
          commit -q -m "$PRT_SUITE $(basename "$out")" &&
        git -C "$dir" -c http.extraHeader="$auth" push -q
    }; }
  rc=$?
  rm -rf "$dir"
  return $rc
}
if [[ -n "${PRT_RESULTS_REPO:-}" && -f "$out/report.json" ]]; then
  publish || echo "run.sh: publishing to $PRT_RESULTS_REPO failed; the suite result stands" >&2
fi
exit "$rc"

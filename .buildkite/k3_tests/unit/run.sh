#!/usr/bin/env bash
# Unit test entrypoint for K8s pods.
# Installs LMCache (no vLLM) + test deps, then runs pytest with coverage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-lmcache-only-env.sh
uv pip install -r requirements/test.txt
# Generated gRPC bindings are intentionally not tracked. Generate them only
# after the test dependencies have installed grpcio-tools.
python lmcache/v1/transport/grpc/_proto_gen/_generate.py

# ── Run unit tests with coverage ─────────────────────────────
# PR builds stop at the first failure for fast feedback. The weekly full build
# runs every test so one regression cannot hide the others, and keeps a JUnit
# report of each failure and skip.
pytest_mode_args=(--maxfail=1)
if [[ "${CI_WEEKLY_FULL:-0}" == "1" ]]; then
    pytest_mode_args=(-rfEs --junitxml=junit-unit.xml)
fi
RUN_DEVDAX_L1_INTEGRATION=1 RUN_DAX_L2_INTEGRATION=1 \
LMCACHE_TEST_DEVDAX_L1_PATHS="" LMCACHE_TRACK_USAGE="false" \
pytest "${pytest_mode_args[@]}" --cov=lmcache \
    --cov-report term --cov-report=html:coverage-test \
    --cov-report=xml:coverage-test.xml --html=durations/test.html \
    --ignore=tests/skipped

cat << EOF | buildkite-agent annotate --style "info"
  Read the <a href="artifact://coverage-test/index.html">uploaded coverage report</a>
EOF

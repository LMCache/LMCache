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

# ── Run unit tests with coverage ─────────────────────────────
LMCACHE_TRACK_USAGE="false" \
pytest --maxfail=1 --cov=lmcache \
    --cov-report term --cov-report=html:coverage-test \
    --cov-report=xml:coverage-test.xml --html=durations/test.html \
    --ignore=tests/disagg --ignore=tests/v1/test_pos_kernels.py \
    --ignore=tests/v1/test_nixl_storage.py \
    --ignore=tests/skipped \
    --ignore=tests/v1/storage_backend/test_eic.py

cat << EOF | buildkite-agent annotate --style "info"
  Read the <a href="artifact://coverage-test/index.html">uploaded coverage report</a>
EOF

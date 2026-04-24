#!/usr/bin/env bash
# Unit test entrypoint for K8s pods.
# Installs LMCache (no vLLM) + test deps, then runs pytest with coverage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Per-build scratch dir ────────────────────────────────────
# /scratch is a shared hostPath mount (see pipeline.yml). Give this build
# its own subdirectory so concurrent pods can't clobber each other, and
# clean it up on exit so /data/gds-scratch on the host doesn't grow. Using
# a direct subdir instead of K8s subPathExpr because the latter breaks GDS
# (cuFile rejects bind-mounted paths).
BUILD_TAG="${BUILDKITE_BUILD_ID:-manual-$$}"
export TMPDIR="/scratch/bk-${BUILD_TAG}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}" 2>/dev/null || true' EXIT

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

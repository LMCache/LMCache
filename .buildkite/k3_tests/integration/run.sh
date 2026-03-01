#!/usr/bin/env bash
# E2E integration test for K8s pods.
#
# The old pipeline created a venv, installed everything manually,
# called pick-free-gpu.sh, and had explicit cleanup steps.
# In K8s: setup-env.sh handles the environment, the device plugin
# handles GPU assignment, and pod termination handles cleanup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh

# E2E-specific deps
uv pip install -r requirements/test.txt
uv pip install matplotlib pandas

# ── GPU is already assigned by K8s device plugin ─────────────
echo "Using K8s-assigned GPU(s): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

# ── Set up the test directory the script expects ─────────────
export LM_CACHE_TEST_DIR="${REPO_ROOT}/tests/e2e"

# ── Run the test ─────────────────────────────────────────────
bash .buildkite/scripts/end-to-end-test.sh

# No cleanup step needed — pod termination handles it.

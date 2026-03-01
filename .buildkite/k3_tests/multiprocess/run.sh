#!/usr/bin/env bash
# Multiprocess test entrypoint for K8s pods.
# Thin wrapper: sets up environment, then delegates to scripts/.
# No Docker -- all processes run natively in the pod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh

# Install test extras (lm-eval for eval workload, openai/pandas/matplotlib for benchmarks)
uv pip install 'lm-eval[api]' openai pandas matplotlib

# ── Run the actual test logic ────────────────────────────────
exec bash "${SCRIPT_DIR}/scripts/run-mp-test.sh"

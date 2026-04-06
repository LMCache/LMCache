#!/usr/bin/env bash
# Multiprocess test entrypoint for K8s pods.
# Usage: run.sh <test_name>
#   test_name: lm_eval | vllm_bench | long_doc_qa | long_doc_qa_l2 | fault_tolerance
# Thin wrapper: sets up environment, then delegates to scripts/.
# No Docker -- all processes run natively in the pod.
set -euo pipefail

TEST_NAME="${1:?Usage: $0 <test_name>  (lm_eval|vllm_bench|long_doc_qa|long_doc_qa_l2|fault_tolerance)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh

# Install test extras (lm-eval for eval workload, openai/pandas/matplotlib for benchmarks)
uv pip install 'lm-eval[api]' openai pandas matplotlib

# ── Ensure all scripts are executable ────────────────────────
chmod +x "${SCRIPT_DIR}"/scripts/*.sh

# ── Run the actual test logic ────────────────────────────────
exec bash "${SCRIPT_DIR}/scripts/run-single-test.sh" "$TEST_NAME"

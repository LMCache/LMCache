#!/usr/bin/env bash
# Multiprocess test entrypoint for K8s pods.
# Usage: run.sh <test_name>
#   test_name: lm_eval | lm_eval_preemption | hma_lm_eval_gemma4 | vllm_bench
#              | long_doc_qa | long_doc_qa_l2 | fault_tolerance | deadlock
#              | restart_recovery | gds_smoke_test | p2p | kimi_linear_tp
#              | dsv4_flash_tp | lazy_offload
# Thin wrapper: sets up environment, then delegates to scripts/.
# No Docker -- all processes run natively in the pod.
set -euo pipefail

TEST_NAME="${1:?Usage: $0 <test_name>  (lm_eval|lm_eval_preemption|hma_lm_eval_gemma4|vllm_bench|long_doc_qa|long_doc_qa_l2|fault_tolerance|deadlock|restart_recovery|cache_stats|lazy_offload|http_api|p2p|kimi_linear_tp|dsv4_flash_tp)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Device configuration ─────────────────────────────────────
export TORCH_DEVICE_TYPE="${TORCH_DEVICE_TYPE:-cuda}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-${TORCH_DEVICE_TYPE}}"
export DEVICE_AFFINITY_VAR="${DEVICE_AFFINITY_VAR:-CUDA_VISIBLE_DEVICES}"
export GPU_MEMORY_PROBE_ENABLED="${GPU_MEMORY_PROBE_ENABLED:-1}"
export BATCH_INVARIANT_DEFAULT="${BATCH_INVARIANT_DEFAULT:-1}"
export DEFAULT_MODEL="${DEFAULT_MODEL:-Qwen/Qwen3-14B}"
export LM_EVAL_NUM_CONCURRENT_DEFAULT="${LM_EVAL_NUM_CONCURRENT_DEFAULT:-50}"
export LM_EVAL_VERIFY_MODE_DEFAULT="${LM_EVAL_VERIFY_MODE_DEFAULT:-samples}"
export LM_EVAL_SCORE_MIN_DEFAULT="${LM_EVAL_SCORE_MIN_DEFAULT:-0.80}"

# ── Environment setup ────────────────────────────────────────
SETUP_ENV_SCRIPT="${BK_SETUP_ENV_SCRIPT:-.buildkite/k3_harness/setup-env.sh}"
source "${SETUP_ENV_SCRIPT}"

# Install test extras (lm-eval for eval workload, openai/pandas/matplotlib for benchmarks)
uv pip install 'lm-eval[api]' openai pandas matplotlib

# ── Ensure all scripts are executable ────────────────────────
chmod +x "${SCRIPT_DIR}"/scripts/*.sh

# ── Run the actual test logic ────────────────────────────────
exec bash "${SCRIPT_DIR}/scripts/run-single-test.sh" "$TEST_NAME"

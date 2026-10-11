#!/usr/bin/env bash
# Multiprocess test entrypoint for K8s pods.
# Usage: run.sh <test_name>
#   test_name: lm_eval | lm_eval_preemption | preemption_correctness
#              | hma_lm_eval_gemma4 | vllm_bench
#              | long_doc_qa | high_concurrency | long_doc_qa_l2
#              | fault_tolerance | deadlock | mp_autostart_tp2
#              | restart_recovery | gds_smoke | p2p | kimi_linear_tp
#              | dsv4_flash_tp | lazy_offload
# Thin wrapper: sets up environment, then delegates to scripts/.
# No Docker -- all processes run natively in the pod.
set -euo pipefail

TEST_NAME="${1:?Usage: $0 <test_name>  (lm_eval|lm_eval_preemption|preemption_correctness|hma_lm_eval_gemma4|vllm_bench|long_doc_qa|high_concurrency|long_doc_qa_l2|fault_tolerance|deadlock|mp_autostart_tp2|restart_recovery|cache_stats|lazy_offload|http_api|gds_smoke|p2p|kimi_linear_tp|dsv4_flash_tp)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
ENGINE_ADAPTER="${SCRIPT_DIR}/scripts/engines/${INFERENCE_ENGINE}.sh"

cd "${REPO_ROOT}"

# ── Device configuration ─────────────────────────────────────
export TORCH_DEVICE_TYPE="${TORCH_DEVICE_TYPE:-cuda}"
export DEVICE_AFFINITY_VAR="${DEVICE_AFFINITY_VAR:-CUDA_VISIBLE_DEVICES}"
export LM_EVAL_NUM_CONCURRENT_DEFAULT="${LM_EVAL_NUM_CONCURRENT_DEFAULT:-50}"
export LM_EVAL_VERIFY_MODE_DEFAULT="${LM_EVAL_VERIFY_MODE_DEFAULT:-samples}"
export LM_EVAL_SCORE_MIN_DEFAULT="${LM_EVAL_SCORE_MIN_DEFAULT:-0.80}"
export LMCACHE_REQUEST_TRANSPORT="${LMCACHE_REQUEST_TRANSPORT:-zmq}"
export LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}"

# ── Environment setup ────────────────────────────────────────
if [[ ! -f "$ENGINE_ADAPTER" ]]; then
    echo "Unsupported inference engine '${INFERENCE_ENGINE}': $ENGINE_ADAPTER not found" >&2
    exit 1
fi
source "$ENGINE_ADAPTER"
source "${SCRIPT_DIR}/scripts/workload-discovery.sh"

# Capability checks happen before installing the engine and test dependencies.
# A shared Buildkite matrix can therefore include engines with different
# feature sets without paying setup cost for unsupported combinations.
if resolve_engine_workload "$TEST_NAME" "${SCRIPT_DIR}/scripts" \
        "$INFERENCE_ENGINE"; then
    :
else
    status=$?
    if [[ "$status" -eq "$WORKLOAD_UNSUPPORTED_STATUS" ]]; then
        exit 0
    fi
    exit "$status"
fi

export DEFAULT_MODEL="${DEFAULT_MODEL:-${ENGINE_DEFAULT_MODEL:-Qwen/Qwen3-14B}}"
engine_setup_environment "$REPO_ROOT"
export INFERENCE_ENGINE

# Install test extras (lm-eval for eval workload, openai/pandas/matplotlib for benchmarks)
uv pip install 'lm-eval[api]' openai pandas matplotlib

# ── Ensure all scripts are executable ────────────────────────
find "${SCRIPT_DIR}/scripts" -type f -name '*.sh' -exec chmod +x {} +

# ── Run the actual test logic ────────────────────────────────
exec bash "${SCRIPT_DIR}/scripts/run-single-test.sh" "$TEST_NAME"

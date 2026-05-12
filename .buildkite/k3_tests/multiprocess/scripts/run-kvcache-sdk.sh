#!/usr/bin/env bash
# Run the end-to-end in-memory KV-cache SDK example in the K3s multiprocess
# test pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
OUT_DIR="${RESULTS_DIR}/kvcache_sdk"

mkdir -p "${OUT_DIR}"

copy_logs() {
    cp "${OUT_DIR}/lmcache.log" "${REPO_ROOT}/kvcache_sdk_lmcache.log" \
        2>/dev/null || true
    cp "${OUT_DIR}/vllm.log" "${REPO_ROOT}/kvcache_sdk_vllm.log" \
        2>/dev/null || true
}
trap copy_logs EXIT

echo "=== KV-cache SDK end-to-end test ==="
echo "Results dir: ${OUT_DIR}"

MODEL="${KVCACHE_SDK_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
TMP_DIR="${OUT_DIR}" \
GPU_DEVICE="${KVCACHE_SDK_GPU_DEVICE:-0}" \
LMCACHE_PORT="${KVCACHE_SDK_LMCACHE_PORT:-${LMCACHE_PORT:-6555}}" \
LMCACHE_HTTP_PORT="${KVCACHE_SDK_LMCACHE_HTTP_PORT:-${LMCACHE_HTTP_PORT:-8080}}" \
VLLM_PORT="${KVCACHE_SDK_VLLM_PORT:-${VLLM_PORT:-8000}}" \
CHUNK_SIZE="${KVCACHE_SDK_CHUNK_SIZE:-256}" \
MIN_PROMPT_TOKENS="${KVCACHE_SDK_MIN_PROMPT_TOKENS:-512}" \
FAKE_PREFIX_TOKENS="${KVCACHE_SDK_FAKE_PREFIX_TOKENS:-32}" \
MAX_MODEL_LEN="${KVCACHE_SDK_MAX_MODEL_LEN:-2048}" \
MAX_TOKENS="${KVCACHE_SDK_MAX_TOKENS:-16}" \
GPU_MEM_UTIL="${KVCACHE_SDK_GPU_MEM_UTIL:-0.6}" \
L1_SIZE_GB="${KVCACHE_SDK_L1_SIZE_GB:-8}" \
REQUEST_TIMEOUT="${KVCACHE_SDK_REQUEST_TIMEOUT:-120}" \
STORE_WAIT_TIMEOUT="${KVCACHE_SDK_STORE_WAIT_TIMEOUT:-120}" \
VLLM_BATCH_INVARIANT=1 \
bash "${REPO_ROOT}/examples/kvcache_sdk/run_e2e_kv_edit.sh"

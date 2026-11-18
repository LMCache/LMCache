#!/usr/bin/env bash
# Verify vLLM sleep and wake-up against a live GPU vLLM + LMCache MP deployment.
#
# vLLM's extensible KV cache is VMM memory that the LMCache server imports.
# Sleep frees it only after the connector unregisters it from the server, and
# wake-up has to register the newly committed memory. sleep_mode.py runs two
# sleep/wake-up cycles and checks registration, GPU memory, LMCache hits, and
# output equality; its docstring lists the steps.
#
# The checks read the registration state of LMCache-driven transfer, so the
# workload refuses any other LMCACHE_MP_TRANSFER_MODE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
GPU_FOR_VLLM="${GPU_FOR_VLLM:-0}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TEST_DIR="${RESULTS_DIR}/sleep_mode"

if [[ "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" != "lmcache_driven" ]]; then
    echo "The sleep_mode workload requires LMCACHE_MP_TRANSFER_MODE=lmcache_driven." >&2
    exit 1
fi

mkdir -p "${TEST_DIR}"

echo "=== Sleep Mode Integration Test ==="
echo "Model: ${MODEL}"
echo "vLLM: http://localhost:${VLLM_PORT} (GPU ${GPU_FOR_VLLM})"
echo "LMCache HTTP: http://localhost:${LMCACHE_HTTP_PORT}"
echo "Results dir: ${TEST_DIR}"

python3 "${SCRIPT_DIR}/sleep_mode.py" \
    --engine-url "http://localhost:${VLLM_PORT}" \
    --lmcache-url "http://localhost:${LMCACHE_HTTP_PORT}" \
    --model "${MODEL}" \
    --gpu-index "${GPU_FOR_VLLM}" \
    --results-dir "${TEST_DIR}"

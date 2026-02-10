#!/usr/bin/env bash
set -euo pipefail

# This script tests single request correctness for APC + LMCache hybrid KV Cache Retrieval
# This test does NOT require batch invariant mode, so it works with all attention backends

#######################################
# Required input
#######################################
: "${BUILD_ID:?BUILD_ID must be set}"

#######################################
# Script arguments
#######################################
ATTENTION_BACKEND="${1:-FLASH_ATTN}"  # Default to FLASH_ATTN if not provided
MODEL="${2:-Qwen/Qwen2.5-14B-Instruct}"  # Default model, can be overridden for DeepSeek tests
echo "[INFO] Using attention backend: ${ATTENTION_BACKEND}"
echo "[INFO] Using model: ${MODEL}"

#######################################
# Configuration
#######################################
WORK_LOG="/tmp/build_${BUILD_ID}_correctness.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
ARTIFACT="build_${BUILD_ID}.log"
SERVER_WAIT_TIMEOUT=180
CORRECTNESS_DIR=".buildkite/correctness"

#######################################
# Prerequisite for this script: 
# Requires manual configuration on CI machines
#######################################

# persist uv's cache somewhere stable:
export UV_CACHE_DIR="$HOME/.cache/uv"

# we will try to reuse as much uv cache as possible across jobs
# while pulling latest changes from vllm, LMCache, and other wheel dependencies
source "$HOME/correctness/.venv/bin/activate"

# update dependencies (nightly vllm and LMCache from the PR)
# --refresh-package tells uv to revalidate cached data for that dependency.
# --reinstall would reinstall all dependencies
uv pip install -U vllm \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --refresh-package vllm

# override previous lmcache from previous jobs
# the source installation is from this PR
uv pip install -e . --reinstall-package lmcache

# additional dependencies (please update manually if needed)
# these packages are pretty stable so should not need to
uv pip install aiohttp tqdm pandas huggingface_hub

# Setup local writable sandbox
CI_CACHE_DIR="$PWD/.vllm_cache_${BUILD_ID}"
mkdir -p "$CI_CACHE_DIR"

# not sure if this is needed on H100 (it was on L40s)
export FLASHINFER_WORKSPACE_DIR="$CI_CACHE_DIR/flashinfer"

#######################################
# Helpers
#######################################
collect_artifact() {
    echo "[INFO] Collecting logs into ${ARTIFACT}"
    cat "${WORK_LOG}" "${VLLM_LOG}" > "${ARTIFACT}" 2>/dev/null || true
}

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "[INFO] Stopping vLLM process (PID: ${VLLM_PID})"
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        sleep 5
    fi
}

find_free_port() {
    for p in $(seq 8000 9000); do
        if ! lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "$p"; return
        fi
    done
    exit 1
}

trap 'rc=$?; stop_vllm; collect_artifact; exit $rc' EXIT INT TERM

exec > >(tee -a "${WORK_LOG}") 2>&1

echo "=== DIAGNOSTICS: GPU STATE before CI ==="
nvidia-smi

echo "[INFO] Selecting free GPU for this build..."
source .buildkite/scripts/pick-free-gpu.sh 90000 1
echo "[INFO] Using GPU(s): ${CUDA_VISIBLE_DEVICES}"

#######################################
# Phase 1: LMCache Server with Single Request Test
#######################################
echo "[INFO] Preparing LMCache config (cpu.yaml)..."
cat <<EOF > cpu.yaml
chunk_size: 16
local_cpu: true 
max_local_cpu_size: 50
EOF

PORT=$(find_free_port)
echo "[INFO] Starting LMCACHE vLLM server on port ${PORT}..."

# Note: NOT using VLLM_BATCH_INVARIANT=1 since this is single request test
LMCACHE_CONFIG_FILE=cpu.yaml \
VLLM_SERVER_DEV_MODE=1 \
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --trust-remote-code \
    --enforce-eager \
    --attention-backend "${ATTENTION_BACKEND}" \
    --gpu-memory-utilization 0.8 \
    -cc.level=0 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "[INFO] Waiting for LMCache server readiness..."
READY=false
START_TIME=$(date +%s)
while [ $(($(date +%s) - START_TIME)) -lt $SERVER_WAIT_TIMEOUT ]; do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${MODEL//\//\\/}"; then
        READY=true; break
    fi
    sleep 5
done

if [ "$READY" = false ]; then
    echo "[ERROR] LMCache vLLM failed to start"; exit 1
fi

#######################################
# Phase 2: man bash Correctness Test
#######################################
echo "[TEST] Running technical man bash correctness test..."
CONTEXT="$(man bash | col -b | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==5000) exit}}')"
HALF_CONTEXT="$(man bash | col -b | tr -s '[:space:]' ' ' | awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==2500) exit}}')"

send_completion() {
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "${MODEL}" --arg content "$1" '{model: $model, temperature: 0, max_tokens: 100, messages: [{role:"user",content:$content}]}')" |
        jq -r '.choices[0].message.content'
}

echo "[STEP 1] Full context (LMCache)"
OUT1="$(send_completion "${CONTEXT}")"

echo "[STEP 2] Reset prefix cache"
curl -s -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null

echo "[STEP 3] Half context"
send_completion "${HALF_CONTEXT}" >/dev/null

echo "[STEP 4] Full context again"
OUT2="$(send_completion "${CONTEXT}")"

if [[ "${OUT1}" != "${OUT2}" ]]; then
    echo "[FAIL] man bash output mismatch!"
    echo "[DEBUG] First output:"
    echo "${OUT1}"
    echo "[DEBUG] Second output:"
    echo "${OUT2}"
    exit 1
fi

echo "[PASS] Single request correctness test passed with ${ATTENTION_BACKEND} backend."

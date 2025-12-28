#!/usr/bin/env bash
set -euo pipefail

#######################################
# Required input
#######################################
: "${BUILD_ID:?BUILD_ID must be set}"

#######################################
# Configuration
#######################################
MODEL="meta-llama/Llama-3.2-1B-Instruct"
WORK_LOG="/tmp/build_${BUILD_ID}_correctness.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
ARTIFACT="build_${BUILD_ID}.log"
SERVER_WAIT_TIMEOUT=180

# 1. Setup a local, writable directory in the current workspace
CI_CACHE_DIR="$PWD/.vllm_cache_${BUILD_ID}"
mkdir -p "$CI_CACHE_DIR"

# Export these so EVERY process (cli, vllm, workers) uses this path
export HF_HOME="$CI_CACHE_DIR/huggingface"
export XDG_CACHE_HOME="$CI_CACHE_DIR/xdg"
export FLASHINFER_WORKSPACE_DIR="$CI_CACHE_DIR/flashinfer"

# Auto-activate venv
if [[ -f ".venv/bin/activate" ]]; then
    echo "[INFO] Activating found venv in .venv"
    source .venv/bin/activate
fi

#######################################
# Artifact collection & Cleanup
#######################################
collect_artifact() {
    echo "[INFO] Collecting logs into ${ARTIFACT}"
    cat "${WORK_LOG}" "${VLLM_LOG}" > "${ARTIFACT}" 2>/dev/null || true
}

cleanup() {
    echo "[INFO] Cleaning up vLLM process"
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    # Optional: rm -rf "$CI_CACHE_DIR" 
}

trap 'rc=$?; cleanup; collect_artifact; exit $rc' EXIT INT TERM

#######################################
# Logging Setup
#######################################
exec > >(tee -a "${WORK_LOG}") 2>&1

echo "=== DIAGNOSTICS: GPU STATE ==="
nvidia-smi

#######################################
# Independent Download Phase
#######################################
# We do this here to ensure the directory structure is created with 
# the correct user permissions before vLLM workers touch it.
echo "[INFO] Pre-downloading model weights to $HF_HOME"
uv pip install huggingface_hub
huggingface-cli download "${MODEL}" --cache-dir "$HF_HOME"

#######################################
# Start vLLM
#######################################
find_free_port() {
    for p in $(seq 8000 9000); do
        if ! lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "$p"
            return
        fi
    done
    exit 1
}

PORT="$(find_free_port)"

echo "[INFO] Starting vLLM on port ${PORT}"
echo "[INFO] FLASHINFER_WORKSPACE_DIR is ${FLASHINFER_WORKSPACE_DIR}"

VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --attention-backend FLASH_ATTN \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    >"${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

#######################################
# Wait for readiness
#######################################
echo "[INFO] Waiting for vLLM (Timeout: ${SERVER_WAIT_TIMEOUT}s)"
READY=false
START_TIME=$(date +%s)
while [ $(($(date +%s) - START_TIME)) -lt $SERVER_WAIT_TIMEOUT ]; do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${MODEL//\//\\/}"; then
        echo "[INFO] vLLM is ready"
        READY=true
        break
    fi
    sleep 5
done

if [ "$READY" = false ]; then
    echo "[ERROR] vLLM failed to start"
    echo "=== VLLM LOG (FULL STARTUP) ==="
    cat "${VLLM_LOG}"
    exit 1
fi

#######################################
# Build test contexts
#######################################
echo "[INFO] Generating test contexts from man bash..."
CONTEXT="$(
    man bash | col -b | tr -s '[:space:]' ' ' |
    awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==5000) exit}}'
)"

HALF_CONTEXT="$(
    man bash | col -b | tr -s '[:space:]' ' ' |
    awk '{for(i=1;i<=NF;i++){printf "%s ",$i; if(++c==2500) exit}}'
)"

#######################################
# Request helper
#######################################
send_completion() {
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "${MODEL}" --arg content "$1" \
            '{
                model: $model,
                temperature: 0,
                max_tokens: 100,
                messages: [{role:"user",content:$content}]
            }')" |
        jq -r '.choices[0].message.content'
}

#######################################
# Test flow
#######################################
echo "[STEP 1] Full context"
OUT1="$(send_completion "${CONTEXT}")"

echo "[STEP 2] Reset prefix cache"
curl -s -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null

echo "[STEP 3] Half context"
send_completion "${HALF_CONTEXT}" >/dev/null

echo "[STEP 4] Full context again"
OUT2="$(send_completion "${CONTEXT}")"

echo "[STEP 5] Equality check"
if [[ "${OUT1}" != "${OUT2}" ]]; then
    echo "[FAIL] Output mismatch"
    printf 'FIRST: %s\n' "${OUT1}"
    printf 'SECOND: %s\n' "${OUT2}"
    exit 1
fi

echo "[PASS] Outputs are identical"
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
# Bumping timeout to 180 as JIT compilation on L4 takes significant time
SERVER_WAIT_TIMEOUT=180

# Auto-activate venv
if [[ -f ".venv/bin/activate" ]]; then
    echo "[INFO] Activating found venv in .venv"
    source .venv/bin/activate
fi

#######################################
# Diagnostics & Cleanup
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
    # Cleanup the fake home sandbox
    rm -rf "/tmp/vllm_home_${BUILD_ID}"
}

trap 'rc=$?; cleanup; collect_artifact; exit $rc' EXIT INT TERM

#######################################
# Logging Setup
#######################################
exec > >(tee -a "${WORK_LOG}") 2>&1

echo "=== DIAGNOSTICS: GPU STATE ==="
nvidia-smi

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
REAL_HOME="$HOME"
FAKE_HOME="/tmp/vllm_home_${BUILD_ID}"

# 1. Setup the fake home structure to bypass permission issues in /var/lib/buildkite-agent
mkdir -p "${FAKE_HOME}/.cache/huggingface"

# 2. Symlink ONLY the 'hub' (weights) to save 15GB+ download time
# This leaves 'modules' and 'transformers' writable in our FAKE_HOME sandbox
REAL_HUB="${REAL_HOME}/.cache/huggingface/hub"
mkdir -p "${REAL_HUB}"
ln -sfn "${REAL_HUB}" "${FAKE_HOME}/.cache/huggingface/hub"

echo "[INFO] Starting vLLM on port ${PORT}"
echo "[INFO] Redirecting HOME to ${FAKE_HOME}"

# 3. Export everything globally for the worker processes (vLLM V1)
export HOME="${FAKE_HOME}"
export XDG_CACHE_HOME="${FAKE_HOME}/.cache"
export HF_HOME="${FAKE_HOME}/.cache/huggingface"
export FLASHINFER_WORKSPACE_DIR="${FAKE_HOME}/.cache/flashinfer"

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
# Build test contexts (YOUR ORIGINAL LOGIC)
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
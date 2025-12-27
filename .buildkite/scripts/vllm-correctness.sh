#!/usr/bin/env bash
set -euo pipefail

#######################################
# Required inputs
#######################################
: "${BUILD_ID:?BUILD_ID must be set}"

#######################################
# Configuration
#######################################
MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT="${PORT:-}"
WORK_LOG="/tmp/build_${BUILD_ID}_correctness.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
ARTIFACT="build_${BUILD_ID}.log"

#######################################
# Artifact collection
#######################################
create_artifact() {
    if ls /tmp/build_"${BUILD_ID}"_*.log >/dev/null 2>&1; then
        cat /tmp/build_"${BUILD_ID}"_*.log > "${ARTIFACT}" || true
    else
        echo "No logs found" > "${ARTIFACT}"
    fi
}

#######################################
# Cleanup
#######################################
cleanup() {
    local rc="${1:-0}"
    echo "[INFO] Cleaning up"
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    exit "${rc}"
}

trap 'rc=$?; create_artifact; cleanup $rc' EXIT INT TERM

#######################################
# Logging
#######################################
exec > >(tee -a "${WORK_LOG}") 2>&1

echo "[INFO] Build ID: ${BUILD_ID}"
echo "[INFO] Work log: ${WORK_LOG}"
echo "[INFO] vLLM log: ${VLLM_LOG}"

#######################################
# Utilities
#######################################
find_available_port() {
    local start="${1:-8000}"
    for p in $(seq "${start}" 9000); do
        if ! lsof -iTCP:"${p}" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "${p}"
            return 0
        fi
    done
    return 1
}

#######################################
# Start vLLM
#######################################
PORT="${PORT:-$(find_available_port 8000)}"
echo "[INFO] Using port ${PORT}"

VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
vllm serve "${MODEL}" \
    --port "${PORT}" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    >"${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

#######################################
# Wait for readiness
#######################################
echo "[INFO] Waiting for vLLM to become ready"

for _ in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[INFO] vLLM is ready"
        break
    fi
    sleep 1
done

if ! curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[ERROR] vLLM failed to start"
    echo "----- vLLM log -----"
    sed -n '1,200p' "${VLLM_LOG}" || true
    exit 1
fi

#######################################
# Build test contexts
#######################################
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
    local content="$1"
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg model "${MODEL}" \
            --arg content "${content}" \
            --argjson max_tokens 100 \
            '{
                model: $model,
                temperature: 0,
                max_tokens: $max_tokens,
                messages: [{role:"user",content:$content}]
            }')" |
        jq -r '.choices[0].message.content'
}

#######################################
# Test flow
#######################################
echo "[STEP 1] Full context"
RESULT_1="$(send_completion "${CONTEXT}")"

echo "[STEP 2] Reset prefix cache"
curl -s -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null

echo "[STEP 3] Half context"
send_completion "${HALF_CONTEXT}" >/dev/null

echo "[STEP 4] Full context again"
RESULT_4="$(send_completion "${CONTEXT}")"

echo "[STEP 5] Equality check"
if [[ "${RESULT_1}" != "${RESULT_4}" ]]; then
    echo "[FAIL] Output mismatch"
    echo "----- RESULT 1 -----"
    printf '%s\n' "${RESULT_1}"
    echo "----- RESULT 4 -----"
    printf '%s\n' "${RESULT_4}"
    exit 1
fi

echo "[PASS] Outputs are identical"

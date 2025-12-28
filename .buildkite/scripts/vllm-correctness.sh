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

#######################################
# Artifact collection
#######################################
collect_artifact() {
    cat /tmp/build_"${BUILD_ID}"_*.log > "${ARTIFACT}" 2>/dev/null || {
        echo "No logs found" > "${ARTIFACT}"
    }
}

#######################################
# Cleanup
#######################################
cleanup() {
    echo "[INFO] Cleaning up"
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}

trap 'rc=$?; collect_artifact; cleanup; exit $rc' EXIT INT TERM

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
find_free_port() {
    for p in $(seq 8000 9000); do
        if ! lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "$p"
            return
        fi
    done
    echo "[ERROR] No free port found"
    exit 1
}

#######################################
# Start vLLM
#######################################
PORT="$(find_free_port)"
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
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null; then
        echo "[INFO] vLLM is ready"
        break
    fi
    sleep 1
done

if ! curl -s "http://localhost:${PORT}/v1/models" >/dev/null; then
    echo "[ERROR] vLLM failed to start"
    echo "----- vLLM log (tail) -----"
    tail -n 200 "${VLLM_LOG}" || true
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
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg model "${MODEL}" \
            --arg content "$1" \
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
    echo "----- FIRST -----"
    printf '%s\n' "${OUT1}"
    echo "----- SECOND -----"
    printf '%s\n' "${OUT2}"
    exit 1
fi

echo "[PASS] Outputs are identical"

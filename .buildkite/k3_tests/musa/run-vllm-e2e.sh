#!/usr/bin/env bash
# Run vLLM + LMCache correctness and cache-hit checks on a MUSA agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/e2e-common.sh"

VLLM_PORT="${MUSA_E2E_VLLM_PORT:-18082}"
VLLM_MODEL_NAME="${MUSA_E2E_SERVING_MODEL:-musa-e2e}"
MAX_MODEL_LEN="${MUSA_E2E_MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${MUSA_E2E_GPU_MEMORY_UTILIZATION:-0.35}"
LMCACHE_CONFIG_FILE="${ARTIFACT_DIR}/lmcache-vllm.yaml"
BASELINE_LOG="${ARTIFACT_DIR}/vllm-baseline.log"
LMCACHE_LOG="${ARTIFACT_DIR}/vllm-lmcache.log"
BASELINE_RESULT="${ARTIFACT_DIR}/vllm-baseline.json"
COLD_RESULT="${ARTIFACT_DIR}/vllm-cold.json"
WARM_RESULT="${ARTIFACT_DIR}/vllm-warm.json"
VARIANT_PROMPT_FILE="${ARTIFACT_DIR}/prompt-variant.txt"

VLLM_LAUNCHER_STRING="${MUSA_E2E_VLLM_LAUNCHER:-vllm}"
read -r -a VLLM_LAUNCHER <<< "${VLLM_LAUNCHER_STRING}"
VLLM_PID=""

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${ARTIFACT_DIR}/vllm-preflight.txt"
import torch
import torch_musa  # noqa: F401 - registers torch.musa
import vllm

assert torch.musa.is_available(), "TorchMUSA is unavailable"
print("torch=", torch.__version__)
print("vllm=", getattr(vllm, "__version__", "unknown"))
print("musa_device_count=", torch.musa.device_count())
PY

write_lmcache_config() {
    cat > "${LMCACHE_CONFIG_FILE}" <<EOF
chunk_size: ${MUSA_E2E_CHUNK_SIZE:-256}
local_cpu: true
max_local_cpu_size: ${MUSA_E2E_LOCAL_CPU_SIZE_GB:-1}
EOF
}

launch_vllm() {
    local mode="$1"
    local port="$2"
    local log_file="$3"
    local common_args=(
        serve "${MODEL}"
        --host 127.0.0.1
        --port "${port}"
        --served-model-name "${VLLM_MODEL_NAME}"
        --trust-remote-code
        --max-model-len "${MAX_MODEL_LEN}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --enforce-eager
        --no-enable-prefix-caching
    )
    if [[ "${mode}" == "lmcache" ]]; then
        log "Launching vLLM with LMCache connector"
        env \
            MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES}" \
            LMCACHE_DEVICE_BACKEND=musa \
            LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE}" \
            LMCACHE_LOG_LEVEL="${MUSA_E2E_LOG_LEVEL:-DEBUG}" \
            PYTHONHASHSEED=0 \
            "${VLLM_LAUNCHER[@]}" "${common_args[@]}" \
            --kv-transfer-config \
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
            > "${log_file}" 2>&1 &
    else
        log "Launching vLLM baseline without LMCache"
        env -u LMCACHE_CONFIG_FILE -u LMCACHE_DEVICE_BACKEND \
            MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES}" \
            PYTHONHASHSEED=0 \
            "${VLLM_LAUNCHER[@]}" "${common_args[@]}" \
            > "${log_file}" 2>&1 &
    fi
    VLLM_PID=$!
    register_pid "${VLLM_PID}"
}

require_model
prepare_prompt
prepare_variant_prompt "${VARIANT_PROMPT_FILE}"
write_lmcache_config

log "Stage: vLLM baseline correctness"
launch_vllm baseline "${VLLM_PORT}" "${BASELINE_LOG}"
wait_for_http "${VLLM_PORT}" "vLLM baseline" "${BASELINE_LOG}"
BASELINE_MODEL_ID="$(model_id "${VLLM_PORT}")"
request_completion "${VLLM_PORT}" "${BASELINE_MODEL_ID}" "${BASELINE_RESULT}"
stop_pid "${VLLM_PID}"
VLLM_PID=""

log "Stage: vLLM LMCache cold/warm correctness and cache hit"
launch_vllm lmcache "${VLLM_PORT}" "${LMCACHE_LOG}"
wait_for_http "${VLLM_PORT}" "vLLM LMCache" "${LMCACHE_LOG}"
LMCACHE_MODEL_ID="$(model_id "${VLLM_PORT}")"
request_completion "${VLLM_PORT}" "${LMCACHE_MODEL_ID}" "${COLD_RESULT}"
request_completion_with_prompt \
    "${VLLM_PORT}" \
    "${LMCACHE_MODEL_ID}" \
    "${ARTIFACT_DIR}/vllm-prefix-b.json" \
    "${VARIANT_PROMPT_FILE}"
HITS_BEFORE="$(log_hit_count "${LMCACHE_LOG}")"
request_completion "${VLLM_PORT}" "${LMCACHE_MODEL_ID}" "${WARM_RESULT}"
HITS_AFTER="$(log_hit_count "${LMCACHE_LOG}")"
HIT_DELTA=$((HITS_AFTER - HITS_BEFORE))

compare_completion_text "${BASELINE_RESULT}" "${COLD_RESULT}"
compare_completion_text "${BASELINE_RESULT}" "${WARM_RESULT}"
compare_completion_text "${COLD_RESULT}" "${WARM_RESULT}"

if (( HIT_DELTA < 1 )); then
    fail "vLLM LMCache warm request produced no retrieval signal (delta=${HIT_DELTA}, pattern=${HIT_PATTERN@Q})"
fi

log "vLLM E2E passed: baseline/cold/warm outputs match and retrieval signal delta=${HIT_DELTA}"

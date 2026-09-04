#!/usr/bin/env bash
# Run SGLang + LMCache correctness and cache-hit checks on a MUSA agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/e2e-common.sh"

if [[ "${MUSA_E2E_ENABLE_SGLANG:-1}" != "1" ]]; then
    log "SGLang E2E disabled by MUSA_E2E_ENABLE_SGLANG"
    exit 0
fi

MODEL="${MUSA_SGLANG_MODEL:-${MUSA_E2E_MODEL:-}}"
SGLANG_PORT="${MUSA_SGLANG_PORT:-18083}"
SGLANG_DEVICE="${MUSA_SGLANG_DEVICE:-musa}"
SGLANG_MAX_TOTAL_TOKENS="${MUSA_SGLANG_MAX_TOTAL_TOKENS:-16384}"
SGLANG_MEM_FRACTION_STATIC="${MUSA_SGLANG_MEM_FRACTION_STATIC:-0.50}"
LMCACHE_DAEMON_PORT="${MUSA_SGLANG_LMCACHE_PORT:-6200}"
LMCACHE_DAEMON_HTTP_PORT="${MUSA_SGLANG_LMCACHE_HTTP_PORT:-7200}"
LMCACHE_CONFIG_FILE="${ARTIFACT_DIR}/lmcache-sglang.yaml"
DAEMON_LOG="${ARTIFACT_DIR}/sglang-lmcache-daemon.log"
LMCACHE_LOG="${ARTIFACT_DIR}/sglang-lmcache.log"
BASELINE_LOG="${ARTIFACT_DIR}/sglang-baseline.log"
BASELINE_RESULT="${ARTIFACT_DIR}/sglang-baseline.json"
COLD_RESULT="${ARTIFACT_DIR}/sglang-cold.json"
WARM_RESULT="${ARTIFACT_DIR}/sglang-warm.json"
VARIANT_PROMPT_FILE="${ARTIFACT_DIR}/prompt-variant.txt"
DAEMON_PID=""
SGLANG_PID=""

SGLANG_LAUNCHER_STRING="${MUSA_SGLANG_LAUNCHER:-python3 -m sglang.launch_server}"
read -r -a SGLANG_LAUNCHER <<< "${SGLANG_LAUNCHER_STRING}"

require_model
prepare_prompt
prepare_variant_prompt "${VARIANT_PROMPT_FILE}"

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${ARTIFACT_DIR}/sglang-preflight.txt"
import importlib
import torch
import torch_musa  # noqa: F401 - registers torch.musa

# First Party
from lmcache.v1.platform.musa import ipc_wrapper as musa_ipc

assert torch.musa.is_available(), "TorchMUSA is unavailable"
sglang = importlib.import_module("sglang")
capabilities = {
    "handle_transfer_enabled": musa_ipc.is_musa_handle_transfer_enabled(),
    "memory_ipc_available": musa_ipc.is_musa_memory_ipc_available(),
    "event_ipc_available": musa_ipc.is_musa_event_ipc_available(),
    "block_transfer_available": musa_ipc.is_musa_block_transfer_available(),
}
print("torch=", torch.__version__)
print("sglang=", getattr(sglang, "__version__", "unknown"))
print("musa_device_count=", torch.musa.device_count())
for capability, available in capabilities.items():
    print(f"{capability}=", available)
missing = [name for name, available in capabilities.items() if not available]
assert not missing, f"MUSA SGLang MP prerequisites are unavailable: {missing}"
PY

cat > "${LMCACHE_CONFIG_FILE}" <<EOF
mp_host: 127.0.0.1
mp_port: ${LMCACHE_DAEMON_PORT}
EOF

launch_daemon() {
    log "Stage: SGLang LMCache daemon"
    env LMCACHE_DEVICE_BACKEND=musa \
        LMCACHE_LOG_LEVEL="${MUSA_E2E_LOG_LEVEL:-DEBUG}" \
        lmcache server \
        --host 127.0.0.1 \
        --port "${LMCACHE_DAEMON_PORT}" \
        --http-host 127.0.0.1 \
        --http-port "${LMCACHE_DAEMON_HTTP_PORT}" \
        --chunk-size "${MUSA_E2E_CHUNK_SIZE:-256}" \
        --l1-size-gb "${MUSA_E2E_DAEMON_L1_SIZE_GB:-1}" \
        --eviction-policy LRU \
        > "${DAEMON_LOG}" 2>&1 &
    DAEMON_PID=$!
    register_pid "${DAEMON_PID}"
    wait_for_http "${LMCACHE_DAEMON_HTTP_PORT}" "LMCache daemon" "${DAEMON_LOG}"
}

launch_sglang() {
    local mode="$1"
    local log_file="$2"
    local launch_args
    local args=(
        --model-path "${MODEL}"
        --host 127.0.0.1
        --port "${SGLANG_PORT}"
        --device "${SGLANG_DEVICE}"
        --max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}"
        --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
        --disable-cuda-graph
    )
    if [[ "${mode}" == "lmcache" ]]; then
        args+=(--enable-lmcache --lmcache-config-file "${LMCACHE_CONFIG_FILE}")
    fi
    printf -v launch_args '%q ' "${args[@]}"
    log "Launching SGLang (${mode}) with ${launch_args% }"
    if [[ "${mode}" == "lmcache" ]]; then
        env MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES}" \
            LMCACHE_DEVICE_BACKEND=musa \
            LMCACHE_LOG_LEVEL="${MUSA_E2E_LOG_LEVEL:-DEBUG}" \
            "${SGLANG_LAUNCHER[@]}" "${args[@]}" > "${log_file}" 2>&1 &
    else
        env -u LMCACHE_CONFIG_FILE -u LMCACHE_DEVICE_BACKEND \
            MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES}" \
            "${SGLANG_LAUNCHER[@]}" "${args[@]}" > "${log_file}" 2>&1 &
    fi
    SGLANG_PID=$!
    register_pid "${SGLANG_PID}"
    wait_for_http "${SGLANG_PORT}" "SGLang ${mode}" "${log_file}"
}

daemon_retrievals() {
    curl -fsS --max-time 3 \
        "http://127.0.0.1:${LMCACHE_DAEMON_HTTP_PORT}/metrics" 2>/dev/null \
        | awk '/^lmcache_mp_l1_read_chunks_total([ {]|$)/ { sum += $NF } END { printf "%d", sum + 0 }' \
        || echo 0
}

flush_sglang_cache() {
    curl -fsS -X POST "http://127.0.0.1:${SGLANG_PORT}/flush_cache" >/dev/null || \
        fail "SGLang /flush_cache endpoint is unavailable; cannot prove an LMCache hit"
}

launch_daemon

log "Stage: SGLang LMCache cold/warm correctness"
launch_sglang lmcache "${LMCACHE_LOG}"
LMCACHE_MODEL_ID="$(model_id "${SGLANG_PORT}")"
request_chat_completion "${SGLANG_PORT}" "${LMCACHE_MODEL_ID}" "${COLD_RESULT}"
request_chat_completion \
    "${SGLANG_PORT}" \
    "${LMCACHE_MODEL_ID}" \
    "${ARTIFACT_DIR}/sglang-prefix-b.json" \
    "${VARIANT_PROMPT_FILE}"
flush_sglang_cache
RETRIEVALS_BEFORE="$(daemon_retrievals)"
LOG_HITS_BEFORE="$(log_hit_count "${LMCACHE_LOG}" "${DAEMON_LOG}")"
request_chat_completion "${SGLANG_PORT}" "${LMCACHE_MODEL_ID}" "${WARM_RESULT}"
RETRIEVALS_AFTER="$(daemon_retrievals)"
RETRIEVAL_DELTA=$((RETRIEVALS_AFTER - RETRIEVALS_BEFORE))
LOG_HITS_AFTER="$(log_hit_count "${LMCACHE_LOG}" "${DAEMON_LOG}")"
LOG_HIT_DELTA=$((LOG_HITS_AFTER - LOG_HITS_BEFORE))
stop_pid "${SGLANG_PID}"
SGLANG_PID=""

log "Stage: SGLang baseline output"
launch_sglang baseline "${BASELINE_LOG}"
BASELINE_MODEL_ID="$(model_id "${SGLANG_PORT}")"
request_chat_completion "${SGLANG_PORT}" "${BASELINE_MODEL_ID}" "${BASELINE_RESULT}"

compare_completion_text "${BASELINE_RESULT}" "${COLD_RESULT}"
compare_completion_text "${BASELINE_RESULT}" "${WARM_RESULT}"
compare_completion_text "${COLD_RESULT}" "${WARM_RESULT}"

if (( RETRIEVAL_DELTA < 1 && LOG_HIT_DELTA < 1 )); then
    fail "SGLang LMCache warm request produced no retrieval signal (metrics_delta=${RETRIEVAL_DELTA}, log_delta=${LOG_HIT_DELTA}, pattern=${HIT_PATTERN@Q})"
fi

log "SGLang E2E passed: baseline/cold/warm outputs match and retrieval signal is present (metrics_delta=${RETRIEVAL_DELTA}, log_delta=${LOG_HIT_DELTA})"

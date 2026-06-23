#!/usr/bin/env bash
#
# End-to-end example for the LMCache KV-cache SDK:
#   1. Start an LMCache MP server with HTTP enabled.
#   2. Start vLLM connected to that server via LMCacheMPConnector.
#   3. Run one source inference so vLLM stores KV cache in LMCache.
#   4. Retrieve the source KV cache with lmcache.sdk.
#   5. Store that KV cache under different target token IDs.
#   6. Send a target request to vLLM and print evaluation results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# if [ -d "${REPO_ROOT}/.venv/bin" ]; then
#     export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
# fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TOKENIZER="${TOKENIZER:-}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-${MODEL}}"
LMCACHE_MODEL_NAME="${LMCACHE_MODEL_NAME:-}"
GPU_DEVICE="${GPU_DEVICE:-0}"

LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
SHM_NAME="${SHM_NAME:-/lmcache_kvcache_sdk_e2e}"
USE_SHM="${USE_SHM:-true}"

CHUNK_SIZE="${CHUNK_SIZE:-256}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-$((CHUNK_SIZE * 2))}"
FAKE_PREFIX_TOKENS="${FAKE_PREFIX_TOKENS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
L1_SIZE_GB="${L1_SIZE_GB:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
TMP_DIR="${TMP_DIR:-/tmp/lmcache_kvcache_sdk_e2e}"

mkdir -p "${TMP_DIR}"

LMCACHE_PID=""
VLLM_PID=""
CLEANED_UP="false"
USE_SETSID="false"
if command -v setsid >/dev/null 2>&1; then
    USE_SETSID="true"
fi

start_logged_process() {
    local log_file="$1"
    shift
    if [ "${USE_SETSID}" = "true" ]; then
        setsid "$@" >"${log_file}" 2>&1 &
    else
        "$@" >"${log_file}" 2>&1 &
    fi
    STARTED_PID=$!
}

process_group_id() {
    local pid="$1"
    ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]'
}

process_running() {
    local pid="$1"
    if [ -z "${pid}" ]; then
        return 1
    fi
    if [ "${USE_SETSID}" = "true" ]; then
        local pgid
        pgid="$(process_group_id "${pid}")"
        if [ -n "${pgid}" ]; then
            kill -0 -- "-${pgid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null
        else
            kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null
        fi
    else
        kill -0 "${pid}" 2>/dev/null
    fi
}

signal_process() {
    local signal="$1"
    local pid="$2"
    if [ -z "${pid}" ]; then
        return 0
    fi
    if [ "${USE_SETSID}" = "true" ]; then
        local pgid
        pgid="$(process_group_id "${pid}")"
        if [ -n "${pgid}" ]; then
            kill "-${signal}" -- "-${pgid}" 2>/dev/null || true
        else
            kill "-${signal}" -- "-${pid}" 2>/dev/null \
                || kill "-${signal}" "${pid}" 2>/dev/null \
                || true
        fi
    else
        kill "-${signal}" "${pid}" 2>/dev/null || true
    fi
}

stop_process() {
    local name="$1"
    local pid="$2"
    if ! process_running "${pid}"; then
        return 0
    fi

    echo "Stopping ${name}..."
    signal_process TERM "${pid}"
    timeout 10 tail --pid="${pid}" -f /dev/null >/dev/null 2>&1 || true
    if ! process_running "${pid}"; then
        return 0
    fi

    echo "Force stopping ${name}..."
    signal_process KILL "${pid}"
}

cleanup() {
    if [ "${CLEANED_UP}" = "true" ]; then
        return
    fi
    CLEANED_UP="true"
    trap - EXIT INT TERM
    echo "--- Cleaning up ---"
    stop_process "vLLM" "${VLLM_PID}"
    stop_process "LMCache" "${LMCACHE_PID}"
    wait "${VLLM_PID}" "${LMCACHE_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_url() {
    local url="$1"
    local timeout="${2:-300}"
    local retry_delay=2
    local retries=$(((timeout + retry_delay - 1) / retry_delay))
    if curl -sf \
        --retry "${retries}" \
        --retry-delay "${retry_delay}" \
        --retry-connrefused \
        "${url}" >/dev/null 2>&1; then
        return 0
    fi
    echo "Timed out waiting for ${url}" >&2
    return 1
}

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command not found: ${command_name}" >&2
        exit 1
    fi
}

require_command lmcache
require_command vllm
require_command curl

echo "============================================"
echo "=== LMCache KV SDK end-to-end example ==="
echo "============================================"
echo "Model:              ${MODEL}"
echo "vLLM model name:    ${VLLM_MODEL_NAME}"
echo "LMCache ZMQ port:   ${LMCACHE_PORT}"
echo "LMCache HTTP port:  ${LMCACHE_HTTP_PORT}"
echo "vLLM port:          ${VLLM_PORT}"
echo "Chunk size:         ${CHUNK_SIZE}"
echo "Fake prefix tokens: ${FAKE_PREFIX_TOKENS}"
echo "Batch invariant:   ${VLLM_BATCH_INVARIANT}"
echo "Log dir:            ${TMP_DIR}"
echo "SHM name:           ${SHM_NAME}"
echo "Use SHM:            ${USE_SHM}"

echo ""
echo "=== Step 1: starting LMCache MP server ==="

USE_SHM_ARGS=()
if [ "${USE_SHM}" = "true" ]; then
    USE_SHM_ARGS+=(--shm-name "${SHM_NAME}")
    USE_SHM_ARGS+=(--no-l1-use-lazy)
fi

start_logged_process "${TMP_DIR}/lmcache.log" \
    lmcache server \
    --l1-size-gb "${L1_SIZE_GB}" \
    --eviction-policy LRU \
    --chunk-size "${CHUNK_SIZE}" \
    --port "${LMCACHE_PORT}" \
    --http-port "${LMCACHE_HTTP_PORT}" \
    "${USE_SHM_ARGS[@]}"
LMCACHE_PID="${STARTED_PID}"
echo "LMCache log: ${TMP_DIR}/lmcache.log"

wait_for_url "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" 60 || {
    tail -80 "${TMP_DIR}/lmcache.log" || true
    exit 1
}
echo "LMCache server is ready."

echo ""
echo "=== Step 2: starting vLLM ==="
KV_TRANSFER_CONFIG=$(cat <<EOF
{
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://localhost",
    "lmcache.mp.port": ${LMCACHE_PORT},
    "lmcache.mp.mq_timeout": 10
  }
}
EOF
)

TRUST_REMOTE_CODE_ARGS=()
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
    TRUST_REMOTE_CODE_ARGS+=(--trust-remote-code)
fi

start_logged_process "${TMP_DIR}/vllm.log" \
    env -u VLLM_PORT \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT}" \
    PYTHONHASHSEED=0 \
    vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
    --override-generation-config '{"temperature": "${TEMPERATURE}"}' \
    "${TRUST_REMOTE_CODE_ARGS[@]}" \
    ${EXTRA_VLLM_ARGS:-}
VLLM_PID="${STARTED_PID}"
echo "vLLM log: ${TMP_DIR}/vllm.log"

wait_for_url "http://localhost:${VLLM_PORT}/v1/models" 600 || {
    tail -120 "${TMP_DIR}/vllm.log" || true
    exit 1
}
echo "vLLM is ready."

echo ""
echo "=== Step 3: running KV retrieve/remap/store evaluation ==="
TOKENIZER_ARGS=()
if [ -n "${TOKENIZER}" ]; then
    TOKENIZER_ARGS+=(--tokenizer "${TOKENIZER}")
fi
LMCACHE_MODEL_ARGS=()
if [ -n "${LMCACHE_MODEL_NAME}" ]; then
    LMCACHE_MODEL_ARGS+=(--lmcache-model-name "${LMCACHE_MODEL_NAME}")
fi
TRUST_REMOTE_CODE_DRIVER_ARGS=()
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
    TRUST_REMOTE_CODE_DRIVER_ARGS+=(--trust-remote-code)
fi

python "${SCRIPT_DIR}/e2e_kv_edit.py" \
    --model "${MODEL}" \
    --vllm-model-name "${VLLM_MODEL_NAME}" \
    --lmcache-url "http://localhost:${LMCACHE_HTTP_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --mq-url "tcp://localhost:${LMCACHE_PORT}" \
    --chunk-size "${CHUNK_SIZE}" \
    --min-prompt-tokens "${MIN_PROMPT_TOKENS}" \
    --fake-prefix-tokens "${FAKE_PREFIX_TOKENS}" \
    --max-tokens "${MAX_TOKENS}" \
    --timeout "${REQUEST_TIMEOUT}" \
    "${TOKENIZER_ARGS[@]}" \
    "${LMCACHE_MODEL_ARGS[@]}" \
    "${TRUST_REMOTE_CODE_DRIVER_ARGS[@]}"

echo ""
echo "Logs are under ${TMP_DIR}"

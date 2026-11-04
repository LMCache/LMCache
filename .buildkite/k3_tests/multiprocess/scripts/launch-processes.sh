#!/usr/bin/env bash
# Launch the LMCache MP server, an inference engine, and an optional baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
ENGINE_ADAPTER="${SCRIPT_DIR}/engines/${INFERENCE_ENGINE}.sh"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
if [[ ! -f "$ENGINE_ADAPTER" ]]; then
    echo "Unsupported inference engine '${INFERENCE_ENGINE}': $ENGINE_ADAPTER not found" >&2
    exit 1
fi
source "$ENGINE_ADAPTER"
engine_configure_defaults

LMCACHE_PORT="${LMCACHE_PORT:-6555}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
ENGINE_LAUNCH_PROFILE="${ENGINE_LAUNCH_PROFILE:-default}"

if [[ "${ENGINE_USE_ALL_DEVICES:-false}" == "true" ]]; then
    echo "Using all visible devices for ${ENGINE_NAME} with LMCache"
else
    echo "Using device $GPU_FOR_ENGINE for ${ENGINE_NAME} with LMCache"
fi
echo "Using device $GPU_FOR_BASELINE for ${ENGINE_NAME} baseline"
if [[ "$ENGINE_LAUNCH_PROFILE" == "default" ]]; then
    engine_prepare_launch "$GPU_FOR_ENGINE"
elif ! declare -F engine_launch_profile >/dev/null; then
    echo "${ENGINE_NAME} does not implement launch profile '$ENGINE_LAUNCH_PROFILE'" >&2
    exit 1
fi

lmcache_args=(
    server
    --transport "$LMCACHE_REQUEST_TRANSPORT"
    --l1-size-gb "$CPU_BUFFER_SIZE"
    --eviction-policy LRU
    --max-workers "$MAX_WORKERS"
    --port "$LMCACHE_PORT"
    --supported-transfer-mode "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}"
)

if [[ -n "${LMCACHE_HOST:-}" ]]; then
    lmcache_args+=(--host "$LMCACHE_HOST")
fi
if [[ -n "${CHUNK_SIZE:-}" ]]; then
    lmcache_args+=(--chunk-size "$CHUNK_SIZE")
fi
if [[ -n "${GDS_L1_PATH:-}" ]]; then
    echo "GDS L1 tier enabled; slab directory: $GDS_L1_PATH"
    lmcache_args+=(--gds-l1-path "$GDS_L1_PATH")
fi
if [[ "${L1_USE_LAZY:-true}" == "false" ]]; then
    lmcache_args+=(--no-l1-use-lazy)
    if [[ "${LMCACHE_MP_TRANSFER_MODE:-}" == "engine_driven" ]]; then
        lmcache_args+=(--shm-name "mp_${BUILD_ID}")
        echo "L1 lazy allocation disabled (SHM transport enabled)"
    else
        echo "L1 lazy allocation disabled"
    fi
fi
if [[ "${SEPARATE_OBJECT_GROUPS:-0}" == "1" || "${SEPARATE_OBJECT_GROUPS:-0}" == "true" ]]; then
    lmcache_args+=(--separate-object-groups)
fi

server_environment=()
if [[ "${ENGINE_USE_ALL_DEVICES:-false}" != "true" ]]; then
    server_environment+=("${DEVICE_AFFINITY_VAR}=${GPU_FOR_ENGINE}")
fi
engine_add_lmcache_server_environment server_environment

> "$PID_FILE"
echo "=== Launching LMCache MP server ==="
echo "Port: $LMCACHE_PORT"
env "${server_environment[@]}" \
    lmcache "${lmcache_args[@]}" \
    > "/tmp/build_${BUILD_ID}_lmcache.log" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"

echo "Waiting for LMCache to initialize..."
sleep 10

echo "=== Launching ${ENGINE_NAME} with LMCache ==="
echo "Model: $MODEL"
echo "Port: $ENGINE_PORT"
if [[ "$ENGINE_LAUNCH_PROFILE" == "default" ]]; then
    engine_launch lmcache "$ENGINE_PORT" "$GPU_FOR_ENGINE" "$ENGINE_LOG_FILE"
else
    engine_launch_profile "$ENGINE_LAUNCH_PROFILE" lmcache "$ENGINE_PORT" \
        "$GPU_FOR_ENGINE" "$ENGINE_LOG_FILE"
fi
echo "$ENGINE_PID" >> "$PID_FILE"
echo "${ENGINE_NAME} with LMCache started (PID=$ENGINE_PID)"
LMCACHE_ENGINE_PID="$ENGINE_PID"

if [[ "${LAUNCH_BASELINE:-true}" == "true" ]]; then
    echo "=== Launching ${ENGINE_NAME} baseline ==="
    echo "Port: $ENGINE_BASELINE_PORT"
    engine_launch baseline "$ENGINE_BASELINE_PORT" "$GPU_FOR_BASELINE" \
        "$ENGINE_BASELINE_LOG_FILE"
    echo "$ENGINE_PID" >> "$PID_FILE"
    echo "${ENGINE_NAME} baseline started (PID=$ENGINE_PID)"
    BASELINE_ENGINE_PID="$ENGINE_PID"
else
    echo "=== Skipping ${ENGINE_NAME} baseline (LAUNCH_BASELINE=false) ==="
fi

echo "=== All processes launched ==="
echo "PIDs: LMCache=$LMCACHE_PID, ${ENGINE_NAME}=$LMCACHE_ENGINE_PID, Baseline=${BASELINE_ENGINE_PID:-skipped}"

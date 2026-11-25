#!/usr/bin/env bash
# Run vLLM on the lmcache-client wheel against a server from the full lmcache.
#
# /opt/venv (vLLM and the editable full lmcache from setup-env.sh) runs the
# server. vLLM runs from a second venv that has the same vLLM and torch plus
# the client wheel built from this checkout. In each transfer mode the server
# must log a store for the cold request, and after vLLM's local prefix cache is
# cleared the same request must be retrieved from LMCache with the same output.
# lmcache_driven runs first and fails the step. engine_driven then runs on the
# same wheel and venv; when only it fails, the script exits
# ENGINE_DRIVEN_FAILED.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
source "${SCRIPT_DIR}/../helpers.sh"
source "${SCRIPT_DIR}/../verification/l1-round-trip-verification.sh"

ENGINE_DRIVEN_FAILED=42
MODEL="${MODEL:-${DEFAULT_MODEL:-Qwen/Qwen3-14B}}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
WORK_DIR="/tmp/client_wheel_${BUILD_ID}"
WORKER_VENV="${WORK_DIR}/worker-venv"
LOG_PREFIX="/tmp/build_${BUILD_ID}_client_wheel"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-20}"
MAX_TOKENS="${MAX_TOKENS:-32}"
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-10}"

cd "$REPO_ROOT"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR" "$RESULTS_DIR/client_wheel"

echo "--- :package: Building lmcache-client from $(git rev-parse --short HEAD)"
uv pip install build
# --skip-dependency-check: build against the pod's torch, not the pinned one.
if ! CLIENT_VERSION="0.0.0+ci" .github/scripts/build_client_wheel.sh \
        "${WORK_DIR}/wheels" --no-isolation --skip-dependency-check \
        > "${LOG_PREFIX}_build.log" 2>&1; then
    tail -n 100 "${LOG_PREFIX}_build.log"
    exit 1
fi
CLIENT_WHEEL="$(ls "${WORK_DIR}"/wheels/lmcache_client-*.whl)"
echo "Built ${CLIENT_WHEEL##*/}"

echo "--- :python: Creating the vLLM venv with the client wheel"
VLLM_VERSION="$(python3 -c 'import vllm; print(vllm.__version__)')"
TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)')"
index_args=(
    --extra-index-url https://wheels.vllm.ai/nightly/cu130
    --extra-index-url https://download.pytorch.org/whl/cu130
    --index-strategy unsafe-best-match
)
if [[ -n "${PINNED_VLLM_ARCHIVE_INDEX_URL:-}" ]]; then
    index_args+=(--extra-index-url "${PINNED_VLLM_ARCHIVE_INDEX_URL}")
fi
uv venv --python 3.12 "$WORKER_VENV"
uv pip install --python "${WORKER_VENV}/bin/python" \
    "vllm==${VLLM_VERSION}" "torch==${TORCH_VERSION}" "${index_args[@]}"
# The CUDA IPC path needs cupy and cuda-python, which the CPU wheel does not
# declare.
uv pip install --python "${WORKER_VENV}/bin/python" "$CLIENT_WHEEL" \
    -r requirements/cuda13_core.txt

(cd /tmp && "${WORKER_VENV}/bin/python" -I - <<'PY'
from importlib import metadata
from pathlib import Path

import lmcache

client = metadata.distribution("lmcache-client")
init = client.locate_file("lmcache/__init__.py")
assert Path(lmcache.__file__).samefile(str(init)), lmcache.__file__
try:
    metadata.distribution("lmcache")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("FAIL: the full lmcache distribution is in the vLLM venv")
print(f"vLLM venv: lmcache-client {client.version} at {lmcache.__file__}")
PY
)
(cd /tmp && python3 -c 'import lmcache; print("Server env: lmcache at", lmcache.__file__)')
CLIENT_NATIVE="$(cd /tmp && "${WORKER_VENV}/bin/python" -I -c '
from importlib import metadata
print(metadata.distribution("lmcache-client").locate_file("lmcache/lmcache_native"))')"

engine_prepare_launch "$GPU_FOR_ENGINE"

# Usage: run_mode <lmcache_driven|engine_driven>
run_mode() (
    set -euo pipefail
    local mode="$1"
    local vllm_log="${LOG_PREFIX}_${mode}_vllm.log"
    local http_port server_pid vllm_pid=""
    LMCACHE_LOG="${LOG_PREFIX}_${mode}_lmcache.log"
    PROMPT_FILE="${RESULTS_DIR}/client_wheel/${mode}_prompt.txt"
    OUT_A="${RESULTS_DIR}/client_wheel/${mode}_cold.txt"
    OUT_B="${RESULTS_DIR}/client_wheel/${mode}_retrieved.txt"
    LMCACHE_PORT="$(find_free_port 16555)"
    http_port="$(find_free_port 18580)"
    ENGINE_PORT="$(find_free_port 18100)"

    local -a server_args=(
        server --transport grpc --port "$LMCACHE_PORT" --http-port "$http_port"
        --l1-size-gb "$CPU_BUFFER_SIZE" --eviction-policy LRU --max-workers 2
        --supported-transfer-mode "$mode"
    )
    if [[ "$mode" == "engine_driven" ]]; then
        server_args+=(--no-l1-use-lazy --shm-name "mp_${BUILD_ID}_client")
    fi

    echo "--- :rocket: ${mode}: LMCache server from the full lmcache"
    env "${DEVICE_AFFINITY_VAR}=${GPU_FOR_ENGINE}" \
        "VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}" LMCACHE_TRACK_USAGE=false \
        lmcache "${server_args[@]}" > "$LMCACHE_LOG" 2>&1 &
    server_pid=$!
    echo "$server_pid" >> "$PID_FILE"
    # The next mode reuses the GPU, so wait until both processes exit.
    trap 'kill $vllm_pid "$server_pid" 2>/dev/null; wait $vllm_pid "$server_pid" 2>/dev/null; sleep 5' EXIT
    local _
    for _ in $(seq 120); do
        curl --noproxy '*' -sf "http://localhost:${http_port}/healthcheck" > /dev/null && break
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -n 100 "$LMCACHE_LOG"
            return 1
        fi
        sleep 1
    done

    local kv_transfer_config
    kv_transfer_config="$(MODE="$mode" PORT="$LMCACHE_PORT" python3 -c '
import json, os
print(json.dumps({
    "kv_connector": "LMCacheMPConnector",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.mp.host": "grpc://localhost",
        "lmcache.mp.port": int(os.environ["PORT"]),
        "lmcache.mp.mq_timeout": 60,
        "lmcache.mp.mp_transfer_mode": os.environ["MODE"],
    },
}))')"

    echo "--- :rocket: ${mode}: vLLM from the client-wheel venv"
    env -u VLLM_PORT "${DEVICE_AFFINITY_VAR}=${GPU_FOR_ENGINE}" \
        "VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}" "${VLLM_ATTENTION_ENV[@]}" \
        VLLM_SERVER_DEV_MODE=1 PYTHONHASHSEED=0 \
        "VLLM_BATCH_INVARIANT=${BATCH_INVARIANT:-${BATCH_INVARIANT_DEFAULT}}" \
        "${WORKER_VENV}/bin/vllm" serve "$MODEL" \
            --load-format dummy --max-model-len "${MAX_MODEL_LEN:-16384}" --max-num-seqs 4 \
            --enforce-eager --no-async-scheduling \
            "${VLLM_ATTENTION_ARGS[@]}" "${VLLM_GPU_MEMORY_ARGS[@]}" \
            --port "$ENGINE_PORT" --kv-transfer-config "$kv_transfer_config" \
            > "$vllm_log" 2>&1 &
    vllm_pid=$!
    echo "$vllm_pid" >> "$PID_FILE"
    wait_for_server "$ENGINE_PORT" 600 "$vllm_log"

    if ! grep -qsF "$CLIENT_NATIVE" /proc/[0-9]*/maps; then
        echo "FAIL: no process loaded ${CLIENT_NATIVE}*.so from the client wheel"
        return 1
    fi
    echo "vLLM loaded lmcache_native from ${CLIENT_NATIVE%/*}"

    l1_round_trip_build_prompt "$PROMPT_FILE" ""
    l1_round_trip_run "client_wheel ${mode}"
    local stores
    stores="$(grep -c "Stored" "$LMCACHE_LOG" || true)"
    echo "LMCache stores logged: ${stores}"
    if ((stores < 1)); then
        echo "FAIL: the server logged no store for the cold request"
        return 1
    fi
    echo "PASS: ${mode} on the lmcache-client wheel"
)

# Called outside a tested context so that set -e applies inside run_mode.
set +e
run_mode lmcache_driven
lmcache_driven_status=$?
set -e
if [[ "$lmcache_driven_status" -ne 0 ]]; then
    echo "FAIL: lmcache_driven on the lmcache-client wheel"
    exit 1
fi
set +e
run_mode engine_driven
engine_driven_status=$?
set -e
if [[ "$engine_driven_status" -ne 0 ]]; then
    echo "engine_driven on the lmcache-client wheel failed;" \
        "logs: ${LOG_PREFIX}_engine_driven_*.log"
    exit "$ENGINE_DRIVEN_FAILED"
fi
echo "PASS: lmcache_driven and engine_driven on the lmcache-client wheel"

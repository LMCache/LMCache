#!/usr/bin/env bash
# P2P write end-to-end test: a main LMCache server that stores into a headless
# peer, in a single 1-GPU pod over localhost.
#
# Topology:
#   coordinator               : http://127.0.0.1:12300
#   main LMCache (GPU 0)      : request 6555, http 7555, p2p 127.0.0.1:8555,
#                               --enable-p2p-write, small L1
#   headless peer LMCache     : request 6556, http 7556, p2p 127.0.0.1:8556,
#                               no engine connects to it
#   vLLM (GPU 0)              : port 8000, connector -> main LMCache,
#                               no prefix cache of its own
#
# Flow (requests and checks are in p2p_write_verify.py):
#   1. The two servers discover each other through the coordinator; the main
#      server's P2P adapter reports write_enabled.
#   2. Fill: distinct long prompts go to vLLM until the two servers together
#      hold more KV than the main server's L1 can. Prompts stored while the
#      main L1 is below --p2p-write-l1-usage-threshold must stay local; later
#      ones must land on the peer.
#   3. Replay: every prompt is sent again and must be served from LMCache with
#      the same answer, including the prompts whose KV exists only on the peer.
#      The replay must not change which server holds which object.
#
# Self-contained: launches every process itself and records PIDs in the shared
# PID file so cleanup.sh tears them down on exit.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
source "${REPO_ROOT}/.buildkite/k3_tests/multiprocess/scripts/engines/vllm.sh"
engine_configure_defaults

MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# The main L1 is small so a handful of prompts fills it; the peer has room for
# everything the main server hands over.
MAIN_L1_SIZE_GB="${MAIN_L1_SIZE_GB:-4}"
PEER_L1_SIZE_GB="${PEER_L1_SIZE_GB:-8}"
P2P_WRITE_USAGE_THRESHOLD="${P2P_WRITE_USAGE_THRESHOLD:-0.5}"

# Ports (localhost, single pod).
COORDINATOR_PORT="${COORDINATOR_PORT:-12300}"
COORDINATOR_URL="http://127.0.0.1:${COORDINATOR_PORT}"
MAIN_PORT="${LMCACHE_PORT:-6555}"
MAIN_HTTP_PORT="${MAIN_HTTP_PORT:-7555}"
MAIN_P2P_PORT="${MAIN_P2P_PORT:-8555}"
PEER_PORT="${PEER_PORT:-6556}"
PEER_HTTP_PORT="${PEER_HTTP_PORT:-7556}"
PEER_P2P_PORT="${PEER_P2P_PORT:-8556}"
MAIN_HTTP_URL="http://127.0.0.1:${MAIN_HTTP_PORT}"
PEER_HTTP_URL="http://127.0.0.1:${PEER_HTTP_PORT}"

# Every cached token on replay must be attributable to LMCache.
export VLLM_DISABLE_PREFIX_CACHING=true

PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
> "$PID_FILE"

launch_lmcache() {
    # $1=request_port $2=http_port $3=p2p_port $4=l1_size_gb $5=instance_id
    # $6=logname, remaining arguments are extra server flags
    local request_port="$1" http_port="$2" p2p_port="$3" l1_size_gb="$4"
    local instance_id="$5" log_name="$6"
    shift 6
    env "${DEVICE_AFFINITY_VAR}=${GPU_FOR_ENGINE}" \
    "VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}" \
    lmcache server \
        --transport "$LMCACHE_REQUEST_TRANSPORT" \
        --supported-transfer-mode "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" \
        --l1-size-gb "$l1_size_gb" \
        --eviction-policy LRU \
        --max-workers "$MAX_WORKERS" \
        --port "$request_port" \
        --http-port "$http_port" \
        --host 127.0.0.1 \
        --instance-id "$instance_id" \
        --coordinator-url "$COORDINATOR_URL" \
        --coordinator-advertise-ip 127.0.0.1 \
        --p2p-advertise-url "127.0.0.1:${p2p_port}" \
        "$@" \
        > "/tmp/build_${BUILD_ID}_${log_name}.log" 2>&1 &
    echo "$!" >> "$PID_FILE"
}

wait_for_url() {
    # $1=url $2=name $3=logfile
    local deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl --noproxy '*' -sf "$1" > /dev/null 2>&1; then
            echo "$2 is up."
            return 0
        fi
        sleep 1
    done
    echo "FAIL: $2 did not come up in ${MAX_WAIT_SECONDS}s"
    tail -50 "$3" 2>/dev/null || true
    return 1
}

main_writes_to_a_peer() {
    curl --noproxy '*' -sf "${MAIN_HTTP_URL}/status" 2>/dev/null | python3 -c "
import json, sys
adapters = json.load(sys.stdin)['storage_manager']['l2_adapters']
sys.exit(0 if any(adapter.get('write_enabled') for adapter in adapters) else 1)
" 2>/dev/null
}

# ── Step 1: coordinator ──────────────────────────────────────
echo "=== Launching coordinator on ${COORDINATOR_URL} ==="
lmcache coordinator --host 127.0.0.1 --port "$COORDINATOR_PORT" \
    > "/tmp/build_${BUILD_ID}_coordinator.log" 2>&1 &
echo "$!" >> "$PID_FILE"
wait_for_url "${COORDINATOR_URL}/healthz" "coordinator" \
    "/tmp/build_${BUILD_ID}_coordinator.log"

# ── Step 2: main server (writes to peers) and headless peer ──
echo "=== Launching LMCache servers ==="
launch_lmcache "$MAIN_PORT" "$MAIN_HTTP_PORT" "$MAIN_P2P_PORT" \
    "$MAIN_L1_SIZE_GB" "p2p-write-main-${BUILD_ID}" "lmcache_main" \
    --enable-p2p-write \
    --p2p-write-l1-usage-threshold "$P2P_WRITE_USAGE_THRESHOLD"
launch_lmcache "$PEER_PORT" "$PEER_HTTP_PORT" "$PEER_P2P_PORT" \
    "$PEER_L1_SIZE_GB" "p2p-write-peer-${BUILD_ID}" "lmcache_peer"
wait_for_url "${MAIN_HTTP_URL}/status" "main LMCache server" \
    "/tmp/build_${BUILD_ID}_lmcache_main.log"
wait_for_url "${PEER_HTTP_URL}/status" "headless peer LMCache server" \
    "/tmp/build_${BUILD_ID}_lmcache_peer.log"

# ── Step 3: the main server must get a write-enabled peer adapter ──
echo "=== Waiting for the main server to discover the peer ==="
discovered=false
for _ in $(seq 1 "$MAX_WAIT_SECONDS"); do
    if main_writes_to_a_peer; then
        echo "The main server has a write-enabled P2P adapter."
        discovered=true
        break
    fi
    sleep 1
done
if [ "$discovered" != "true" ]; then
    echo "FAIL: the main server got no write-enabled P2P adapter in time"
    echo "--- main log (tail) ---"; tail -50 "/tmp/build_${BUILD_ID}_lmcache_main.log" || true
    echo "--- peer log (tail) ---"; tail -50 "/tmp/build_${BUILD_ID}_lmcache_peer.log" || true
    echo "--- coordinator instances ---"; curl -s "${COORDINATOR_URL}/instances" || true
    exit 1
fi

# ── Step 4: one vLLM, connected to the main server ───────────
echo "=== Launching vLLM ==="
export LMCACHE_PORT="$MAIN_PORT"
engine_prepare_launch "$GPU_FOR_ENGINE"
engine_launch lmcache "$ENGINE_PORT" "$GPU_FOR_ENGINE" "$ENGINE_LOG_FILE"
echo "$ENGINE_PID" >> "$PID_FILE"
wait_for_server "$ENGINE_PORT" "$MAX_WAIT_SECONDS" "$ENGINE_LOG_FILE"

# ── Step 5: fill, replay and check ───────────────────────────
echo "=== Running the P2P write workload ==="
if ! python3 "${SCRIPT_DIR}/p2p_write_verify.py" \
        --engine-url "http://127.0.0.1:${ENGINE_PORT}" \
        --main-url "$MAIN_HTTP_URL" \
        --peer-url "$PEER_HTTP_URL" \
        --model "$MODEL" \
        --usage-threshold "$P2P_WRITE_USAGE_THRESHOLD" \
        --results-dir "$RESULTS_DIR/p2p_write"; then
    echo "--- main log (tail) ---"; tail -50 "/tmp/build_${BUILD_ID}_lmcache_main.log" || true
    echo "--- peer log (tail) ---"; tail -50 "/tmp/build_${BUILD_ID}_lmcache_peer.log" || true
    exit 1
fi

echo "============================================"
echo "=== P2P write test PASSED ==="
echo "============================================"

#!/usr/bin/env bash
# Verify that LMCacheMPConnector auto-starts the MP server under vLLM TP=2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
LMCACHE_PORT="${LMCACHE_PORT:-15555}"
VLLM_PORT="${VLLM_PORT:-8000}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm_autostart_tp2.log"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-20}"
MAX_WORKERS="${MAX_WORKERS:-2}"

echo "=== LMCache MP autostart TP=2 smoke test ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT"
echo "vLLM port: $VLLM_PORT"

if lsof -iTCP:"${LMCACHE_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FAIL: LMCache port ${LMCACHE_PORT} is already listening before vLLM starts"
    exit 1
fi

SERVER_ARGS="--l1-size-gb ${CPU_BUFFER_SIZE} --eviction-policy LRU --max-workers ${MAX_WORKERS} --supported-transfer-mode non_gpu"
KV_TRANSFER_CONFIG="$(
    LMCACHE_PORT="${LMCACHE_PORT}" \
    SERVER_ARGS="${SERVER_ARGS}" \
    python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.port": int(os.environ["LMCACHE_PORT"]),
                "lmcache.mp.mq_timeout": 60,
                "lmcache.mp.autostart": True,
                "lmcache.mp.autostart.wait_timeout": 120,
                "lmcache.mp.autostart.server_args": os.environ["SERVER_ARGS"],
            },
        }
    )
)
PY
)"

SAVED_VLLM_PORT="$VLLM_PORT"
unset VLLM_PORT

echo "=== Launching vLLM TP=2 with LMCache MP autostart enabled ==="
VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp \
    --load-format dummy \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.5 \
    --attention-backend FLASH_ATTN \
    --port "$SAVED_VLLM_PORT" \
    --no-async-scheduling \
    --enforce-eager \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
echo "vLLM started (PID=$VLLM_PID)"

VLLM_PORT="$SAVED_VLLM_PORT"

echo "=== Waiting for vLLM to become ready ==="
if ! wait_for_server "$VLLM_PORT" 600 "$VLLM_LOG"; then
    exit 1
fi

echo "=== Verifying vLLM auto-started the MP server ==="
if ! grep -q "Auto-starting LMCache MP server" "$VLLM_LOG"; then
    echo "FAIL: vLLM log does not show MP server auto-start"
    tail -200 "$VLLM_LOG" 2>/dev/null || true
    exit 1
fi

if ! grep -q "LMCache MP server became healthy" "$VLLM_LOG"; then
    echo "FAIL: vLLM log does not show MP server becoming healthy"
    tail -200 "$VLLM_LOG" 2>/dev/null || true
    exit 1
fi

LMCACHE_PORT="${LMCACHE_PORT}" python3 - <<'PY'
import os
import sys

import zmq

from lmcache.integration.vllm.mp_server_launcher import is_mp_server_healthy

server_url = f"tcp://localhost:{os.environ['LMCACHE_PORT']}"
context = zmq.Context()
try:
    if not is_mp_server_healthy(server_url, context, timeout=5.0):
        print(f"FAIL: auto-started MP server is not healthy at {server_url}", file=sys.stderr)
        sys.exit(1)
finally:
    context.term()
print(f"MP server responded to ZMQ PING at {server_url}")
PY

echo "PASS: vLLM TP=2 auto-started the LMCache MP server"

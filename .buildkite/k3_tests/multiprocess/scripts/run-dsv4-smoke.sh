#!/usr/bin/env bash
# DeepSeek-V4-Flash smoke test for MP mode.
#
# Goal: verify the model can be served end-to-end through the LMCache
# MP connector and that the cache is actually exercised (store on the
# first request, retrieve from L1 on the second).
#
# What this test does NOT do:
#   * No accuracy / lm-eval check -- this is a smoke test, not a quality gate.
#   * No L2 backend -- L1 (CPU) only, fastest possible boot.
#
# Pass criteria (all three must hold):
#   1. Both /v1/completions responses are valid JSON with non-empty text.
#   2. The LMCache server log contains a "Prefetch request completed (L1+L2)"
#      line with at least 1 L1 hit, emitted AFTER the second request was sent.
#
# Runtime budget: ~3-5 min on a 4-GPU node (--load-format dummy skips weight
# download/load, so vLLM ready in ~60-120s for DSV4-Flash).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# ── Configuration ───────────────────────────────────────────
MODEL="deepseek-ai/DeepSeek-V4-Flash"
LMCACHE_PORT="${LMCACHE_PORT:-15554}"
VLLM_PORT="${VLLM_PORT:-8000}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
CHUNK_SIZE=256

LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"

# ── Cleanup ────────────────────────────────────────────────
# vLLM with TP=4 + --distributed-executor-backend mp forks one parent +
# 4 worker processes. SIGTERM to the parent alone is not guaranteed to
# cascade to the workers, so leaked workers can hold ~10s of GB of GPU
# memory each and OOM the next PR's run. Mitigation:
#   1. Reverse order: kill vLLM (last entry) before LMCache, otherwise
#      vLLM's MP connector logs spurious socket-hang-up errors that
#      pollute the failure diagnostics.
#   2. pkill -P <pid> first, so all TP workers (children of the vLLM
#      parent) receive the signal too.
#   3. Then signal the parent itself.
#   4. Two-phase: SIGTERM, wait up to 10s for graceful exit, then
#      SIGKILL whatever is still alive.
cleanup() {
    echo "=== Cleaning up ==="
    [[ -f "$PID_FILE" ]] || { sleep 2; return; }

    mapfile -t pids < "$PID_FILE"

    # Phase 1: SIGTERM in reverse order, children first then parent.
    for ((i=${#pids[@]}-1; i>=0; i--)); do
        pid="${pids[i]}"
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "  SIGTERM children of PID $pid (TP workers)"
            pkill -TERM -P "$pid" 2>/dev/null || true
            echo "  SIGTERM PID $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Wait up to 10s for graceful shutdown.
    for _ in $(seq 1 10); do
        alive=0
        for pid in "${pids[@]}"; do
            [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { alive=1; break; }
        done
        [[ $alive -eq 0 ]] && break
        sleep 1
    done

    # Phase 2: SIGKILL stragglers (still children-first, then parent).
    for ((i=${#pids[@]}-1; i>=0; i--)); do
        pid="${pids[i]}"
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "  SIGKILL children of PID $pid"
            pkill -KILL -P "$pid" 2>/dev/null || true
            echo "  SIGKILL PID $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    rm -f "$PID_FILE"
    sleep 2
}
trap cleanup EXIT

# ── 1. Launch LMCache server ───────────────────────────────
echo "=== Launching LMCache server ==="
echo "Port: $LMCACHE_PORT, chunk-size: $CHUNK_SIZE"

lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb 50 \
    --eviction-policy LRU \
    --max-workers 4 \
    > "$LMCACHE_LOG" 2>&1 &

LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache server started (PID=$LMCACHE_PID)"
sleep 10

if ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
    echo "FAIL: LMCache server died during startup"
    tail -100 "$LMCACHE_LOG" 2>/dev/null || true
    exit 1
fi

# ── 2. Launch vLLM with DeepSeek-V4-Flash TP=4 ─────────────
echo "=== Launching vLLM (DeepSeek-V4-Flash, TP=4, dummy weights) ==="
echo "Model: $MODEL"
echo "Port:  $VLLM_PORT"

# vLLM internally calls get_open_port() for torch.distributed; if the env
# var VLLM_PORT is set it would collide with the serving port.
SAVED_VLLM_PORT="$VLLM_PORT"
unset VLLM_PORT

FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_SERVER_DEV_MODE=1 \
vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --load-format dummy \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --port "$SAVED_VLLM_PORT" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 60}}" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
echo "vLLM started (PID=$VLLM_PID)"

VLLM_PORT="$SAVED_VLLM_PORT"

# ── 3. Wait for vLLM to be ready ──────────────────────────
echo "=== Waiting for vLLM to be ready ==="
if ! wait_for_server "$VLLM_PORT" 600 "$VLLM_LOG"; then
    echo "FAIL: vLLM failed to start"
    exit 1
fi

# ── 4. Build a prompt that spans exactly one chunk ────────
# CHUNK_SIZE = 256 tokens. We need a prompt whose tokenised length
# (including special tokens added by vLLM) lands comfortably in
# (CHUNK_SIZE, 2 * CHUNK_SIZE), so exactly one chunk is store/retrieved.
#
# Empirically with DeepSeek-V4-Flash's tokenizer:
#   "Hello world. " * 180  -->  ~350 tokens
# That places us ~94 tokens above the 256 boundary and ~162 tokens below
# the 512 boundary, so the prompt is robust to minor tokenizer updates
# (DeepSeek occasionally refreshes tokenizer.json) while still producing
# exactly one full chunk -- giving a deterministic "1/1 retained keys
# (1 L1, 0 L2)" line on the second request.
PROMPT=$(python3 -c 'print("Hello world. " * 180, end="")')

send_request() {
    local prompt="$1"
    curl -sf "http://localhost:${VLLM_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"${prompt}\",
            \"max_tokens\": 16,
            \"temperature\": 0
        }"
}

validate_response() {
    local label="$1" body="$2"
    if ! echo "$body" | python3 -c '
import sys, json
r = json.load(sys.stdin)
text = r["choices"][0]["text"]
assert text, "empty generation"
print(text[:80])
'; then
        echo "FAIL: ${label} -- invalid response body:"
        echo "$body"
        return 1
    fi
}

# ── 5. First request: cold (store path) ───────────────────
echo "=== Sending request 1 (cold -- store path) ==="
RESP1=$(send_request "$PROMPT") || { echo "FAIL: request 1 curl failed"; exit 1; }
echo -n "Response 1: "
validate_response "request 1 (cold)" "$RESP1" || exit 1

# Snapshot the LMCache log size BEFORE request 2, so we only inspect
# new lines emitted as a result of the second request.
LMCACHE_LINES_BEFORE_REQ2=$(wc -l < "$LMCACHE_LOG")

# ── 6. Second request: warm (retrieve path) ──────────────
echo "=== Sending request 2 (warm -- retrieve path) ==="
RESP2=$(send_request "$PROMPT") || { echo "FAIL: request 2 curl failed"; exit 1; }
echo -n "Response 2: "
validate_response "request 2 (warm)" "$RESP2" || exit 1

# Give the async prefetch logger a moment to flush.
sleep 2

# ── 7. Assert LMCache served the second request from L1 ──
echo "=== Verifying LMCache L1 retrieve happened on request 2 ==="
LMCACHE_DELTA=$(tail -n "+$((LMCACHE_LINES_BEFORE_REQ2 + 1))" "$LMCACHE_LOG")

# Match: "Prefetch request completed (L1+L2): 1/1 retained keys (1 L1, 0 L2) ..."
# Because the prompt is sized to produce exactly one chunk (see §4), we can
# assert the precise hit shape rather than just "L1 hits >= 1". A 0/N miss
# would not be logged at all -- see storage_manager.py: `if total_hits > 0`.
HIT_REGEX='Prefetch request completed \(L1\+L2\): 1/1 retained keys \(1 L1, 0 L2\)'

if ! echo "$LMCACHE_DELTA" | grep -qE "$HIT_REGEX"; then
    echo "FAIL: no L1 cache retrieve detected on the second request"
    echo ""
    echo "--- LMCache log delta (after request 2) ---"
    echo "$LMCACHE_DELTA" | tail -n 100
    echo ""
    echo "--- vLLM log (last 50 lines) ---"
    tail -50 "$VLLM_LOG" 2>/dev/null || true
    exit 1
fi

# Echo the matched line(s) for build log readability.
echo "Matched LMCache hit line(s):"
echo "$LMCACHE_DELTA" | grep -E "$HIT_REGEX" | head -3

echo ""
echo "=== PASS: DeepSeek-V4-Flash MP smoke test ==="

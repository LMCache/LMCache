#!/usr/bin/env bash
# Self-contained deadlock regression test.
#
# Launches a model with LMCache, sends concurrent long-prefill requests, and
# verifies they all complete within 3 minutes. A driver/GIL/transfer deadlock
# would cause requests to hang indefinitely, failing the timeout.
#
# This test is self-contained: it handles its own server lifecycle
# instead of using the standard launch-processes.sh / wait-for-servers.sh.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# ── Configuration ───────────────────────────────────────────
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
LMCACHE_PORT="${LMCACHE_PORT:-15554}"
VLLM_PORT="${VLLM_PORT:-8000}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
TIMEOUT_SECONDS=180   # 3 minutes
TORCH_DEVICE_TYPE="${TORCH_DEVICE_TYPE:-cuda}"
GPU_FOR_VLLM="${GPU_FOR_VLLM:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16000}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-64}"
RANDOM_PREFILL_REQUEST_LEN="${DEADLOCK_REQUEST_LEN:-30000}"
RANDOM_PREFILL_NUM_REQUESTS="${DEADLOCK_NUM_REQUESTS:-50}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-dummy}"

# ── Install py-spy for deadlock diagnosis ──────────────────
echo "=== Installing py-spy ==="
uv pip install py-spy
PY_SPY="$(which py-spy)"
echo "py-spy installed at: $PY_SPY"

PYSPY_LOG="/tmp/build_${BUILD_ID}_pyspy.log"

# ── Helper: dump stacks of server processes via py-spy ─────
dump_stacks() {
    echo "" | tee -a "$PYSPY_LOG"
    echo "=== py-spy stack dump (native + Python) ===" | tee -a "$PYSPY_LOG"

    if kill -0 "$LMCACHE_PID" 2>/dev/null; then
        echo "" | tee -a "$PYSPY_LOG"
        echo "--- LMCache server (PID=$LMCACHE_PID) ---" | tee -a "$PYSPY_LOG"
        sudo "$PY_SPY" dump --pid "$LMCACHE_PID" --native 2>&1 | tee -a "$PYSPY_LOG" || true
    fi

    # Copy to repo root so cleanup.sh collects it as a Buildkite artifact
    cp "$PYSPY_LOG" "${REPO_ROOT}/build_${BUILD_ID}_pyspy.log" 2>/dev/null || true
}

# ── 1. Launch LMCache server ───────────────────────────────
echo "=== Launching LMCache server ==="
echo "Port: $LMCACHE_PORT"

DEVICE_AFFINITY_VAR="CUDA_VISIBLE_DEVICES"
VLLM_DEVICE_ENV=(VLLM_TARGET_DEVICE="cuda")
if [ "$TORCH_DEVICE_TYPE" = "xpu" ]; then
    DEVICE_AFFINITY_VAR="ZE_AFFINITY_MASK"
    VLLM_DEVICE_ENV=(VLLM_TARGET_DEVICE="xpu")
    unset CUDA_VISIBLE_DEVICES || true
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        # shellcheck disable=SC1091
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    fi
fi

L1_LAZY_ARG=""
SHM_NAME_ARG=""
if [ "${L1_USE_LAZY:-true}" = "false" ]; then
    L1_LAZY_ARG="--no-l1-use-lazy"
    if [ "${LMCACHE_MP_TRANSFER_MODE:-}" = "engine_driven" ]; then
        SHM_NAME_ARG="--shm-name ${BUILD_ID}_1"
    fi
fi
TRANSFER_MODE_ARG="--supported-transfer-mode ${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}"

env "${DEVICE_AFFINITY_VAR}=${GPU_FOR_VLLM}" \
    "${VLLM_DEVICE_ENV[@]}" \
lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size 256 \
    --l1-size-gb 50 \
    --eviction-policy LRU \
    --max-workers 2 \
    ${L1_LAZY_ARG} \
    ${SHM_NAME_ARG} \
    ${TRANSFER_MODE_ARG} \
    > "/tmp/build_${BUILD_ID}_lmcache.log" 2>&1 &

LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache server started (PID=$LMCACHE_PID)"
sleep 10

# ── 2. Launch vLLM ─────────────────────────────────────────
echo "=== Launching vLLM ==="
echo "Model: $MODEL"
echo "Port: $VLLM_PORT"

# Save VLLM_PORT before unsetting — vLLM's internal get_open_port()
# would otherwise collide with the serving port for torch.distributed.
SAVED_VLLM_PORT="$VLLM_PORT"
unset VLLM_PORT

ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
ATTENTION_BACKEND_ARG=""
if [ "$ATTENTION_BACKEND" != "auto" ]; then
    ATTENTION_BACKEND_ARG="--attention-backend $ATTENTION_BACKEND"
fi
PREFIX_CACHING_ARG="--enable-prefix-caching"
if [ "${VLLM_DISABLE_PREFIX_CACHING:-false}" = "1" ] || [ "${VLLM_DISABLE_PREFIX_CACHING:-false}" = "true" ]; then
    PREFIX_CACHING_ARG="--no-enable-prefix-caching"
fi
CHUNKED_PREFILL_ARG="--enable-chunked-prefill"
if [ "${VLLM_DISABLE_CHUNKED_PREFILL:-false}" = "1" ] || [ "${VLLM_DISABLE_CHUNKED_PREFILL:-false}" = "true" ]; then
    CHUNKED_PREFILL_ARG="--no-enable-chunked-prefill"
fi
ENFORCE_EAGER_ARG=""
if [ "${ENFORCE_EAGER:-0}" = "1" ] || [ "${ENFORCE_EAGER:-0}" = "true" ]; then
    ENFORCE_EAGER_ARG="--enforce-eager"
fi
MAX_NUM_BATCHED_TOKENS_ARG=""
if [ "$MAX_NUM_BATCHED_TOKENS" != "auto" ]; then
    MAX_NUM_BATCHED_TOKENS_ARG="--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}"
fi
HUGGING_FACE_OVERRIDES_ARG=()
if [ -n "${VLLM_HF_OVERRIDES:-}" ]; then
    HUGGING_FACE_OVERRIDES_ARG=(--hf-overrides "$VLLM_HF_OVERRIDES")
fi
if [ "$TORCH_DEVICE_TYPE" = "xpu" ]; then
    BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
else
    BATCH_INVARIANT="${BATCH_INVARIANT:-1}"
fi

env "${DEVICE_AFFINITY_VAR}=${GPU_FOR_VLLM}" \
    "${VLLM_DEVICE_ENV[@]}" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_SERVER_DEV_MODE=1 \
    VLLM_BATCH_INVARIANT="$BATCH_INVARIANT" \
    PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --block-size "$VLLM_BLOCK_SIZE" \
    --trust-remote-code \
    --load-format "$VLLM_LOAD_FORMAT" \
    "${HUGGING_FACE_OVERRIDES_ARG[@]}" \
    $PREFIX_CACHING_ARG \
    $CHUNKED_PREFILL_ARG \
    --gpu-memory-utilization 0.8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    $MAX_NUM_BATCHED_TOKENS_ARG \
    --scheduling-policy fcfs \
    --port "$SAVED_VLLM_PORT" \
    $ENFORCE_EAGER_ARG \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 60}}" \
    > "/tmp/build_${BUILD_ID}_vllm.log" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
echo "vLLM started (PID=$VLLM_PID)"

VLLM_PORT="$SAVED_VLLM_PORT"

# ── 3. Wait for vLLM to be ready ──────────────────────────
echo "=== Waiting for vLLM to be ready ==="
if ! wait_for_server "$VLLM_PORT" 600; then
    echo "vLLM failed to start. Last 100 lines of log:"
    tail -100 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

# ── 4. Run benchmark with timeout ─────────────────────────
echo "=== Running lmcache bench engine (random-prefill, ${RANDOM_PREFILL_NUM_REQUESTS} reqs, ~${RANDOM_PREFILL_REQUEST_LEN} tokens) ==="
echo "Timeout: ${TIMEOUT_SECONDS}s"

if ! timeout "$TIMEOUT_SECONDS" lmcache bench engine \
        --engine-url "http://localhost:${VLLM_PORT}" \
        --workload random-prefill \
        --tokens-per-gb-kvcache 6000 \
        --rp-request-length "$RANDOM_PREFILL_REQUEST_LEN" \
        --rp-num-requests "$RANDOM_PREFILL_NUM_REQUESTS" \
        --no-interactive \
        --no-csv \
        -q; then
    echo "FAIL: Benchmark failed or timed out (possible deadlock)"
    echo ""
    echo "=== LMCache log (last 50 lines) ==="
    tail -50 "/tmp/build_${BUILD_ID}_lmcache.log" 2>/dev/null || true
    echo ""
    echo "=== vLLM log (last 50 lines) ==="
    tail -50 "/tmp/build_${BUILD_ID}_vllm.log" 2>/dev/null || true
    exit 1
fi

echo ""
echo "=== Benchmark completed within ${TIMEOUT_SECONDS}s ==="
echo "PASS: No deadlock detected"

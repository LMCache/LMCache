#!/usr/bin/env bash
# Launch LMCache MP server, vLLM with LMCache, and vLLM baseline
# as native background processes (no Docker).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration (inherited from run-mp-test.sh)
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
vllm_port="${VLLM_PORT:-8000}"
vllm_baseline_port="${VLLM_BASELINE_PORT:-9000}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-80}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"

# K8s assigns exactly 2 GPUs as devices 0 and 1
GPU_FOR_VLLM=0
GPU_FOR_BASELINE=1
echo "Using GPU $GPU_FOR_VLLM for vLLM with LMCache"
echo "Using GPU $GPU_FOR_BASELINE for vLLM baseline"

# Check GPU memory and set gpu-memory-utilization for very large GPUs.
# Without this, vLLM allocates so much KV cache that APC covers all prefixes
# and LMCache's cache path is never exercised, making the test pass vacuously.
GPU_MEMORY_UTIL_ARG=""
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_FOR_VLLM}" | tr -d ' ')
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
echo "Detected GPU memory: ${GPU_MEMORY_GB}GB (${GPU_MEMORY_MB}MB)"

if [ "$GPU_MEMORY_GB" -gt 90 ]; then
    echo "GPU memory > 90GB, adding --gpu-memory-utilization 0.5"
    GPU_MEMORY_UTIL_ARG="--gpu-memory-utilization 0.5"
fi

# Store PIDs in a file so cleanup.sh can find them
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
> "$PID_FILE"

# ── 1. LMCache Multiprocess Server ──────────────────────────
echo "=== Launching LMCache MP server ==="
echo "Port: $LMCACHE_PORT"

CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
python -m lmcache.v1.multiprocess.server \
    --l1-size-gb "$CPU_BUFFER_SIZE" \
    --eviction-policy LRU \
    --max-workers "$MAX_WORKERS" \
    --port "$LMCACHE_PORT" \
    > "/tmp/build_${BUILD_ID}_lmcache.log" 2>&1 &

LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"

# Wait for LMCache to initialize
echo "Waiting for LMCache to initialize..."
sleep 10

# Unset VLLM_PORT so vLLM's internal get_open_port() picks a random
# ephemeral port for torch.distributed instead of trying serving_port+1.
# Without this, both instances fight over the same internal port.
unset VLLM_PORT

# ── 2. vLLM with LMCache ────────────────────────────────────
echo "=== Launching vLLM with LMCache ==="
echo "Model: $MODEL"
echo "Port: $vllm_port"

CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT}}" \
    --attention-backend FLASH_ATTN \
    --port "$vllm_port" \
    --no-async-scheduling \
    $GPU_MEMORY_UTIL_ARG \
    > "/tmp/build_${BUILD_ID}_vllm.log" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
echo "vLLM with LMCache started (PID=$VLLM_PID)"

# ── 3. vLLM Baseline (without LMCache) ──────────────────────
echo "=== Launching vLLM baseline ==="
echo "Port: $vllm_baseline_port"

CUDA_VISIBLE_DEVICES="${GPU_FOR_BASELINE}" \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --attention-backend FLASH_ATTN \
    --port "$vllm_baseline_port" \
    --no-async-scheduling \
    $GPU_MEMORY_UTIL_ARG \
    > "/tmp/build_${BUILD_ID}_vllm_baseline.log" 2>&1 &

VLLM_BASELINE_PID=$!
echo "$VLLM_BASELINE_PID" >> "$PID_FILE"
echo "vLLM baseline started (PID=$VLLM_BASELINE_PID)"

echo "=== All processes launched ==="
echo "PIDs: LMCache=$LMCACHE_PID, vLLM=$VLLM_PID, Baseline=$VLLM_BASELINE_PID"

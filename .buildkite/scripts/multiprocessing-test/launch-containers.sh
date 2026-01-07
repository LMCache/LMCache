#!/bin/bash
# Launch LMCache and vLLM containers for multiprocessing tests
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Container names (exported so cleanup script can use them)
export LMCACHE_CONTAINER_NAME="${LMCACHE_CONTAINER_NAME:-lmcache-mp-test}"
export VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-mp-test}"

# Configuration
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
VLLM_PORT="${VLLM_PORT:-8000}"
CPU_BUFFER_SIZE="${CPU_BUFFER_SIZE:-50}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"

echo "=== Launching LMCache container ==="
echo "Container name: $LMCACHE_CONTAINER_NAME"
echo "Port: $LMCACHE_PORT"

docker run -d \
    --name "$LMCACHE_CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --network host \
    --ipc host \
    --entrypoint /opt/venv/bin/python3 \
    lmcache/vllm-openai:test \
    -m lmcache.v1.multiprocess.server \
    --cpu-buffer-size "$CPU_BUFFER_SIZE" \
    --max-workers "$MAX_WORKERS" \
    --port "$LMCACHE_PORT"

echo "LMCache container started"

# Wait a bit for LMCache to initialize
echo "Waiting for LMCache to initialize..."
sleep 10

echo "=== Launching vLLM container ==="
echo "Container name: $VLLM_CONTAINER_NAME"
echo "Model: $MODEL"

docker run -d \
    --name "$VLLM_CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --volume ~/.cache/huggingface:/root/.cache/huggingface \
    --network host \
    --ipc host \
    --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    --env VLLM_BATCH_INVARIANT=1 \
    --env PYTHONHASHSEED=0 \
    lmcache/vllm-openai:test \
    "$MODEL" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT}}" \
    --no-async-scheduling

echo "vLLM container started"

echo "=== Containers launched ==="
docker ps --filter "name=$LMCACHE_CONTAINER_NAME" --filter "name=$VLLM_CONTAINER_NAME"


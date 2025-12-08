#!/bin/bash
# Test vLLM with LMCache GDS backend - Small buffer to trigger cache full

set -e

echo "=== Activating virtual environment ==="
source /home/jpzeng/test/LMCache/.venv/bin/activate

echo "=== Cleaning GDS cache directory ==="
rm -rf /tmp/lmcache_gds_small
mkdir -p /tmp/lmcache_gds_small

LOG_FILE="/tmp/lmcache_gds_small/lmcache_gds_test.log"
echo "=== Starting vLLM with LMCache (GDS backend, SMALL buffer: 8MB, max_gds_size: 0.01GB) ==="
echo "=== Logs will be written to: $LOG_FILE ==="
echo "=== Use 'tail -f $LOG_FILE' in another terminal to watch logs ==="
echo "=== Or use 'grep GDS $LOG_FILE' to filter GDS cache logs ==="

# LMCACHE_CHUNK_SIZE=256 \
# # LMCACHE_STORAGE_BACKEND=gds \
# LMCACHE_GDS_PATH="/tmp/lmcache_gds_small" \
# LMCACHE_LOCAL_CPU=False \

# # LMCACHE_STORAGE_BACKEND=loacl \
# # LMCACHE_GDS_PATH=/tmp/lmcache_local \
# LMCACHE_CUFILE_BUFFER_SIZE=8 \
# LMCACHE_MAX_GDS_SIZE=0.01 \
# LMCACHE_CACHE_POLICY=LRU \
# LMCACHE_EXTRA_CONFIG='{"use_cufile": false}' \
# LMCACHE_LOG_LEVEL=DEBUG \
# vllm serve Qwen/Qwen2.5-0.5B-Instruct \
#   --max-model-len 2048 \
#   --gpu-memory-utilization 0.7 \
#   --kv-offloading-backend lmcache \
#   --kv-offloading-size 1 \
#   --disable-hybrid-kv-cache-manager \
#   --port 8000 2>&1 | tee $LOG_FILE



CUDA_VISIBLE_DEVICES=1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_GDS_PATH="/tmp/lmcache_gds_small" \
LMCACHE_CUFILE_BUFFER_SIZE=8 \
LMCACHE_MAX_GDS_SIZE=0.01 \
LMCACHE_CACHE_POLICY=LRU \
LMCACHE_EXTRA_CONFIG='{"use_cufile": false}' \
LMCACHE_LOG_LEVEL=DEBUG \
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 1 \
  --disable-hybrid-kv-cache-manager \
  --port 8000 2>&1 | tee $LOG_FILE
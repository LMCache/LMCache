#!/bin/bash
set -e

cd /root/learning/LMCache
source .venv/bin/activate

MODEL="Qwen/Qwen3-4B-Instruct-2507"
GPU_ID=0
PORT=8000

echo "=========================================="
echo "Test 1: WITHOUT LMCache (Baseline)"
echo "=========================================="

# Start vLLM server without LMCache
echo "Starting vLLM server without LMCache..."
CUDA_VISIBLE_DEVICES=$GPU_ID \
vllm serve $MODEL \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port $PORT \
  --no-enable-prefix-caching > /tmp/vllm_no_lmcache.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "Server failed to start"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
done

# Run benchmark
echo "Running benchmark without LMCache..."
vllm bench serve --model $MODEL \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 20 \
  --request-rate 0.5 \
  --max-concurrency 1 2>&1 | tee /tmp/benchmark_no_lmcache.log

# Stop server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 5

echo ""
echo "=========================================="
echo "Test 2: WITH LMCache"
echo "=========================================="

# Start vLLM server with LMCache
echo "Starting vLLM server with LMCache..."
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=recipes/dense_instruct_cpu_hot_cache.yaml \
CUDA_VISIBLE_DEVICES=$GPU_ID \
vllm serve $MODEL \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port $PORT \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' > /tmp/vllm_with_lmcache.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "Server failed to start"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
done

# Run benchmark
echo "Running benchmark with LMCache..."
vllm bench serve --model $MODEL \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 20 \
  --request-rate 0.5 \
  --max-concurrency 1 2>&1 | tee /tmp/benchmark_with_lmcache.log

# Stop server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results without LMCache:"
grep -E "(Mean TTFT|Median TTFT|P99 TTFT)" /tmp/benchmark_no_lmcache.log || true
echo ""
echo "Results with LMCache:"
grep -E "(Mean TTFT|Median TTFT|P99 TTFT)" /tmp/benchmark_with_lmcache.log || true

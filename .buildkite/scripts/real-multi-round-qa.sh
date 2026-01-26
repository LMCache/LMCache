#!/bin/bash

set -ex

MODEL="meta-llama/Llama-3.2-1B-Instruct"

# Clone LMBenchmark repository
git clone --depth 1 -b multiround https://github.com/ningziwen/LMBenchmark.git /tmp/LMBenchmark

# Install requirements
pip install -r /tmp/LMBenchmark/real-multi-round-qa/requirements.txt

# Download novel data (64k)
python3 /tmp/LMBenchmark/real-multi-round-qa/prepare.py \
    --output /tmp/novels \
    --model $MODEL \
    --start 0 \
    --end 100

# Start the vLLM server with LMCache
LMCACHE_TRACK_USAGE="false" lmcache_vllm serve $MODEL \
    --disable-log-requests > lmcache_vllm.log 2>&1 &

echo "Waiting for service to start..."
timeout=120
elapsed=0
until grep -q "Uvicorn running on" lmcache_vllm.log; do
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout: Service did not start within $timeout seconds."
        cat lmcache_vllm.log
        exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
    echo "Waiting... ($elapsed seconds elapsed)"
done
echo "Service started successfully."

# Run the real-multi-round-qa benchmark (64k)
python3 /tmp/LMBenchmark/real-multi-round-qa/multi-round-qa.py \
    --num-users 5 \
    --num-rounds 5 \
    --src-dir /tmp/novels/64k \
    --answer-len 512 \
    --model $MODEL \
    --base-url http://localhost:8000 \
    --timeout 300 \
    --time 600 \
    --output real-multi-round-qa-64k-results.json

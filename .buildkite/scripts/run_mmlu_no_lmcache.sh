#!/bin/bash
set -euxo pipefail

# Activate virtual environment
source mmlu_venv/bin/activate

MODEL=deepseek-ai/DeepSeek-V2-Lite
PORT=8000

VLLM_MLA_DISABLE=0 \
VLLM_USE_V1=0 \
python3 -m vllm.entrypoints.api_server \
  --model $MODEL \
  --trust-remote-code \
  --served-model-name deepseek_test \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 2 &
SERVER_PID=$!
sleep 15

python3 sglang/benchmark/mmlu/bench_other.py \
  --backend vllm \
  --host http://localhost \
  --port $PORT \
  --parallel 16 \
  > mmlu-results/v0_deepseek2.txt || true

kill $SERVER_PID || true

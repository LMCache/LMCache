#!/bin/bash
set -euxo pipefail

eval "$(conda shell.bash hook)"
conda activate buildkite

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
sleep 45

python3 .buildkite/mmlu_scripts/mmlu_bench.py \
  --nsub 60 \
  --parallel 16 \
  > mmlu-results/v0_deepseek2.txt || true

kill $SERVER_PID || true

#!/bin/bash

rm -rf ../lmcache-vllm
git clone https://github.com/LMCache/lmcache-vllm.git ../lmcache-vllm
cd ../lmcache-vllm
pip install .
cd ../multi-round-qa
pip install -r ./benchmarks/requirements.txt
lmcache_vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests &
sleep 100
python3 benchmarks/multi-round-qa.py \
    --num-users 10 \
    --num-rounds 5 \
    --qps 0.5 \
    --shared-system-prompt 1000 \
    --user-history-prompt 2000 \
    --answer-len 100 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --base-url http://localhost:8000/v1 \
    --time 300
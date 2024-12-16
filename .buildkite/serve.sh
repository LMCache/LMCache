#!/bin/bash

rm -rf ../lmcache-vllm
git clone https://github.com/LMCache/lmcache-vllm.git ../lmcache-vllm
cd ../lmcache-vllm
pip install .
cd ../multi-round-qa
pip install -r ./benchmarks/requirements.txt
lmcache_vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests

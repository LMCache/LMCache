#!/bin/bash

pip install -r ./benchmarks/requirements.txt
lmcache_vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests

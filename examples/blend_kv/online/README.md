# KV blending online example
This is a minimal online example demonstrating the KV blending functionality of LMCache.

## How to run
### Start api server
TP = 1  
```
LMCACHE_CONFIG_FILE=../example_blending.yaml python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000
```
TP = x, x > 1  
```
LMCACHE_CONFIG_FILE=../example_blending.yaml VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --tensor-parallel-size x
```
(Add VLLM_WORKER_MULTIPROC_METHOD=spawn and --tensor-parallel-size)  
### Send requests
```
python3 online_blend.py 8000
```

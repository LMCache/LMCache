# P2P KV Cache Sharing
This is an example to demonstrate P2P KV cache sharing.
## Prerequisites
Your server should have at least 2 GPUs.  

This will use the port 8000 and 8001 for 2 vllms,
And will use port 8200 and 8201 for 2 distributed cache servers,
And will use port 8100 for lookup server.
## Steps
1. Pull redis docker and start lookup server at port 8100:
```bash
docker pull redis
docker run --name some-redis -d -p 8100:6379 redis
``` 

2. Start two vllm engines:

Start vllm engine 1 at port 8000:
```bash
CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```
Start vllm engine 2 at port 8001:
```bash
CUDA_VISIBLE_DEVICES=0=1 LMCACHE_CONFIG_FILE=example2.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8001 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'  
```
Note that the two distributed cache servers will start at port 8200 and 8201.


3. Send request to vllm engine 1:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10
  }'
```

4. Send request to vllm engine 2:  
```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10
  }'
```
The cache will be automatically retrieved from vllm engine 1.

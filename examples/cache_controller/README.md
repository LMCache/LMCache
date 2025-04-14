# LMCache Controller (lookup as an example)
This is an example to demonstrate how to check the existence of a request's KV cache in an LMCacheEngine externally.

## Prerequisites
Your server should have at least 1 GPU.  

This will use port 8000 for 1 vllm and port 8001 for LMCache controller.

## Steps
1. Start the vllm engine at port 8000:

```bash
CUDA_VISIBLE_DEVICES=0 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}'
```

2. Start the cache controller at port 8001:

```bash
lmcache_controller localhost 8001
```

3. Send a request to vllm engine:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10
  }'
```


4. Send a lookup request to lmcache controller:  
```bash
curl -X POST http://localhost:8001/lookup \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_default_instance",
    "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13]
  }'
```
The above request queries how many tokens have been stored in LMCache. Note that we only support using `tokens` as input for now.

You should be able to see a return message:

```plaintext
{"0":{"res":12}}
```

`12` indicates 12 tokens are stored in LMCache.



curl -X POST http://localhost:8001/clear \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_default_instance"
  }'
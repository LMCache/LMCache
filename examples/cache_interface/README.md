# User Controllable Caching
This is an example to demonstrate user controllable caching (e.g., specify whether to cache a request or not).
## Prerequisites
Your server should have at least 1 GPU.  

This will use the port 8000 for 1 vllm.

## Steps
1. Start the vllm engine at port 8000:

```bash
CUDA_VISIBLE_DEVICES=0 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}'
```



3. Send a request to vllm engine with `store_cache: True`:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10,
    "store_cache": True,
  }'
```

You should be able to see logs indicating the KV cache is stored:

```plaintext
DEBUG LMCache: Store skips 0 tokens and then stores 13 tokens [2025-03-02 21:58:55,147]
```

4. Send request to vllm engine 2:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "What's the weather today in Chicago?",
    "max_tokens": 10,
    "store_cache": False,
  }'
```

You should be able to see logs indicating the KV cache is NOT stored:

```plaintext
DEBUG LMCache: User has specified not to store the cache [2025-03-02 21:54:58,380]
```

Note that cache is stored by default.

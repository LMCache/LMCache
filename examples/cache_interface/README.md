# User Controllable Caching
This is an example to demonstrate user controllable caching (e.g., specify whether to cache a request or not).
## Prerequisites
Your server should have at least 1 GPU.  

This will use the port 8000 for 1 vllm.

## Steps
1. Start the vllm engine at port 8000:

```bash
CUDA_VISIBLE_DEVICES=0 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```



3. Send a request to vllm engine with `lmcache.skip_save: false` to store the KV cache:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Explain the significance of KV cache in language models.",
    "max_tokens": 10,
    "kv_transfer_params": {
      "lmcache.skip_save": false,
    }
  }'
```

You should be able to see logs indicating the KV cache is stored:

```plaintext
[2025-11-18 09:11:17,227] LMCache INFO: Storing KV cache for 8 out of 8 tokens (skip_leading_tokens=0) for request cmpl-e184aa1d5d884d5d9a36abb6afcb198e-0 (vllm_v1_adapter.py:1094:lmcache.integration.vllm.vllm_v1_adapter)
```

4. Send request to vllm engine with `lmcache.skip_save: true` to skip storing the KV cache:
```bash
curl POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "What's the weather today in Chicago?",
    "max_tokens": 10,
    "kv_transfer_params": {
      "lmcache.skip_save": true,
    }
  }'
```

If the KV cache is not stored, you will not see any storing logs.

Note that cache is stored by default.

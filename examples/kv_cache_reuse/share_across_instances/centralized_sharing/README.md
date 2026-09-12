# Sharing KV cache across multiple vLLM instances
This shows how to share KV across different vLLM instances using LMCache.  
## Prerequisites
Your server should have at least 2 GPUs.  

This will use the port 8000 and 8001 (for vLLM) and port 65432 (for LMCache).  

**Important**: For centralized cache sharing (which is cross-process cases), ensure all processes use the same `PYTHONHASHSEED` to keep the hash of the KV cache consistent across processes.:
```bash
export PYTHONHASHSEED=0
```

## Steps
1.  Start the lmcache centralized server,
```bash
lmcache_server localhost 65432
```  
2. In a different terminal,  
```bash
export PYTHONHASHSEED=0

LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```  
In another terminal,   
```bash
export PYTHONHASHSEED=0

LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=1 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8001 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```  
Wait until both of the engines are ready.

3.  Send one request to the engine at port 8000.
    Repeat the sentence so the prompt exceeds ``chunk_size`` (256 in
    `example.yaml`). A one-sentence prompt is only ~12 tokens and is
    not stored unless ``save_unfull_chunk: true`` is set, so the second
    engine would miss.

    The ``model`` field must match the served model.
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"mistralai/Mistral-7B-Instruct-v0.2\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```
4. Send the same request to the engine at port 8001,
```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"mistralai/Mistral-7B-Instruct-v0.2\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```

The second engine should log a non-zero ``LMCache hit tokens`` value.
If you see ``LMCache hit tokens: 0``, the prompt did not fill a chunk
or ``PYTHONHASHSEED`` differs across processes.
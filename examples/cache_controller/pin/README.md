# LMCache Controller (pin as an example)
This is an example to demonstrate how to pin a request's KV cache in an LMCacheEngine externally so that the cache can only be evicted when it expires.

## Prerequisites
Your server should have at least 1 GPU. 

## Steps
1. Start the lmcache controller at port 9000 and the monitor at port 9001:

```bash
lmcache_controller --host localhost --port 9000 --monitor-port 9001
```

2. Start the vllm engine at port 8000:

```bash
CUDA_VISIBLE_DEVICES=0 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}'
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


4. Send a pin request to lmcache controller:  
```bash
curl -X POST http://localhost:9000/pin\
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_default_instance",
    "tokens": [128000, 849, 21435, 279, 26431, 315, 85748, 6636, 304, 4221, 4211, 13],
  }'
```
The above request pins the KV cache such that it won't be evicted.

You should be able to see a return message indicating the sucess of pinning:

```plaintext
{"res": true}
```
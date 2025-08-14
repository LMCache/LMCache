# Cache by request
This is an example to cache by request, use the `kv_transfer_params.caching` field to control whether to cache this request.

## Prerequisites
Your server should have at least 1 GPU.

This will use the port 8000 for 1 vllm.

## Steps
1. Start the vllm engine at port 8000:

```bash
VLLM_USE_V1=1 \
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_TRACK_USAGE=false \
LMCACHE_CONFIG_FILE=example.yaml \
vllm serve /disc/f/models/opt-125m/ \
           --served-model-name "facebook/opt-125m" \
           --enforce-eager  \
           --port 8000 \
           --gpu-memory-utilization 0.8 \
           --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
           --trust-remote-code
```

3. Send a request to vllm engine with `kv_transfer_params: {caching: false}`:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Explain the significance of KV cache in language models." * 100,
    "max_tokens": 10,
	"kv_transfer_params": {
	  "caching": false
	}
  }'
```

This request will not be cached.

4. Send a request to vllm engine with `kv_transfer_params: {caching: true}` or not pass the param:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Explain the significance of KV cache in language models." * 100,
    "max_tokens": 10,
	"kv_transfer_params": {
	  "caching": true
	}
  }'
```
This request will be cached.

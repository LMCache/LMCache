# Cache by tags
This is an example to cache by tags, use the `kv_transfer_params.user` field to pass the `user` tag, which is enable for user-isolated caching.
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

3. Send a request to vllm engine with `kv_transfer_params: {user: example_user_1}`:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Explain the significance of KV cache in language models." * 100,
    "max_tokens": 10,
	"kv_transfer_params": {
	  "user": "example_user_1"
	}
  }'
```

You should be able to see logs `Retrieved xxx out of xxx out of total xxx tokens` at the second time use the same input,
but if you change the `user` field, the first time will not hit the cache.

```plaintext
LMCache INFO: Retrieved 512 out of 512 out of total 512 tokens
```

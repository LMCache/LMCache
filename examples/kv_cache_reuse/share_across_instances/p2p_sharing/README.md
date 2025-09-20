# P2P KV Cache Sharing
This is an example to demonstrate P2P KV cache sharing.
## Prerequisites
Your server should have at least 2 GPUs.
[NIXL](https://github.com/ai-dynamo/nixl) should be installed as well. 

This will use the port 8010 and 8011 for 2 vllms,
And will use port 8200 and 8201 for 2 distributed cache servers,
And will use port 8100 for lookup server.
## Steps
1. Pull redis docker and start lookup server at port 8100:
```bash
PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'
``` 

2. Start two vllm engines:

Start vllm engine 1 at port 8010:
```bash
PYTHONHASHSEED=123 CUCX_TLS=rc CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8 --port 8010 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```
Start vllm engine 2 at port 8011:
```bash
PYTHONHASHSEED=123 CUCX_TLS=rc CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=example2.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct  --gpu-memory-utilization 0.8 --port 8011 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'  
```
Note that the two p2p initialization ports will start at port 8200 and 8201.


3. Send request to vllm engine 1:  
```bash
curl -X POST http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```

4. Send request to vllm engine 2:  
```bash
curl -X POST http://localhost:8011/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```
The cache will be automatically retrieved from vllm engine 1.

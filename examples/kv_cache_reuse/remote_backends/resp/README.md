# LMCacheRedis KV Connector

The cpp binding is in `csrc/redis/*`.

The key optimization is multi-threading and batching on the C layer and the python side being awoken through the eventfd API where a callback consumes the completion belonging to a non-blocking submission. 

An GET / SET benchmark is in `examples/kv_cache_reuse/remote_backends/resp/benchmark_resp_client.py`. The python client lives inside of `lmcache/v1/storage_backend/resp_client.py`.

Install lmcache from source, then run a sanity check:

```bash
python benchmark_resp_client.py
```

## Quickstart

Start up redis with multiple io threads: 
```bash
git clone https://github.com/redis/redis.git
cd redis
git checkout 8.2
make -j
./src/redis-server --protected-mode no --save '' --appendonly no --io-threads 4
```

Clear the state between queries
```bash
sudo apt install redis-cli
redis-cli -p 6379 FLUSHALL
redis-cli -p 6379 DBSIZE
```

Deploy LMCache with the custom LMCacheRedis KV Connector
```bash
LMCACHE_CONFIG_FILE=resp.yaml \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
    --disable-log-requests --no-enable-prefix-caching \
    --load-format dummy
```

Coming Soon: 
MP Mode Controller

Send twice. First time for store. Second time for retrieve. 
```bash
curl -X POST http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "'"$(printf 'abElaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
    "max_tokens": 10
  }'
```
lmcache could use [mooncakestore](https://github.com/kvcache-ai/Mooncake) as a backend storage.

Mooncakestore is a memory storage which support RDMA and TCP. lmcache's mooncakestore connector can use TCP/RDMA transport for now.

This is a simple instruction how to use lmcache with mooncakestore

# install mooncakestore

1. build mooncake and start mooncake store and dependent services.

Please follow [build guild](https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md)

2. install mooncakestore mooncake_vllm_adaptor 

For now, you have to build the mooncake from source follow [build guild](https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md).
But mooncake will supply pip install for nearly future.

# start mooncake store and lmcache

1. start mooncake store

Reference [mooncake guide](https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/vllm-integration-v1.md#run-example)

For example.

```
mooncake_master -v=1 -port=50051
```

2. start vllm with lmcache connector
```
VLLM_USE_V1=0 \
MOONCAKE_CONFIG_PATH=./mooncake.json \
LMCACHE_USE_EXPERIMENTAL=True LMCACHE_TRACK_USAGE=false \
LMCACHE_CHUNK_SIZE=16 LMCACHE_LOCAL_CPU=False LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 \
LMCACHE_REMOTE_URL=mooncakestore://localhost:50051/ \
LMCACHE_REMOTE_SERDE="cachegen" \
vllm serve /disc/f/models/opt-125m/ \
           --served-model-name "facebook/opt-125m" \
           --enforce-eager  \
           --port 8000 \
           --gpu-memory-utilization 0.8 \
           --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}' \
           --trust-remote-code
```

The `./mooncake.json` can reference [mooncake store config](https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/vllm-integration-v1.md#configuration)

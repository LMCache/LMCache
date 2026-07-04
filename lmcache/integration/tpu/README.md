# LMCache TPU integration

Enables LMCache as a host-side KV storage tier behind the vLLM
[tpu-inference](https://github.com/vllm-project/tpu-inference) `TPUOffloadConnector`.

## Model

Unlike the CUDA / XPU / HPU GPU connectors, this path does **not** move torch
tensors out of device memory. On TPU, KV lives as JAX arrays in an XLA mesh, and
`TPUOffloadConnector` already owns the HBM↔host movement. It hands LMCache a flat,
host-resident **byte buffer** per KV block (serialized bit-exactly by the
connector's value bridge) plus a stable string key.

`LMCacheStorageKVStore` exposes the minimal contract the connector needs:

```python
put(key: str, data: bytes) -> None
get(key: str) -> bytes | None
remove(key: str) -> None
contains(key: str) -> bool
```

and persists through LMCache machinery:
- content-addressed `CacheEngineKey` (`model@world@worker@hash@dtype`),
- disk serialization with LRU disk-usage accounting.

This module has **no** `torch_xla` / JAX dependency and runs on any host.

## Usage (from tpu-inference)

```bash
export TPU_OFFLOAD_LMCACHE=1
export TPU_OFFLOAD_LMCACHE_BACKEND=lmcache
export TPU_OFFLOAD_LMCACHE_PATH=/mnt/kvcache
vllm serve <model> --kv-transfer-config '{"kv_connector":"TPUOffloadConnector", ...}'
```

`tpu-inference`'s `host_backend_factory` imports
`lmcache.integration.tpu.lmcache_storage_kv_store.LMCacheStorageKVStore` lazily,
so LMCache is only required when `TPU_OFFLOAD_LMCACHE_BACKEND=lmcache`.

## Roadmap

The current backend provides content-addressed **disk** persistence. The next
step wires the full `StorageManager` so TPU serving can use LMCache's remote /
distributed tiers (Redis, Mooncake, NIXL, P2P) for **cross-instance prefix
sharing** across a TPU replica fleet — the primary payoff for large-scale RL
rollout and multi-replica serving.

## Tests

`tests/v1/integration/tpu/test_lmcache_storage_kv_store.py` (host-only, no
`torch_xla` / JAX / transformers). Run with:

```bash
python -m pytest tests/v1/integration/tpu/ -q --noconftest
```

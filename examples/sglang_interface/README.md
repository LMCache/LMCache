# LMCache API for SGLang

This module provides an integration between the LMCache KV-cache storage engine and SGLang, enabling efficient key-value cache storage, retrieval, and reuse for large language model inference.

The system supports:

- Chunk-based KV cache storage and retrieval
- Prefix-aware continuation with hash chaining
- Hash-based block lookup
- Batch storage and retrieval with status tracking

## Installation

Ensure the following dependencies are installed:

- `torch`
- `lmcache`
- `sglang`


## API Review

This section introduces the key APIs provided by `lmcache.integration.sglang.sglang_adapter` for interacting with LMCache in SGLang. Each API is accompanied by usage examples and parameter descriptions.

---

### `init_lmcache_engine`

Initializes an LMCache engine instance for use in SGLang.

```python
init_lmcache_engine(model_config, rank, world_size, tensor_parallel_size=1)
```

**Parameters:**

- `model_config` (`ModelConfig`): The model configuration from SGLang.
- `rank` (`int`): Rank of the current process.
- `world_size` (`int`): Total number of distributed processes.
- `tensor_parallel_size` (`int`, optional): Size of tensor parallel group (default: `1`).
- `--dram_connector_version`(`int`, optional): Data Layout of cache memory. With `--dram_connector_version 1`, the data layout is each chunk is `layer, chunksize, head_num, head_dim` while with `--dram_connector_version 1`, the data layout is each chunk is `chunksize, layer, head_num, head_dim`

**Returns:** `LMCacheEngine` or `None` if already initialized.

---
### `get_hash`
Generates hash keys for a given token sequence. These can later be used to store or retrieve cached KV blocks.
```python
get_hash(engine, token_ids, mask=None, prefix_hash=None)
```
**Parameters:**

- `engine` (`LMCacheEngine`): The cache engine.
- `token_ids` (`torch.Tensor`): 1D tensor of token IDs.
- `mask` (`torch.Tensor`, optional): Optional mask.
- `prefix_hash` (`CacheEngineKey`, optional): If continuing from a prefix.

**Returns:** `List[CacheEngineKey]`

---
### `lmcache_store_kv`
Stores KV cache blocks associated with a token sequence in LMCache.
```python
lmcache_store_kv(engine, token_ids, kv_caches, prefix_hash=None)
```
**Parameters:**

- `engine` (`LMCacheEngine`): Cache engine instance.
- `token_ids` (`torch.Tensor`): 1D tensor of token IDs.
- `kv_caches` (`torch.Tensor` or `List[torch.Tensor]`): KV tensors to store.
- `prefix_hash` (`CacheEngineKey`, optional): Prefix hash for continuation.

**Returns:** `Tuple(bool, List[CacheEngineKey])`. The `bool` stands for the success of the operation and `List[CacheEngineKey]` is the hash prefix.



---
### `lmcache_store_kv_hash`
Stores KV blocks directly using precomputed hash keys.
```
lmcache_store_kv_hash(engine, hash_, kv_caches)
```
**Parameters:**

- `engine` (`LMCacheEngine`): Cache engine instance.
- `hash_` (`List[CacheEngineKey]`): Precomputed hash keys.
- `kv_caches` (`List[torch.Tensor]`): KV tensors to store.

**Returns:** `Tuple(bool, List[CacheEngineKey])`

---
### `lmcache_retrieve_kv`
Retrieves KV cache blocks based on token IDs.
```
lmcache_retrieve_kv(engine, token_ids, kv_caches,  prefix_hash=None)
```
**Parameters:**

- `engine` (`LMCacheEngine`): Cache engine.
- `token_ids` (`torch.Tensor`): 1D tensor of token IDs.
- `kv_caches` (`torch.Tensor` or `List[torch.Tensor]`): Buffer to fill.
- `prefix_hash` (`CacheEngineKey`, optional): Prefix hash for continuation.

**Returns:** `bool` stands for the success


---
### `lmcache_retrieve_kv_hash`
Retrieves KV blocks using precomputed hash keys.
```
lmcache_retrieve_kv_hash(engine, hash_, kv_caches, retrieve_status)
```
**Parameters:**

- `engine` (`LMCacheEngine`): Cache engine.
- `hash_` (`List[CacheEngineKey]`): Hash keys for retrieval.
- `kv_caches` (`torch.Tensor` or `List[torch.Tensor]`): Buffer to fill.

**Returns:** `bool` stands for the success

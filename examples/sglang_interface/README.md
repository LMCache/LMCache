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

```bash
pip install torch lmcache sglang
```

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
### `Status`
Enums used to track chunk-level store and retrieve operation states.

```python
class StoreStatus(IntEnum):
    FAIL = -1
    PREFILLING = 0

class RetrieveStatus(IntEnum):
    FAIL = -1
    PREFILLING = 0

```
These enums are used to initialize and check store_status and retrieve_status tensors or lists during operations. Notice: For SGLang integration, each status is mapped to one block/chunk.

---
### `lmcache_store_kv`
Stores KV cache blocks associated with a token sequence in LMCache.
```python
lmcache_store_kv(engine, token_ids, kv_caches, store_status, prefix_hash=None)
```
**Parameters:**

- `engine` (`LMCacheEngine`): Cache engine instance.
- `token_ids` (`torch.Tensor`): 1D tensor of token IDs.
- `kv_caches` (`torch.Tensor` or `List[torch.Tensor]`): KV tensors to store.
- `store_status` (`List[StoreStatus]`): Status list for each chunk.
- `prefix_hash` (`CacheEngineKey`, optional): Prefix hash for continuation.

**Returns:** `List[CacheEngineKey]` used for storage.



---
### `lmcache_store_kv_hash`
Stores KV blocks directly using precomputed hash keys.
```
lmcache_store_kv_hash(engine, hash_, kv_caches, store_status)
```
**Parameters:**

- `engine` (`LMCacheEngine`): Cache engine instance.
- `hash_` (`List[CacheEngineKey]`): Precomputed hash keys.
- `kv_caches` (`List[torch.Tensor]`): KV tensors to store.
- `store_status` (`List[StoreStatus]`): Storage status for each chunk.

**Returns:** `List[str]` of stored hash key strings.


---
### `lmcache_retrieve_kv`
Retrieves KV cache blocks based on token IDs.
```
lmcache_retrieve_kv(engine, token_ids, kv_caches, retrieve_status, prefix_hash=None)
```
**Parameters:**

- `engine` (`LMCacheEngine`): Cache engine.
- `token_ids` (`torch.Tensor`): 1D tensor of token IDs.
- `kv_caches` (`torch.Tensor` or `List[torch.Tensor]`): Buffer to fill.
- `retrieve_status` (`List[RetrieveStatus]`): Status list for each chunk.
- `prefix_hash` (`CacheEngineKey`, optional): Prefix hash for continuation.

**Returns:** None


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
- `retrieve_status` (`List[RetrieveStatus]`): Retrieval status per chunk.

**Returns:** None

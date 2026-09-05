## Add hidden state store/retrieve alongside KV cache

### Problem

When LMCache restores KV cache from CPU, the serving engine skips prefill for cached tokens. However, some downstream consumers (e.g., multi-stage pipelines, cross-attention layers) need the **full-sequence hidden states** from the prefill, not just KV. Without HS restore, these consumers receive partial hidden states and produce incorrect output.

### Solution

Add general-purpose hidden state store/retrieve to LMCache, using the same chunk-based token hash mechanism as KV:

- **`CacheEngineKey.to_hs_key()`** — derives a parallel key with `:hs` model_name suffix so HS entries don't collide with KV entries but share the same `chunk_hash`
- **`MemoryFormat.HS_TD`** — new memory format for `[num_tokens, hidden_dim]` tensors, routed through the pinned memory allocator like existing KV formats
- **`LMCacheEngine.store_hidden_states()`** — stores HS chunks to CPU storage using the same `token_database.process_tokens()` chunking as KV (guarantees identical chunk boundaries)
- **`LMCacheEngine.retrieve_hidden_states()`** — retrieves HS chunks and assembles into a contiguous CPU tensor
- **`_enable_hs_offload` config flag** — opt-in via `kv_connector_extra_config["lmcache.enable_hidden_state_offload"]` so vanilla vLLM is completely unaffected

### Key Design Decisions

- **HS bypasses GPU connector entirely** — HS tensors are already on CPU, so no `multi_layer_kv_transfer` or CUDA kernel needed
- **Parallel keys, not modified MemoryObj** — HS uses separate `CacheEngineKey` entries (`:hs` suffix) rather than extending the KV MemoryObj format, keeping backward compatibility
- **Same chunk boundaries as KV** — `store_hidden_states` reuses `token_database.process_tokens()`, so HS and KV chunks are always aligned
- **Opt-in flag** — `_enable_hs_offload` defaults to `False`. No impact on existing users

### Changes

| File | Change |
|---|---|
| `lmcache/utils.py` | `CacheEngineKey.to_hs_key()` method |
| `lmcache/v1/memory_management.py` | `MemoryFormat.HS_TD` enum + allocator/free support |
| `lmcache/v1/cache_engine.py` | `store_hidden_states()` and `retrieve_hidden_states()` |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | `_enable_hs_offload` config flag |
| `tests/v1/test_hidden_state_offload.py` | Unit tests for key derivation and store/retrieve roundtrip |

### Related

- Companion PR: vllm-project/vllm-omni#2530 (wires store/restore from the model runner side)

### Testing

- Unit tests: key derivation (distinct, idempotent, preserves configs, no KV collision), store/retrieve roundtrip (single chunk, multi-chunk), missing returns None
- E2E verified with vllm-omni (Qwen3-Omni-30B, 2×A100, 20K input tokens): both `prefix_caching=true` and `prefix_caching=false` paths produce correct full-sequence HS after KV restore

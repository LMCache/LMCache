# ActivationStore

`ActivationStore` is an LMCache-internal store dedicated to caching per-token
activation tensors next to KV chunks. It generalizes the hidden-state store
(PR #3221): the same machinery caches **any** per-token activation -- hidden
states, query (Q) projections, K/V, MLP intermediates -- keyed by
`(CacheEngineKey, ActivationKind, layer_idx)`. It exists to support multi-stage
pipelines (e.g. vLLM-Omni thinker -> talker) and activation-reuse schemes where
a downstream consumer needs a cached prefix's intermediate activations rather
than recomputing them.

## Relationship to HiddenStateStore

`HiddenStateStore` (PR #3221) is the `kind=ActivationKind.HIDDEN` special case
of this store. The only structural change is the slot key: where the
hidden-state store keyed each chunk by `layer_idx` alone, `ActivationStore` keys
by `(ActivationKind, layer_idx)`. The pinned pool, chunking, LRU, and
coupled-eviction logic are unchanged. One deliberate behavioral change: the
store **preserves the activation's native dtype** (bf16/fp16) instead of forcing
float32, halving the footprint for typical activations.

## Goals

- Keep KV cache logic and activation cache logic in **separate classes**.
  `LMCacheEngine` owns one instance and exposes it via `engine.activation_store`.
- Use a **separate pinned CPU memory pool** so activation allocations cannot
  fragment the KV pool.
- **Reuse** existing chunking and key generation
  (`ChunkedTokenDatabase.process_tokens` -> `CacheEngineKey`) so each activation
  chunk is keyed identically to its KV counterpart.
- Implement the **coupled-but-asymmetric** eviction rule:
  - KV evicted -> activations for that chunk are dropped on next access.
  - Activation evicted -> KV stays.
  - Restore stops at the first chunk where the requested activation is missing.

## Class layout

```
LMCacheEngine
  - storage_manager     (KV: LocalCPUBackend + LocalDiskBackend + ...)
  - activation_store    (ActivationStore, owns its own pinned pool)
       - _allocator     (MixedMemoryAllocator, independent buffer)
       - _chunks: dict[CacheEngineKey, dict[(ActivationKind, layer_idx), MemoryObj]]
       - _lru:    OrderedDict[CacheEngineKey, None]
```

`ActivationStore` does not depend on `StorageManager` for storage. It holds a
reference to the storage manager (set via `bind_storage_manager()`) so that on
retrieve it can ask "is KV still here for this key?" via
`storage_manager.contains(key)`.

## Public API

Integrators call these on **`engine.activation_store`** when it is not `None`
(when `config.enable_activation_cache` is `True`). Callers **must** check for
`None` themselves (same pattern as vLLM-Omni `OmniGPUModelRunner`).

- `store_activation(token_ids, activation, *, kind, layer_idx=0, token_offset=0) -> int`
  Chunks `token_ids` with the engine's `token_database`, copies the matching
  rows of `activation` into pinned memory under the KV chunk key, returns the
  number of chunks stored. `token_offset` supports incremental stores (only
  chunks at or after the offset are written; partially-covered chunks are
  skipped to stay atomic with KV boundaries).
- `retrieve_activation(token_ids, *, kind, layer_idx=0) -> torch.Tensor | None`
  Walks chunks in order. Stops at the first chunk where either KV is missing
  (lazy coupled-eviction cleanup) or the requested activation slot is missing
  (prefix-strict). Returns the contiguous prefix tensor (native dtype) or `None`.
- `close()` -- frees the pinned pool.

Multiple kinds/layers are handled by invoking the calls once per
`(kind, layer_idx)`.

## Activation kinds and capture points (producer side)

`ActivationKind` enumerates what may be cached. Each is a dense
`[num_tokens, feature_dim]` tensor mapping 1:1 to tokens. The store is
capture-agnostic; the producer (vLLM integration) decides where each is read:

| Kind               | feature_dim          | vLLM capture point                                                              |
| ------------------ | -------------------- | ------------------------------------------------------------------------------- |
| `HIDDEN`           | hidden_size          | already exposed via EAGLE3 `aux_hidden_states` (`vllm/v1/spec_decode/extract_hidden_states.py`, `gpu_model_runner` aux layers) |
| `QUERY`            | n_q_heads*head_dim   | needs a forward hook at `q_proj` output (e.g. model `*Attention.forward`)       |
| `KEY` / `VALUE`    | n_kv_heads*head_dim  | already in the KV cache; cache here only if a non-paged copy is wanted          |
| `MLP_INTERMEDIATE` | d_ffn                | needs a forward hook at the MLP gate/up output                                  |

Only `HIDDEN` has existing vLLM plumbing; the others require a capture hook.

## Eviction (lazy coupled-check)

- Allocator pressure: `ActivationStore` evicts its own LRU entry and retries;
  KV is never touched.
- KV evictions are **not** observed eagerly. On every retrieve we ask
  `storage_manager.contains(key)`. If KV is gone for a chunk we hold, we drop
  that activation entry and stop the prefix there.

This keeps the engine surface tiny (no callbacks, no shared index) while
satisfying: KV evict implies activation evict (next read drops the orphan);
activation evict does not imply KV evict; restore stops at the first missing
activation *or* missing KV chunk.

## Configuration

| Field                          | Meaning                                                                 |
| ------------------------------ | ----------------------------------------------------------------------- |
| `enable_activation_cache`      | Master toggle. When `False`, `engine.activation_store` is `None`.       |
| `max_activation_cpu_size` (GiB)| Independent pinned-CPU pool size for the activation allocator.          |
| `activation_layers`            | Optional allowlist of `layer_idx` values accepted on store.            |

## Why lazy eviction (not callbacks)

- No invasive changes to `LocalCPUBackend` or any cache policy.
- Works uniformly for any backend implementing `contains()`.
- Costs one extra `contains()` call per chunk on retrieve -- a cheap dict
  lookup for the local case.

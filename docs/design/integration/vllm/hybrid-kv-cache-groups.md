# Hybrid KV Cache Groups

## Summary

Minimal hybrid memory allocator (HMA) design for the multiprocess vLLM
connector. It separates three concepts:

- **Engine KV cache group** — a group defined by the serving engine (vLLM's
  `KVCacheConfig.kv_cache_groups`). Each is one distinct paged-block address
  space; block IDs are only meaningful within one group.
- **`EngineGroupInfo`** — LMCache's engine-neutral, `msgspec`-encoded record of
  one such group (`group_view.py`). A `list[EngineGroupInfo]` is the
  registration contract.
- **`KVLayerGroupInfo`** — the server's runtime transfer-kernel dispatch unit,
  built from the engine group infos + the real tensors (`kv_layer_groups.py`).

vLLM groups layers by cache behavior; LMCache must transfer by physical layout
(kv_size, num_heads, head_size, block_size, dtype) *and* keep distinct engine
block-id spaces separate. So at registration we build engine group infos, and
store/retrieve address those infos directly.

## Goals / Non-Goals

- Keep the ZMQ API engine-neutral; confine vLLM field reads to
  `lmcache.integration.vllm`.
- Registration defines the protocol-visible group order; store/retrieve block
  IDs are indexed by that order.
- Reuse one grouping primitive (`group_layers_by_identity`) on both the vLLM and
  server sides so group order matches.
- **Not** in scope: sliding-window load-plan trimming; DeepSeek-V4 slot
  compression (`compress_ratio > 1`, packing several logical tokens per physical
  slot — the per-group machinery exists but is validated separately); HMA on the
  non-GPU transfer path (it rejects multi-group); removing `layout_hints` (still
  used for tensor layout). Per-group block *sizes* and cross-layer KV sharing
  *are* supported (see Store and retrieve).

## Types

- **`EngineGroupInfo`** (`msgspec.Struct`): `engine_group_id` (which engine
  block group its layers live in; dense from 0) + `layer_indices`. Several infos
  may share an `engine_group_id` when one engine group is split by physical
  transfer identity. The list order is the protocol-visible group order; an
  empty list means a single non-hybrid group.
- Helpers in `group_view.py` operate on `Sequence[EngineGroupInfo]`:
  `num_engine_groups`, `num_engine_group_infos`, `expand_engine_block_ids`,
  `get_engine_group_indices`.
- **`KVLayerGroupInfo`** (runtime, server-only): layer indices,
  `PageBufferShapeDesc`, dtype, compress ratio, physical chunk size,
  `engine_group_idx`. Derived from real tensors — never the API contract.

## Data flow

```text
vLLM KVCacheConfig + registered kv_caches
  | integration.vllm.kv_cache_groups.create_engine_group_infos_from_vllm
  v
list[EngineGroupInfo]  --REGISTER_KV_CACHE (msgspec)-->  server msgspec-decode
  | KVLayerGroupsManager validates against real tensors
  v
KVLayerGroupInfo list   --STORE/RETRIEVE block_ids per info-->  transfer kernels
```

## Registration

`create_engine_group_infos_from_vllm` (the only place that reads vLLM `KVCacheConfig`):

1. Inspect registered tensors for physical layout/dtype.
2. Map each registered layer to its engine group index; layers absent from
   every group's `layer_names` (cross-layer KV-sharing layers) are tagged
   `EXCLUDED_ENGINE_GROUP` and dropped (see Cross-layer KV sharing).
3. `group_layers_by_identity` splits layers by transfer identity
   `(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype)` — the
   `engine_group_idx` term keeps identically-shaped layers from different engine
   groups in separate infos.
4. Emit one `EngineGroupInfo` per identity; send the list in the
   `REGISTER_KV_CACHE` payload (the message queue encodes it).

## Store and retrieve

vLLM reports block IDs per engine group. The worker adapter re-indexes them to
engine-group-info order with `expand_engine_block_ids(engine_group_infos, block_ids)` (each
info reuses its source engine group's block IDs), so `STORE`/`RETRIEVE` receive
`list[list[int]]` indexed by info order. The server loop is then trivial: for
info `i`, use `gpu_block_ids[i]`.

### Per-group block sizes

Engine groups may use *different* `block_size`s. When a hybrid model's
attention types have different per-token page sizes, vLLM unifies the physical
page size by scaling the smaller-page group's `block_size` up (e.g.
`google/gemma-4-E4B-it`: sliding-window groups `block_size=32`, full-attention
groups `block_size=16`). The connector's block accounting (hit counts,
`blocks_in_chunk`, the `start`/`end` range) stays in the *canonical* unit —
`cache_config.block_size`, the GCD of all group block sizes — while each group's
block IDs are in its own `block_size`. So the scheduler-side slice divides the
canonical range by `k_g = group_block_size / canonical` per group
(`_slice_block_ids`), and the server counts `blocks_per_chunk = chunk // bs` per
group (`GPUCacheContext.blocks_for_tokens`). The server's per-group
`compress_ratio` is derived from the *per-group* logical block size
(`max(canonical, bs)`), so an uncompressed larger-block group gets
`compress_ratio == 1` rather than being rejected.

### Cross-layer KV sharing

Some models (e.g. `google/gemma-4-E4B-it`) reuse one layer's KV cache for
another (`kv_caches[layer] = kv_caches[target]`). vLLM lists only cache-*owning*
layers in `kv_cache_groups`; a sharing layer is absent from every group's
`layer_names`. Such a layer's KV physically lives in its target owner's blocks,
so storing/retrieving the owner already covers it. Registration therefore tags
unlisted layers with `EXCLUDED_ENGINE_GROUP` and `group_layers_by_identity`
skips them — they never form their own info. (Placing them in a group would
duplicate work and, when their block size differs from the group they default
into, corrupt the per-group block-id counts.)

**Store is all-or-nothing (fail-closed):** if the block IDs don't fully cover
every chunk for every group (e.g. a caller bug), or a copy fails, the whole
store is skipped and nothing is committed — a later retrieve simply misses and
the engine recomputes. The non-GPU transfer path rejects multi-group transfers
outright.

## Example

vLLM exposes two engine groups — group 0: layers [0,2,4], group 1: [1,3]. If
layers 0–3 share a shape but layer 4 differs, registration produces:

```text
info 0: engine group 0, layers [0, 2]
info 1: engine group 1, layers [1, 3]
info 2: engine group 0, layers [4]
```

Block IDs `{group 0: [10,11], group 1: [20,21]}` are sent as
`[[10,11], [20,21], [10,11]]` (infos 0 and 2 share group 0's IDs).

## Invariants

- The `list[EngineGroupInfo]` order is the protocol-visible group order; callers
  send one block-id list per info.
- vLLM-specific access stays in `lmcache.integration.vllm`; infos carry neutral
  metadata only.
- The server reproduces grouping with the same `group_layers_by_identity`; real
  tensors remain the source of truth for shape/dtype/stride.

## Not supported

Mamba / linear-attention hybrids (e.g. Qwen3-Next): their recurrent state caches
have no LMCache transfer format yet. vLLM still exposes them as KV cache groups,
but LMCache cannot store/retrieve those layers.

## Code map

| Area | File |
|---|---|
| Engine group info (IPC type) + helpers | `lmcache/v1/multiprocess/group_view.py` |
| Shared grouping primitive | `lmcache/v1/kv_layer_groups.py` |
| vLLM → `list[EngineGroupInfo]` | `lmcache/integration/vllm/kv_cache_groups.py` |
| Register / store / retrieve | `lmcache/integration/vllm/{lmcache_mp_connector,vllm_multi_process_adapter}.py` |
| Server GPU context / transfer | `lmcache/v1/multiprocess/{gpu_context,modules/gpu_transfer}.py` |
| ZMQ protocol | `lmcache/v1/multiprocess/protocols/engine.py` |

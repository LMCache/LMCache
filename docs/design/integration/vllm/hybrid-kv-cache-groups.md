# Hybrid KV Cache Groups

## Summary

Minimal hybrid memory allocator (HMA) design for the multiprocess vLLM
connector. It separates three concepts:

- **Engine KV cache group** — a group defined by the serving engine (vLLM's
  `KVCacheConfig.kv_cache_groups`). Each is one distinct paged-block address
  space; block IDs are only meaningful within one group.
- **`LMCacheGroupView`** — LMCache's engine-neutral, `msgspec`-encoded view of
  one such group (`group_view.py`). A `list[LMCacheGroupView]` is the
  registration contract.
- **`KVLayerGroupInfo`** — the server's runtime transfer-kernel dispatch unit,
  built from the views + the real tensors (`kv_layer_groups.py`).

vLLM groups layers by cache behavior; LMCache must transfer by physical layout
(kv_size, num_heads, head_size, block_size, dtype) *and* keep distinct engine
block-id spaces separate. So at registration we build group views, and
store/retrieve address those views directly.

## Goals / Non-Goals

- Keep the ZMQ API engine-neutral; confine vLLM field reads to
  `lmcache.integration.vllm`.
- Registration defines the protocol-visible group order; store/retrieve block
  IDs are indexed by that order.
- Reuse one grouping primitive (`group_layers_by_identity`) on both the vLLM and
  server sides so group order matches.
- **Not** in scope: sliding-window load-plan trimming; DeepSeek-V4
  logical→physical block translation; HMA on the non-GPU transfer path (it
  rejects multi-group); removing `layout_hints` (still used for tensor layout).

## Types

- **`LMCacheGroupView`** (`msgspec.Struct`): `engine_group_id` (which engine
  block group its layers live in; dense from 0) + `layer_indices`. Several views
  may share an `engine_group_id` when one engine group is split by physical
  transfer identity. The list order is the protocol-visible group order; an
  empty list means a single non-hybrid group.
- Helpers in `group_view.py` operate on `Sequence[LMCacheGroupView]`:
  `num_engine_groups`, `num_group_views`, `expand_block_ids_to_views`,
  `get_engine_group_indices`.
- **`KVLayerGroupInfo`** (runtime, server-only): layer indices,
  `PageBufferShapeDesc`, dtype, compress ratio, physical chunk size,
  `engine_group_idx`. Derived from real tensors — never the API contract.

## Data flow

```text
vLLM KVCacheConfig + registered kv_caches
  | integration.vllm.kv_cache_groups.create_group_views_from_vllm
  v
list[LMCacheGroupView]  --REGISTER_KV_CACHE (msgspec)-->  server msgspec-decode
  | KVLayerGroupsManager validates against real tensors
  v
KVLayerGroupInfo list   --STORE/RETRIEVE block_ids per view-->  transfer kernels
```

## Registration

`create_group_views_from_vllm` (the only place that reads vLLM `KVCacheConfig`):

1. Inspect registered tensors for physical layout/dtype.
2. Map each registered layer to its engine group index.
3. `group_layers_by_identity` splits layers by transfer identity
   `(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype)` — the
   `engine_group_idx` term keeps identically-shaped layers from different engine
   groups in separate views.
4. Emit one `LMCacheGroupView` per identity; send the list in the
   `REGISTER_KV_CACHE` payload (the message queue encodes it).

## Store and retrieve

vLLM reports block IDs per engine group. The worker adapter re-indexes them to
group-view order with `expand_block_ids_to_views(group_views, block_ids)` (each
view reuses its source engine group's block IDs), so `STORE`/`RETRIEVE` receive
`list[list[int]]` indexed by view order. The server loop is then trivial: for
view `i`, use `gpu_block_ids[i]`.

**Store is all-or-nothing (fail-closed):** if the block IDs don't fully cover
every chunk for every group (e.g. a caller bug), or a copy fails, the whole
store is skipped and nothing is committed — a later retrieve simply misses and
the engine recomputes. The non-GPU transfer path rejects multi-group transfers
outright.

## Example

vLLM exposes two engine groups — group 0: layers [0,2,4], group 1: [1,3]. If
layers 0–3 share a shape but layer 4 differs, registration produces:

```text
view 0: engine group 0, layers [0, 2]
view 1: engine group 1, layers [1, 3]
view 2: engine group 0, layers [4]
```

Block IDs `{group 0: [10,11], group 1: [20,21]}` are sent as
`[[10,11], [20,21], [10,11]]` (views 0 and 2 share group 0's IDs).

## Invariants

- The `list[LMCacheGroupView]` order is the protocol-visible group order; callers
  send one block-id list per view.
- vLLM-specific access stays in `lmcache.integration.vllm`; views carry neutral
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
| Group view (IPC type) + helpers | `lmcache/v1/multiprocess/group_view.py` |
| Shared grouping primitive | `lmcache/v1/kv_layer_groups.py` |
| vLLM → `list[LMCacheGroupView]` | `lmcache/integration/vllm/kv_cache_groups.py` |
| Register / store / retrieve | `lmcache/integration/vllm/{lmcache_mp_connector,vllm_multi_process_adapter}.py` |
| Server GPU context / transfer | `lmcache/v1/multiprocess/{gpu_context,modules/gpu_transfer}.py` |
| ZMQ protocol | `lmcache/v1/multiprocess/protocols/engine.py` |

# Hybrid KV Cache Groups

## Summary

This document describes the minimal hybrid memory allocator (HMA) KV cache
group design used by the multiprocess vLLM connector.

The key idea is to separate three concepts that are easy to conflate:

- Engine KV cache groups: groups defined by the serving engine.
- The LMCache KV spec (`list[LMCacheGroupView]`): LMCache's engine-neutral,
  `msgspec`-encoded group contract.
- `KVLayerGroupInfo`: LMCache's runtime transfer-kernel dispatch groups.

vLLM may organize KV cache groups by engine-side cache behavior. LMCache needs
to transfer KV tensors by physical layout compatibility: KV size, number of
heads, head size, physical block size, dtype, and the engine block-id space.
Therefore, vLLM-side groups are built into LMCache KV groups at registration
time. Store and retrieve requests then address those LMCache groups directly.

## Motivation

Serving engines may expose multiple KV cache groups. Those groups can represent
different block-id spaces, cache policies, or layer families. LMCache must keep
those engine block-id spaces separate while also grouping layers by the physical
properties required by its transfer kernels.

This design focuses on the minimal HMA contract: layers from different engine
KV cache groups must not be merged into one LMCache transfer group, because
their block IDs come from different engine block-id spaces. Model-specific
logical/physical block mapping, sliding-window trimming, and transfer
optimizations are separate concerns layered on top of this contract.

## Goals

- Keep the ZMQ API engine-neutral.
- Keep vLLM-specific field reads in `lmcache.integration.vllm`.
- Make registration define the protocol-visible LMCache KV group order.
- Make store and retrieve block IDs indexed by LMCache KV group index.
- Reuse the same layer-grouping logic for vLLM-side group construction and
  server-side runtime group construction.
- Keep real tensors as the source of truth for physical transfer shape.

## Non-Goals

- This design does not implement sliding-window load-plan trimming.
- This design does not implement DeepSeek V4 logical-to-physical block ID
  translation.
- This design does not make the non-GPU transfer path HMA-aware.
- This design does not remove `layout_hints`; it narrows their role to tensor
  layout and shape normalization metadata.

## Terminology

### Engine KV Cache Group

An engine KV cache group is a serving-engine-native group. In vLLM this comes
from `KVCacheConfig.kv_cache_groups`.

The engine group ID is recorded as `engine_group_id` because block IDs
reported by vLLM are indexed by engine group (one distinct paged-block address
space per group).

### `LMCacheGroupView`

`LMCacheGroupView` is LMCache's neutral group descriptor, a `msgspec.Struct`. It does
not contain vLLM objects. Each group records:

- `engine_group_id`;
- `layer_indices`.

Each `LMCacheGroupView` corresponds to one protocol-visible LMCache KV group. Several
`LMCacheGroupView`s may share the same `engine_group_id` when one engine block
group is split by physical transfer identity. Store and retrieve block IDs are
indexed by this group order.

### LMCache KV spec (`list[LMCacheGroupView]`)

The registration contract is simply a `list[LMCacheGroupView]` — there is no wrapper
type. Because each element is a `msgspec.Struct`, the multiprocess message queue
encodes/decodes the list directly in the `REGISTER_KV_CACHE` payload (typed
`list[LMCacheGroupView]`); there is no separate JSON serialization step. An empty
list means a single non-hybrid group (the default for engines that do not report
KV cache group metadata).

Engine-neutral module-level helpers in `group_view.py` operate on a
`Sequence[LMCacheGroupView]`:

- `num_engine_groups(groups)`;
- `num_lmc_kv_cache_groups(groups)`;
- `expand_block_ids_to_lmc_groups(groups, engine_side_block_ids)`;
- `get_engine_group_indices(groups, num_registered_layers)`.

### `KVLayerGroupInfo`

`KVLayerGroupInfo` is runtime-only server metadata for one transfer-kernel
dispatch group. It contains kernel-facing data such as:

- layer indices;
- `PageBufferShapeDesc`;
- dtype;
- compression ratio;
- physical chunk size;
- engine group index.

It should not be serialized as the API contract because it depends on the real
registered tensors and kernel implementation details.

## Data Flow

```text
vLLM KVCacheConfig + registered kv_caches
        |
        | lmcache.integration.vllm.kv_cache_groups
        v
list[LMCacheGroupView]
        |
        | REGISTER_KV_CACHE over ZMQ
        v
LMCache server msgspec-decodes list[LMCacheGroupView]
        |
        | KVLayerGroupsManager validates against real tensors
        v
KVLayerGroupInfo list
        |
        | STORE / RETRIEVE block_ids indexed by LMCache group index
        v
Transfer kernels
```

## Registration

During `register_kv_caches`, the vLLM connector builds the `list[LMCacheGroupView]`
via `create_engine_groups_from_vllm(...)`.

The process is:

1. Inspect the registered KV tensors for physical layout and dtype.
2. Read vLLM's native KV cache groups to map each registered layer to its
   engine group index (the only place that reads vLLM `KVCacheConfig`).
3. Reuse `group_layers_by_identity(...)` to split layers by LMCache physical
   transfer identity.
4. Emit one `LMCacheGroupView` per LMCache transfer identity.
5. Pass the `list[LMCacheGroupView]` in the `REGISTER_KV_CACHE` payload; the message
   queue `msgspec`-encodes it directly.

The layer identity used for grouping is:

```text
(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype)
```

The `engine_group_idx` component is important. It prevents layers with
identical tensor shape from being merged if their block IDs come from different
engine KV cache groups.

## Store and Retrieve

vLLM scheduler metadata still naturally starts as engine-group-indexed block
IDs:

```text
block_ids[engine_group_idx] -> list[int]
```

Before sending requests to the LMCache server, the worker adapter expands these
block IDs to LMCache group order with
`expand_block_ids_to_lmc_groups(groups, engine_side_block_ids)`, where each
LMCache group reuses the block IDs from its source engine group
(`groups[lmc_group_idx].engine_group_id`).

The ZMQ `STORE` and `RETRIEVE` APIs therefore receive:

```text
list[list[int]]
```

where the outer list is indexed by LMCache KV group index, not engine group
index.

This makes the server-side transfer loop simple: for each `KVLayerGroupInfo`
at `lmc_group_idx`, read `gpu_block_ids[lmc_group_idx]`.

## Example

Suppose vLLM exposes two engine KV cache groups:

```text
engine group 0: layers [0, 2, 4]
engine group 1: layers [1, 3]
```

If layers 0, 1, 2, and 3 have the same transfer shape, but layer 4 has a
different hidden dimension, the conversion produces:

```text
LMCache group 0: engine group 0, layers [0, 2]
LMCache group 1: engine group 1, layers [1, 3]
LMCache group 2: engine group 0, layers [4]
```

When vLLM reports block IDs:

```text
engine group 0: [10, 11]
engine group 1: [20, 21]
```

the worker adapter sends:

```text
LMCache group 0: [10, 11]
LMCache group 1: [20, 21]
LMCache group 2: [10, 11]
```

## Invariants

- The `list[LMCacheGroupView]` order is the protocol-visible LMCache group order.
- Store and retrieve request block IDs are indexed by that LMCache group order:
  callers must send one block-id list per LMCache group.
- vLLM-specific metadata access stays in `lmcache.integration.vllm`.
- `LMCacheGroupView` contains neutral metadata only.
- `KVLayerGroupsManager` derives runtime `KVLayerGroupInfo` from real tensors.
- Server-side runtime grouping must use the same `group_layers_by_identity(...)`
  helper as vLLM-side group construction.
- Serialized metadata may guide grouping, but real tensors remain the source of
  truth for shape, dtype, and stride.

## Caller Contract

Every caller sends block IDs in LMCache group order (one `list[int]` per LMCache
group). There is no server-side single-list fallback.

- The vLLM worker adapter expands engine-group block IDs to LMCache group order
  via `expand_block_ids_to_lmc_groups(...)`.
- Callers with a single block-id space (the TRT-LLM adapter, the
  `lmcache bench kvcache` CLI) register an empty `list[LMCacheGroupView]`. When the
  server derives several physical groups from a multi-shape registration, the
  caller sends one block-id list per group (the bench replicates its single
  logical list `num_lmc_groups` times).

## Relationship to `layout_hints`

`layout_hints` are intentionally not used for group mapping.

They remain for tensor interpretation metadata such as:

- vLLM physical KV layout (`NHD` or `HND`);
- TRT-LLM reshape metadata (`num_kv_heads`, `tokens_per_block`, `head_dim`);
- inference-engine logical block size, currently used to derive compression
  metadata.

Future work may move logical block-size or compression policy into the
`LMCacheGroupView` metadata, but HMA group identity should not be carried in
`layout_hints`.

## Alternatives Considered

### Keep `per_layer_engine_group_idx` in `layout_hints`

This was rejected because it mixes tensor-layout hints with cache-group
semantics. It also makes the group mapping look like an incidental hint rather
than part of the registration contract.

### Let the server independently regroup everything

The server still validates and builds runtime `KVLayerGroupInfo`, but the
protocol-visible group order must be established during registration. If vLLM
and LMCache independently produce different group orders, block IDs can be
applied to the wrong layer group.

### Serialize `KVLayerGroupInfo`

This was rejected because `KVLayerGroupInfo` is runtime/kernel metadata. It
contains fields derived from real tensors and should not become an external API
contract.

### Fully Remove `layout_hints`

This is not part of the minimal HMA design. `layout_hints` still serve tensor
layout and reshape purposes for vLLM and TRT-LLM. Removing them would be a
separate API migration.

## Follow-Up Work

- Add sliding-window and Mamba metadata fields to `LMCacheGroupView` as neutral
  per-group or per-layer semantics.
- Add a load-plan interface that trims retrieved tokens according to full
  attention and sliding-window cache availability.
- Add DeepSeek V4 logical/physical block ID translation as a separate layer on
  top of this group contract.
- Make non-GPU transfer paths explicitly reject or support multi-group HMA.
- Add end-to-end tests with a real vLLM HMA model configuration.

## Code Map

| Area | File |
|---|---|
| Neutral group view (IPC type) + helpers | `lmcache/v1/multiprocess/group_view.py` |
| Shared physical grouping helper | `lmcache/v1/kv_layer_groups.py` |
| vLLM conversion to `list[LMCacheGroupView]` | `lmcache/integration/vllm/kv_cache_groups.py` |
| vLLM register/store/retrieve path | `lmcache/integration/vllm/lmcache_mp_connector.py`, `lmcache/integration/vllm/vllm_multi_process_adapter.py` |
| Server-side GPU context | `lmcache/v1/multiprocess/gpu_context.py` |
| Server-side transfer loop | `lmcache/v1/multiprocess/modules/gpu_transfer.py` |
| ZMQ protocol docs | `lmcache/v1/multiprocess/protocols/engine.py` |

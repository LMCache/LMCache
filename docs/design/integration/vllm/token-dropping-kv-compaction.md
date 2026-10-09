# GPU-Native Token Dropping in LMCache

## Summary

LMCache already has a [KV Cache SDK](../../../source/mp/sdk.rst) for offline KV
transformations such as token dropping. It runs outside the vLLM worker, stages
KV and Q through CPU memory, and needs external orchestration around the serving
engine.

The MVP moves token-dropping scoring and compaction into the vLLM worker through
the existing KVConnector lifecycle. R-KV reads post-RoPE Q and paged KV directly
on GPU, compacts KV in place, and reports only control state back to the
scheduler. The LMCache server is not on the compaction data path.

Dropped KV is returned to vLLM's allocator and becomes reusable capacity for
other requests.

**Invariant:** compaction changes the resident physical KV sequence. It does not
shorten or renumber the request's logical token sequence.

## Architecture overview

```text
Prior SDK:  worker GPU -> CPU-staged KV/Q -> external SDK -> worker GPU

MVP:
  vLLM worker GPU
    post-RoPE Q + resident paged KV
        -> R-KV retained positions
        -> in-place LMCache compaction
        -> absolute resident KV length
                   |
                   | KVConnector control metadata
                   v
  vLLM scheduler / allocator
        -> commit resident KV footprint
        -> reclaim unused tail blocks
        -> normal allocation from the resident frontier
                   |
                   | next scheduler output
                   v
  vLLM worker
        logical model positions unchanged
        physical KV addressing uses resident length + current block table
```

The design deliberately keeps only two vLLM lengths:

- **logical progress**: how far the model/request has advanced;
- **resident KV length**: how many KV entries physically remain after
  compaction.

There is no cumulative dropped-token counter. The worker reports the resulting
absolute resident length after each successful compaction.

## State and ownership

| State | Source of truth | Used for |
|---|---|---|
| Logical progress | vLLM request | Model positions, RoPE, request progress |
| Resident KV length | vLLM KV allocator | Allocation, reclaim, physical KV addressing |
| Block IDs | vLLM scheduler / allocator | Physical blocks owned by the request |
| Recent R-KV query window | LMCache worker | Token-selection policy only |

For example, after 1,000 logical tokens a request may have only 256 resident KV
entries. The next model position is still 1,000, while KV read/write addressing
continues from physical position 256.

vLLM remains authoritative for allocation and block ownership. LMCache owns the
token-selection policy and the GPU KV rewrite.

## R-KV and compaction contract

R-KV returns retained positions from the request's **current physical KV
sequence**. The MVP uses one request-level retained set shared by all KV heads
and layers.

For example:

```text
Current KV:      [A B C D | E F G H]
Keep (0-based):  [1, 3, 4, 7]
Compacted KV:    [B D E H | - - - -]
```

If the request is compacted again, indices refer to the already-compacted
sequence:

```text
[A B C D E F G H] -- keep [1,3,4,7] --> [B D E H]
[B D E H]         -- keep [1,3]     --> [D H]
```

No original-position map is required.

The worker already has the request's physical block table on GPU. It maps the
retained sequence positions directly to source physical slots and maps the
front of the same resident sequence to destination slots:

```text
source_slots      = resident_slots[retained]
destination_slots = resident_slots[:budget]
```

The executor applies those source/destination copies in place. For
FlashAttention's NHD cache layout, the worker views
`[blocks, 2, slots, heads, dim]` as `[blocks, slots, 2, heads, dim]`, so one
physical-slot copy moves K and V together.

The MVP compacts only private blocks. Prefix caching/shared-block copy-on-write
is out of scope.

## Query capture

The MVP does not use the SDK QRingBuffer or send Q through the LMCache server.

Two small vLLM connector-context hooks expose data that already exists in the
worker:

1. after the worker has condensed/reordered its batch, its request IDs are
   passed to `start_load_kv(..., request_ids=...)`;
2. the attention wrapper passes its post-RoPE `query` tensor through the
   existing `save_kv_layer(..., **kwargs)` layer-I/O hook.

The attention metadata already contains `query_start_loc`, physical
`seq_lens`, and the current `block_table`. LMCache therefore slices query
rows directly by request without block-ID intersection or a second Q transport.

Each request/layer keeps only the trailing eight query rows needed by R-KV.

## Trigger

The MVP uses the minimum trigger cadence that guarantees a fresh eight-query
window between repeated compactions:

```text
compact when resident_len >= budget + 8
compact back to budget
```

There is no separate compression buffer or trigger-reserve setting.

## Scheduler / worker flow

Compaction happens at the end of a completed forward step, before the next
synchronous scheduling step:

```text
Step N forward
  -> capture post-RoPE Q
  -> R-KV selects retained physical positions
  -> compact every KV layer on GPU
  -> worker reports absolute resident length

scheduler side
  -> commit the new resident length in vLLM's KVCacheManager
  -> truncate/reclaim tail blocks
  -> run normal allocation for Step N+1 from the resident frontier
  -> if ownership changed, send the full authoritative block table
     together with the resident-length refresh

worker
  -> update cached resident length + block table
  -> model positions remain logical
  -> KV read/write addressing uses the resident frontier
  -> Step N+1 forward
```

A resident-state refresh is one semantic event: the worker receives the new
resident length and replaces its cached block table with the scheduler's full
authoritative table. Normal later allocations remain append-only.

## vLLM allocation contract

After compaction, vLLM allocation should behave exactly like an ordinary
request whose current live KV length equals the resident KV length.

For example:

```text
logical progress = 104
resident KV      = 32

next token:
  model position       104
  resident KV           32 -> 33
  allocator             allocates the third physical block
```

This is implemented as a small resident-footprint primitive in
`KVCacheManager`. It avoids R-KV-specific allocator policy, null holes, or
compatibility headroom.

## vLLM integration boundary

The MVP targets pinned vLLM v0.25.1's classic GPU runner and needs six narrow
integration pieces:

1. KVCacheManager tracks an optional resident KV footprint and exposes a
   shrink-only absolute commit operation;
2. connector worker output can return absolute resident-length updates for the
   scheduler to commit;
3. a resident update refreshes the worker's resident length and full block
   table;
4. KV read/write addressing uses the resident frontier while model positions
   remain logical;
5. worker-order request IDs are exposed through the existing
   `start_load_kv(..., **kwargs)` hook;
6. post-RoPE Q is exposed through the existing
   `save_kv_layer(..., **kwargs)` hook.

These are generic connector/allocator semantics. vLLM does not know R-KV's
budget, trigger, scoring rule, or retained positions.

## Preemption and failure

On preemption, vLLM frees the request's block ownership, which also clears its
resident-footprint override. LMCache clears its rolling R-KV query state using
the existing preemption/flush signal. The request can then follow vanilla vLLM
recomputation from logical state.

No R-KV-specific request termination or recovery protocol is needed for normal
preemption.

All detectable unsupported conditions are rejected before mutation. If an
unexpected runtime/CUDA failure occurs after in-place compaction has begun, the
MVP fails the worker/engine rather than attempting request-local recovery from
partially rewritten KV.

## MVP scope

The MVP supports:

- one GPU;
- classic vLLM model runner;
- eager execution;
- synchronous scheduling;
- full prefill followed by decode;
- one ordinary full-attention KV cache group;
- FlashAttention NHD paged KV;
- BF16/FP16 KV.

The MVP rejects or excludes:

- prefix caching/shared KV blocks;
- LMCache STORE/RETRIEVE in the same R-KV request stream;
- chunked prefill;
- speculative decoding;
- cascade attention;
- async scheduling;
- multi-GPU execution;
- hybrid/quantized KV layouts.

Requests without `lmcache.mp.rkv_budget` keep the normal LMCache/vLLM path.

## Follow-up

Possible follow-up work includes shared-block copy-on-write, persistence of
compressed KV with compression configuration in the cache identity, broader KV
layouts, async scheduling, multi-GPU support, and performance optimization.

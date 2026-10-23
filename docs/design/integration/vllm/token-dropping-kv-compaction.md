# GPU-Native Token Dropping in LMCache

## Summary

LMCache already has a [KV Cache SDK](../../../source/mp/sdk.rst) for offline KV
transformations such as token dropping. It runs outside the vLLM worker, stages
KV and Q through CPU memory, and needs external orchestration around the serving
engine.

The MVP moves token-dropping scoring and compaction into the vLLM worker through
the existing KVConnector lifecycle. R-KV reads post-RoPE Q and paged KV directly
on GPU; LMCache compacts KV in place and reports only control state back to the
scheduler. The LMCache server is not on the compaction data path.

Dropped KV is returned to vLLM's allocator and becomes reusable capacity for
other requests.

**Invariant:** compaction changes the resident physical KV sequence. It does not
shorten or renumber the request's logical token sequence.

## Key capabilities

- **Prefill + decode:** Observe Q and compact KV in both phases. Prefill compaction completes before the first decode.
- **Full R-KV behavior (MVP target):** Independent per-layer/KV-head selection, including prefill and repeated decode compaction.
- **Per-request configuration:** The serving app selects an algorithm and config per request; normal requests are unchanged.
- **Algorithm-defined policy:** Algorithms choose which Q to observe, when to compact, and which tokens to keep; LMCache handles GPU access and compaction.

SnapKV compatibility: SnapKV on MHA is compatible with this API but not included in the MVP. Original GQA SnapKV requires independent KV selection per Q head, which this API does not support.

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
| Resident KV length | LMCache allocator adapter | Allocation, reclaim, physical KV addressing |
| Block IDs | vLLM scheduler / allocator | Physical blocks owned by the request |
| Q observation state | Each request's token-dropping algorithm | Token-selection policy only |

For example, after 1,000 logical tokens a request may have only 256 resident KV
entries. The next model position is still 1,000, while KV read/write addressing
continues from physical position 256.

vLLM remains authoritative for allocation and block ownership. LMCache uses
R-KV's selected positions to compact KV on GPU.

## API Contract Between LMCache and Token-Dropping Algorithms (e.g., R-KV)

LMCache creates one algorithm instance per token-dropping request using the selected factory. The instance
implements the four methods below.

### 1. Create an algorithm instance

```python
def from_serving_config(
    config: Mapping[str, Any],
) -> TokenDropAlgorithm: ...
    # {} = empty config; use defaults
```

### 2. Choose how much Q to capture

```python
def should_observe_token_queries(
    phase: Literal["prefill", "decode"],
    decoded_tokens_before_step: int,
) -> int: ...
    # Capture the last N Q rows from this forward; 0 = none
```

### 3. Record observed Q

```python
def observe_token_queries(
    queries_by_layer: Mapping[str, torch.Tensor],
) -> None: ...
    # Post-RoPE Q; one or more tokens per layer
```

### 4. Decide whether to compact KV

```python
def should_compact_kv(
    phase: Literal["prefill", "decode"],
    resident_kv_tokens: int,
    decoded_tokens_before_step: int,
) -> bool: ...
    # resident_kv_tokens includes this step's KV write
```

### 5. Select KV entries to retain

```python
def select_kept_token_positions(
    kv_by_layer: Mapping[str, KVView],
) -> Mapping[str, torch.Tensor]: ...
    # Read-only access to GPU-resident K/V per layer
    # Per-layer/KV-head positions in current resident KV; algorithm-defined order
    # Same retained count across all heads/layers
```

## R-KV and compaction contract

R-KV returns retained positions from the request's **current physical KV
sequence**. Each layer and KV head selects its own positions, with the same
retained count across all heads and layers.

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

The MVP compacts only private blocks. Prefix caching/shared-block copy-on-write
is out of scope.

## Query capture

The MVP does not use the SDK QRingBuffer or send Q through the LMCache server.

The attention metadata already contains `query_start_loc`, physical
`seq_lens`, and the current `block_table`. LMCache therefore slices query
rows directly by request without block-ID intersection or a second Q transport.

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
  -> commit the new resident length in LMCache's scheduler-side connector
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

This is handled by LMCache's allocator adapter. It avoids R-KV-specific
allocator policy, null holes, or compatibility headroom.

## vLLM integration boundary

The MVP targets pinned vLLM v0.25.1's classic GPU runner and needs six narrow
integration pieces:

1. LMCache's allocator adapter tracks an optional resident KV footprint and
   receives shrink-only absolute updates from the scheduler-side connector;
2. connector worker output can return absolute resident-length updates for the
   scheduler to commit;
3. a resident update refreshes the worker's resident length and full block
   table;
4. KV read/write addressing uses the resident frontier while model positions
   remain logical;
5. worker-order request IDs and their batch rows are exposed to the
   token-dropping worker;
6. post-RoPE Q is captured through the attention backend's `forward` call.

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
- eager or PIECEWISE CUDA graph execution;
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

Requests without token dropping keep the normal LMCache/vLLM path.

## Follow-up

Possible follow-up work includes shared-block copy-on-write, persistence of
compressed KV with compression configuration in the cache identity, broader KV
layouts, async scheduling, multi-GPU support, and performance optimization.

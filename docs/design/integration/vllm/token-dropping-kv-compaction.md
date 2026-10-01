# GPU-Native Token Dropping in LMCache

## Summary

LMCache already has a token-dropping SDK path for offline experiments. It runs
outside the vLLM worker, stages KV and Q through CPU memory, and needs external
orchestration around the serving engine.

M1 moves the token-dropping algorithm and compaction into the vLLM worker. The
R-KV algorithm reads post-RoPE query (Q) and KV directly on GPU. LMCache
compacts the selected KV there, and the LMCache server stays off the compaction
path. Only control metadata crosses the worker / scheduler boundary.

The goal is not just fewer KV entries. **Dropped KV should become real vLLM
capacity that other requests can reuse.** R-KV is the first token-dropping
algorithm on this path; the compaction mechanism only consumes the retained
positions produced by the algorithm.

**Invariant:** compaction changes the physical KV layout; it does not shorten or
renumber the request's logical token sequence.

## Architecture overview

```text
Prior SDK:  worker GPU -> CPU-staged KV/Q -> external SDK -> worker GPU
M1:         worker GPU -> token-dropping algorithm + compaction -> worker GPU
```

For M1's one-KV-entry-per-token layout, at a step boundary:

- `L` is the number of logical tokens already processed for the request.
- `D` is the number of KV entries dropped so far.
- `P = L - D` is the number of KV entries still resident.

```text
vLLM worker (GPU)
  post-RoPE Q + paged KV
          |
     R-KV algorithm
          | retained KV indices
          v
  LMCache compaction
          | rewrite KV in place on GPU
          | report Δ = entries dropped this step
          v
  worker result
          |
          v
vLLM scheduler
  LMCache connector: D += Δ; P = L - D
  vLLM allocator: return unused tail blocks
                  run normal allocation
          |
          | next-step metadata
          | P + full block IDs if changed
          v
LMCache worker adapter
  model positions use L
  KV reads / writes use P + block IDs
          |
      next forward
```

Both updates use vLLM's existing connector interface: the worker adds `Δ` to
its result, and the scheduler sends `P` plus updated block IDs with the next
step. The token-dropping algorithm runs in the worker; M1 adds no separate
algorithm process or RPC.

## State and ownership

| State | Source of truth | Used for |
|---|---|---|
| Logical progress `L` | vLLM request | Model positions and request progress |
| Cumulative dropped count `D` | LMCache scheduler-side connector | Deriving `P` |
| Physical KV length `P` | `L - D` (derived) | Allocation and worker KV addressing |
| Block IDs | vLLM scheduler / allocator | Physical blocks owned by the request |

Before compaction, `L == P`. Afterward they can be very different: after 1,000
logical tokens, only 256 KV entries may remain in memory. The model's next
position is still 1,000; attention reads the 256 resident entries.

The worker reports `Δ` for a successful compaction, or a request-level failure
otherwise. On success, LMCache's scheduler-side connector accumulates `D`,
derives `P`, and sends `P`, not `D`, back to the worker. The worker only needs
the current physical length.

Block ownership stays with vLLM. LMCache can change the contents and effective
occupancy of a request's blocks, but it does not maintain a second allocator.

## Algorithm and compaction contract

A token-dropping algorithm returns retained positions, in sequence order, from
the request's **current physical KV sequence**. M1 compaction consumes one
request-level retained set shared by all KV heads and layers. The algorithm
decides when to produce that set; LMCache owns the GPU movement and the vLLM
integration.

For example (`|` marks a block boundary):

```text
Current KV:      [A B C D | E F G H]
Keep (0-based):  [1, 3, 4, 7] -> [B, D, E, H]
Compacted KV:    [B D E H | - - - -]
```

Moving a KV entry to a new physical slot does not change its model position.
Physical KV layout and logical token position are separate after compaction.

The planner maps those retained positions through the request's vLLM block IDs
to concrete GPU source and destination slots. The executor copies the relevant
KV entries between those slots.

If the request is compacted again, indices refer to the already-compacted
sequence:

```text
[A B C D E F G H] -- keep [1,3,4,7] --> [B D E H]
[B D E H]         -- keep [1,3]     --> [D H]
```

Compaction therefore does not need a persistent map back to original token
positions; vLLM continues to own logical progress through `L`.

```text
R-KV owns:        trigger, scoring, budget, recent Q, retained indices
Compaction owns:  retained indices -> physical KV movement
```

M1 does not support shared KV blocks. A compacted request therefore owns every
block it rewrites and can compact them in place. Supporting shared or
prefix-cached blocks later requires copy-on-write before compaction.

## Scheduler / worker flow

M1 compacts at the end of a forward step, before the next step is scheduled:

```text
Step N forward
  -> R-KV selects retained entries
  -> worker compacts KV on GPU
  -> worker reports Δ

scheduler side
  -> D += Δ
  -> P = L - D
  -> return tail blocks no longer needed by P
  -> run normal vLLM allocation for Step N+1
  -> send P + full block IDs if allocation changed

worker
  -> keep model positions based on L
  -> address KV using P + current block IDs
  -> Step N+1 forward
```

Tail blocks are returned only after the scheduler receives the compaction result
for that step; until then the worker may still be using the old allocation. The
worker result itself is the handoff, so M1 does not add another acknowledgement
before those blocks can be reclaimed.

For KV allocation, the request behaves as if it currently had `P` resident
entries. Future growth continues through normal vLLM allocation; token dropping
does not add its own reserve logic. If allocation changes the request's block
IDs, the scheduler sends the full current IDs before the next forward and the
worker mirrors them.

## vLLM integration boundary

M1 targets the pinned vLLM v0.25.1 classic GPU model runner. All pinned-vLLM
access stays in one compatibility adapter. The rest of the feature depends on
five vLLM capabilities:

1. read the post-RoPE Q needed by the token-dropping algorithm;
2. prepare KV reads and writes from `P` while model positions remain based on `L`;
3. return unused tail blocks to vLLM's block pool;
4. keep token-dropping requests from reading or publishing to the local prefix
   cache;
5. terminate a token-dropping request after preemption or compaction failure.

Keeping this boundary in one adapter lets stable upstream APIs replace the
private hooks later without changing the algorithm, compaction API, or
`L / D / P` state model.

## M1 and M2 scope

### M1: live GPU-native token dropping

M1 is complete when a running request can compact KV, return the unused blocks
to vLLM, and continue correctly without a CPU round trip. M1 supports token
dropping after a full prefill or decode step. Chunked prefill is out of scope;
M1 requires it to be disabled and rejects the configuration at startup.

M1 requires synchronous scheduling with one in-flight step. Requests co-batched
with a compaction wait for it before the next decode step. This keeps one
unambiguous physical state between steps. With async scheduling, a later step
can be scheduled before the previous `Δ` is committed. Supporting that path
needs per-step physical-state tracking, and reclaimed blocks cannot be reused
until the matching compaction has finished.

For requests with token dropping enabled, M1 bypasses LMCache persistence: it
does not store, look up, or restore KV. Token-dropping requests use private vLLM
blocks: they neither read from nor publish to the local prefix cache. Requests
without token dropping keep the existing cache behavior. M1 does not support
speculative decoding. The supported path is the classic GPU runner
on a single GPU with ordinary paged-attention KV and one physical KV entry per
token. Hybrid or sliding-window KV layouts and multi-GPU execution are deferred.

On compaction failure, the worker reports a request-level failure instead of
`Δ`. M1 does not recover token-dropping requests after preemption or compaction
failure: the scheduler terminates the request and discards its local KV and
compression state. It does not roll back or partially recover compacted state.

### M2: compacted persistence and broader runtime support

M2 adds compacted KV store and restore. A compacted cache entry has both logical
progress `L` and physical KV length `P`, so restore must recover both. M2
therefore defines store, cache identity, and restore together rather than adding
a write-only format in M1.

Other follow-up work includes async scheduling and compaction overlap,
shared-block copy-on-write, broader KV layouts, multi-GPU support, and
profiling-driven compaction optimizations.

## Code / PR map

| Area | Change |
|---|---|
| Compaction planner | [PR #5352](https://github.com/LMCache/LMCache/pull/5352) |
| GPU compaction executor | [PR #5405](https://github.com/LMCache/LMCache/pull/5405) |
| R-KV algorithm + GPU Q capture | follow-up |
| Scheduler state + block reclaim | follow-up |
| Worker physical addressing | follow-up |

Each implementation PR should point back to this document for the end-to-end
state and ownership contract.

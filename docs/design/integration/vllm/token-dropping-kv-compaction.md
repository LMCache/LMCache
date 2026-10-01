# GPU-Native Token Dropping in LMCache

## Summary

LMCache already has a [KV Cache SDK](../../../source/mp/sdk.rst) for offline KV
transformations such as token dropping. It runs outside the vLLM worker, stages
KV and Q through CPU memory, and needs external orchestration around the serving
engine.

The MVP moves the token-dropping algorithm and compaction into the vLLM worker.
The R-KV algorithm reads post-RoPE query (Q) and KV directly on GPU. LMCache
compacts the selected KV there, and the LMCache server stays off the compaction
path. Only control metadata crosses the worker / scheduler boundary.

Dropped KV becomes reusable vLLM capacity for other requests.

**Invariant:** compaction changes the physical KV layout; it does not shorten or
renumber the request's logical token sequence.

## Architecture overview

```text
Prior SDK:  worker GPU -> CPU-staged KV/Q -> external SDK -> worker GPU
MVP:        worker GPU -> token-dropping algorithm + compaction -> worker GPU
```

For the MVP, each processed token corresponds to one KV position, so physical KV
length can be tracked as:

- `logicalLen` is the number of logical tokens already processed.
- `totalDropped` is the number of KV entries dropped so far.
- `kvLen = logicalLen - totalDropped` is the number of resident KV entries.

```text
vLLM worker (GPU)
  post-RoPE Q + paged KV
          |
     R-KV algorithm
          | retained KV indices
          v
  LMCache compaction
          | rewrite KV in place on GPU
          | report stepDropped
          v
  worker result
          |
          v
vLLM scheduler
  LMCache connector:
    totalDropped += stepDropped
    kvLen = logicalLen - totalDropped
  vLLM allocator: return unused tail blocks
                  run normal allocation
          |
          | next-step metadata
          | kvLen + full block IDs if changed
          v
LMCache worker adapter
  model positions use logicalLen
  KV addressing uses kvLen + block IDs
          |
      next forward
```

Both directions use vLLM's existing connector interface: the worker reports
`stepDropped`, and the scheduler sends `kvLen` plus the full current block IDs
when allocation changes.

## State and ownership

| State | Source of truth | Used for |
|---|---|---|
| Logical progress `logicalLen` | vLLM request | Model positions and request progress |
| Cumulative dropped count `totalDropped` | LMCache scheduler-side connector | Deriving `kvLen` |
| Physical KV length `kvLen` | `logicalLen - totalDropped` (derived) | Allocation and worker KV addressing |
| Block IDs | vLLM scheduler / allocator | Physical blocks owned by the request |

After 1,000 logical tokens, 256 KV entries may remain; the model position is
still 1,000 while attention reads those 256 entries.

The worker reports `stepDropped` for a successful compaction, or a request-level
failure otherwise. On success, the scheduler-side connector accumulates
`totalDropped`, derives `kvLen`, and sends `kvLen` to the worker.

vLLM retains block ownership and allocation; LMCache only rewrites KV contents
and occupancy.

## Algorithm and compaction contract

A token-dropping algorithm returns retained positions, in sequence order, from
the request's **current physical KV sequence**. The MVP uses one request-level
retained set shared by all KV heads and layers.

For example (`|` marks a block boundary):

```text
Current KV:      [A B C D | E F G H]
Keep (0-based):  [1, 3, 4, 7] -> [B, D, E, H]
Compacted KV:    [B D E H | - - - -]
```

Moving an entry changes only its physical slot, not its model position.

The planner maps retained positions through the request's vLLM block IDs to
concrete GPU source and destination slots. The executor copies the corresponding
KV entries between those slots.

If the request is compacted again, indices refer to the already-compacted
sequence:

```text
[A B C D E F G H] -- keep [1,3,4,7] --> [B D E H]
[B D E H]         -- keep [1,3]     --> [D H]
```

Indices always refer to the current physical sequence; no original-position map
is required.

```text
R-KV owns:        trigger, scoring, retained indices
Compaction owns:  retained indices -> physical KV movement
```

The MVP compacts only private vLLM blocks; shared or prefix-cached blocks require
copy-on-write and are follow-up.

## Scheduler / worker flow

The MVP compacts at the end of a forward step, before the next step is scheduled:

```text
Step N forward
  -> R-KV selects retained entries
  -> worker compacts KV on GPU
  -> worker reports stepDropped

scheduler side
  -> totalDropped += stepDropped
  -> kvLen = logicalLen - totalDropped
  -> return tail blocks no longer needed by kvLen
  -> run normal vLLM allocation for Step N+1
  -> send kvLen + full block IDs if allocation changed

worker
  -> keep model positions based on logicalLen
  -> address KV using kvLen + current block IDs
  -> Step N+1 forward
```

The scheduler reclaims tail blocks only after receiving that step's compaction
result; that result is the reclaim handoff.

After compaction, normal vLLM allocation continues from `kvLen` resident
entries. If allocation changes the request's block IDs, the scheduler sends the
full current IDs before the next forward and the worker mirrors them.

## vLLM integration boundary

The MVP targets the pinned vLLM v0.25.1 classic GPU model runner. All pinned-vLLM
access stays in one compatibility adapter. The MVP needs vLLM integration
support for:

1. access the post-RoPE Q needed by the token-dropping algorithm;
2. keep model positions based on `logicalLen` while KV addressing uses `kvLen`;
3. reclaim unused tail blocks through vLLM's allocator;
4. keep token-dropping requests out of the local prefix cache;
5. terminate token-dropping requests after preemption or compaction failure.

## MVP scope

The MVP supports token dropping after a full prefill or decode step on a single
GPU with ordinary paged-attention KV. Chunked prefill is unsupported and rejected
at startup; speculative decoding is unsupported.

The MVP requires synchronous scheduling with one in-flight step, so compaction
is committed before reclaimed blocks can be reused.

Token-dropping requests bypass LMCache persistence and local prefix caching.
Requests without token dropping are unchanged: they continue to use the normal
LMCache and vLLM cache paths.

**Failure and preemption:** token-dropping requests are terminated; the MVP does
not recover compacted state.

## Follow-up

Initial persistence scoping includes an algorithm-defined `compression_spec` as
part of the cache key, so compacted KV is reused only with a compatible
compression configuration.

Other follow-up work includes async scheduling, shared-block copy-on-write,
broader KV layouts, multi-GPU support, and compaction performance optimizations.

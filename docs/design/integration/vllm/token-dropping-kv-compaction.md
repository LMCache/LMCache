# GPU-Native Token Dropping in LMCache

## Summary

LMCache already has a token-dropping SDK path for offline experiments. It runs
outside the vLLM worker, stages KV and Q through CPU memory, and needs external
orchestration around the serving engine.

M1 moves token dropping into the worker. Q and KV stay on GPU while R-KV scores
and compacts them; the LMCache server is not on the compaction path. Only
control state crosses the worker / scheduler boundary.

The goal is not just fewer KV entries. **Compaction should free real vLLM
blocks that other requests can reuse.**

One invariant drives the design: compaction changes the physical KV layout; it
does not shorten or renumber the request's logical token sequence.

## Architecture overview

```text
Prior SDK:  worker GPU -> CPU-staged KV/Q -> external SDK -> worker GPU
M1:         worker GPU -> R-KV + compaction in the worker -> worker GPU
```

The runtime keeps logical progress `L` separate from physical KV length `P`;
`D` tracks how many KV entries have been dropped.

```text
vLLM worker (GPU)
  post-RoPE Q + paged KV
          |
      R-KV policy
          | retained KV indices
          v
  compaction planner + executor
          | compact KV in place on GPU
          | report Δ = entries dropped this step
          |
          +---------------------------> vLLM scheduler + LMCache state
                                         D += Δ
                                         P = L - D
                                             |
                                      reclaim unused tail blocks
                                             |
                                      normal vLLM allocation
                                             |
          <--------------------------- P + block IDs if changed
          |
  worker adapter
      model positions use L
      KV reads / writes use P + block IDs
          |
      next forward
```

There is no separate algorithm process and no new RPC path.

## State and ownership

| State | Source of truth | Used for |
|---|---|---|
| Logical progress `L` | vLLM request | Model positions and request progress |
| Cumulative dropped count `D` | LMCache scheduler-side state | Deriving `P` |
| Physical KV length `P` | `L - D` (derived) | Allocation and worker KV addressing |
| Block IDs | vLLM scheduler / allocator | Physical blocks owned by the request |

The logical and physical lengths are identical before the first compaction.
Afterward they can be very different: the next token may be logical position
1,000 while only 256 KV entries remain in memory.

The scheduler is the only owner of cumulative compression state. The worker
reports `Δ` for the compaction it just ran; the scheduler updates `D` and
derives the new `P`. The worker receives `P`, not `D`, because it only needs
the current physical KV length.

Block ownership stays with vLLM. The worker may rewrite KV inside the blocks it
currently owns, but it does not independently allocate or free blocks.

## Compaction contract

A token-dropping policy chooses entries from the request's **current physical
KV sequence**. Its output is just the retained indices; LMCache owns the GPU
movement and vLLM integration.

For example:

```text
Current KV:       [A B C D | E F G H]
Keep indices:      1   3     4     7

After compaction: [B D E H | - - - -]
```

The planner uses the request's vLLM block IDs to turn those indices into
physical GPU copies. The executor applies the copies to K and V.

If the request is compacted again, indices refer to the already-compacted
sequence:

```text
[A B C D E F G H] -- keep [1,3,4,7] --> [B D E H]
[B D E H]         -- keep [1,3]     --> [D H]
```

No original-position map is needed for compaction itself.

This keeps policy and mechanism separate:

```text
R-KV:        when to compact, scoring, budget, recent Q
Compaction:  retained indices -> physical KV movement
```

M1 does not support shared KV blocks. A compacted request therefore owns the
blocks it rewrites and can compact them in place. Supporting shared blocks
later requires copy-on-write before compaction.

## Scheduler / worker flow

M1 compacts at the end of a forward step, before the next step is scheduled:

```text
Step N forward
  -> R-KV selects retained entries
  -> worker compacts KV on GPU
  -> worker reports Δ

scheduler
  -> D += Δ
  -> P = L - D
  -> return tail blocks no longer needed by P
  -> run normal vLLM allocation for Step N+1
  -> send P + updated block IDs if allocation changed

worker
  -> keep model positions based on L
  -> address KV using P + current block IDs
  -> Step N+1 forward
```

The scheduler frees tail blocks only after it receives the worker's compaction
result for that step. Until then, the worker may still be using the old block
allocation. That normal worker result is the handoff; M1 adds no extra ACK or
two-phase free protocol.

After compaction, vLLM allocates as if the request currently had `P` KV
entries. Future growth continues through normal vLLM allocation; token dropping
does not add a second allocator or its own reserve policy.

`P` tells the worker how much KV is live, but not which vLLM blocks hold it. If
allocation changes the block IDs, the scheduler sends the full current IDs
with `P` before the next forward.

The two control updates use the connector interface that already crosses this
boundary:

```text
worker    -- Δ --------------------> scheduler
scheduler -- P + block IDs if changed -> worker
```

M1 isolates pinned-vLLM internals in one version-gated adapter. The rest of the
feature depends on three vLLM capabilities only:

1. read the post-RoPE Q needed by the policy;
2. use a physical KV length that can differ from logical progress;
3. return unused tail blocks to vLLM's block pool.

This keeps vLLM-specific access out of the policy and compaction code, and lets
those private accesses be replaced by stable upstream APIs without changing
the state model above.

## Roadmap

### M1: live GPU-native token dropping

M1 is complete when a running request can compact KV, return the unused blocks
to vLLM, and continue decoding correctly without a CPU round trip.

M1 uses a synchronous hot path: requests co-batched with a compaction wait for
it before the next decode step. This keeps compaction, block reclaim, and the
next physical state ordered within one step. Overlapping compaction with later
steps is follow-up work.

M1 also assumes private request blocks, one in-flight scheduling step, the
classic vLLM GPU runner, a single-GPU path, and paged attention KV with one
physical KV entry per token.

Normal uncompressed LMCache store/load stays unchanged. Once a request has been
compacted, M1 does not persist that compacted state back to LMCache.[^restore]

[^restore]: Today, a cache hit has one length: it tells vLLM both how far the
request can advance logically and how many KV entries need to be allocated and
loaded. Compaction breaks that 1:1 relationship. For example, a 1,000-token
logical prefix may contain only 256 physical KV entries. Restoring it therefore
requires the cache-hit path to carry both lengths, advance model state by the
logical length, allocate and load by the physical length, and initialize
`D = L - P` for later steps. The KV copy itself is straightforward; the
missing piece is this new lookup/admission contract across
LMCache and vLLM, which deserves a separate design rather than being folded
into M1.

### M2: compacted persistence and broader runtime support

We have done initial scoping of LMCache's offload path for compacted KV.
Storing post-compaction KV is feasible; restore is the part that changes the
runtime contract. An initial direction is to use a compression-specific cache
identity, restore both `L` and `P`, and initialize `D = L - P` before decoding
continues. Because this changes LMCache lookup/admission semantics, store and
restore should be designed together rather than adding a write-only format to
M1.

Other follow-up work includes async scheduling / compaction overlap,
shared-block copy-on-write, broader KV layouts and multi-GPU support, and
profiling-driven compaction optimizations.

## Code / PR map

| Area | Change |
|---|---|
| Compaction planner | [PR #5352](https://github.com/LMCache/LMCache/pull/5352) |
| GPU compaction executor | [PR #5405](https://github.com/LMCache/LMCache/pull/5405) |
| R-KV policy + GPU Q capture | follow-up |
| Scheduler state + block reclaim | follow-up |
| Worker physical addressing | follow-up |

Each implementation PR should point back to this document for the end-to-end
state and ownership contract.

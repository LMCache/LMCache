# GPU-Native Token Dropping in LMCache (vLLM 0.25.1 MVP)

> **Status:** AIO correctness reference only. Draft / DO NOT MERGE.
> This document describes the S3 + S5 + AIO integration, not earlier designs.

## Goal and invariants

Run request-local token-dropping algorithms, including R-KV, in a vLLM worker
without CPU-staging KV or modifying upstream vLLM. Selection reads post-RoPE Q
and resident paged K/V on GPU. After completed compaction the scheduler returns
unused native KV blocks to the vLLM allocator.

Two independent frontiers must never be confused:

- **L (logical):** model/request progress and RoPE position. L never decreases
  when KV is dropped.
- **P (physical):** resident KV entries and addresses. P is the absolute count
  after compaction and grows from that count at the next allocation.

The scheduler is authoritative for native private block ownership and P. The
worker is authoritative only for transient Q history, per-request algorithm
instances, and completed GPU compaction observations. There is no shadow
allocator, original-position map, or cumulative dropped-token counter. The TD
scheduler owns resident frontiers, allocation references, and generation/lifecycle
state. The allocator adapter resolves that same resident state via the native
block pool bound by vLLM. The binding table weakly references both pool and
state, rejects conflicting live owners, and unbinds on connector shutdown.
The APC publication guard uses the same bound-pool scope. No request state
is shared across distinct block pools.

## Activation and supported execution

An engine must explicitly set the boolean extra config
`lmcache.experimental.token_dropping=true`; individual requests opt in with
`lmcache.token_drop={algorithm, config}` in their request KV-transfer params.
The algorithm is selected from the `lmcache.token_drop_algorithms` entry-point
group. Each request receives a separate algorithm instance and generation.

The pinned, fail-fast deployment contract is:

- vLLM **0.25.1**, classic **V1** GPU model runner (explicitly use
  `VLLM_USE_V2_MODEL_RUNNER=0` if vLLM would auto-select V2),
  synchronous scheduler, no microbatching, one GPU, one resolved ordinary
  full-attention KV group, FlashAttention paged K/V. Sliding/local attention,
  sink attention, quantized or asymmetric K/V layouts reject at admission.
  Standard FlashAttention block-stride KV indexing remains supported;
- full prefill and subsequent decode, with BF16/FP16 CUDA-resident KV;
  unsupported KV dtypes (including FP8) fail worker compatibility validation
  before any TD compaction;
- **Batch-local CUDA Graph compatibility.** Engine graph configuration is
  unchanged. Batches containing scheduled TD requests exclude runtime FULL
  replay, because it can bypass the Python Q capture hook. Native dispatch
  selects PIECEWISE if eligible, otherwise NONE. Normal-only batches use
  native dispatch unchanged;
- no chunked prefill, speculative decoding, cascade attention, async scheduler,
  multi-GPU, shared TD blocks, hybrid/quantized layouts, or TD KV offload/restore.

The experimental engine flag defaults to false: no TD worker initialization,
hook installation or TD-specific version restriction is required in that case.
Explicit TD requests are rejected if the flag is disabled. With the flag
enabled, only request-local opt-in activates TD.

Normal requests preserve ordinary LMCache and vLLM behavior. TD requests bypass
LMCache lookup/store and vLLM prefix-cache lookup/publication, and require
exclusively owned, unhashed native KV blocks.

## Integration and data flow

All vLLM-private interception lives in
`lmcache/integration/vllm/experimental/token_drop_adapter.py`. Production
LMCache connector changes remain small; no upstream vLLM source changes.

1. **Scheduler admission / ownership:** TD scheduler verifies supported modes,
   marks the request to skip prefix-cache lookup, and registers native private
   blocks. An allocator wrapper uses a request-local view of P instead of L
   while retaining the original request's logical position.
2. **S4 scoped worker addressing:** For TD rows after compaction, the worker
   computes CPU-packed physical positions and publishes the scheduler's block
   IDs to the native block table. Only the current block-table instance's
   `compute_slot_mapping` is temporarily intercepted during
   `GPUModelRunner._prepare_inputs`; a `finally` restores it. vLLM's own
   logical `positions` remain untouched for RoPE. KV-facing sequence lengths
   are updated with P after the native input preparation. Normal rows keep
   their ordinary physical slots and logical positions.
   The worker adapter also scopes a FULL exclusion to the dispatcher
   instance during `GPUModelRunner.execute_model` for batches with scheduled
   TD metadata. It preserves native mode constraints and batch descriptors,
   and restores the previous dispatcher method even if execution raises.
3. **S5 post-RoPE Q:** After normal RETRIEVE submission succeeds, a reversible
   instance-level `impl.forward` wrapper captures post-RoPE query tensors.
   Request IDs are resolved to explicit worker rows, not by assuming scheduler
   metadata order matches worker order. The wrapper is removed when the step
   completes, including compaction exceptions. The hook adapter owns restoration
   state; the worker supplies layer names and a Q callback.
4. **Per-request algorithm:** During forward the request-local implementation
   receives its selected observation rows via
   `observe_token_queries(layer_queries)`. The request decides whether to
   observe or compact with `should_observe_token_queries` and
   `should_compact_kv`. R-KV's window, buffer, budget and scoring policy are
   plugin settings, not a hardcoded LMCache `budget + 8` trigger. Algorithms
   receive a read-only KV interface, not the compaction operation; only the
   worker may mutate native KV after all selection validation succeeds.
5. **S7 compaction:** The algorithm supplies
   `select_kept_token_positions(kv_by_layer)`, mapping each layer to an
   integer `[kept_tokens, kv_heads]` tensor of source positions in the
   **current resident physical sequence**. All layers' selection shapes,
   counts, device/dtype and index ranges are preflight-validated before any
   destructive write. GPU gather/scatter compacts each head's K and V into
   the same private paged slots; the current CUDA stream is synchronized before
   publishing an absolute resident-P completion.
6. **S8/S9 reclaim and fencing:** Worker output carries
   `request_id -> (generation, absolute_P)`. The synchronous scheduler
   discards stale/finished/preempted generations, validates private ownership
   and all lengths, frees native tail blocks, updates the request's native
   block ID list, and then publishes the new P for the following scheduler
   step. Completion is a barrier; never reclaim blocks before GPU writes finish.

No LMCache server or CPU KV transport participates in TD compaction.

### Query capture

Some token dropping algorithms need post-RoPE Q tensors to determine which KV tokens to keep.

The existing [`save_kv_layer` hook](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/attention/kv_transfer_utils.py#L50-L59) handles KV transfer and does not receive Q. Query capture is a separate concern, and we don't want to extend the KV connector's data contract just to support token dropping.

Instead, we capture Q directly where it is already available, through a reversible, instance-level hook on [`impl.forward`](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/attention/attention.py#L828-L840).

## Failure, preemption, and cleanup

- Pre-write failures may skip compaction only while original KV is intact.
  Plugin observation/selection exceptions may have partially mutated internal
  history, so that request generation is **quarantined from further
  compaction**; decode continues using its last successfully resident KV.
  Incomplete multi-layer Q capture is treated the same way.
- Any GPU exception after a potentially destructive KV write must propagate:
  do **not** attempt request-local recovery from partially rewritten KV.
- Preemption discards TD physical-frontier overrides and worker query history,
  then allows vLLM to recompute from logical state. A rescheduled request in
  the same step preserves its newly allocated blocks, not stale ones.
- Finish/abort and request-ID reuse clear request-local state. Completions
  are generation-fenced; late reports cannot reclaim successor blocks. The
  scheduler also marks finished TD IDs in the next worker metadata, so a TD
  request aborted before its first forward cannot be misreported as a normal
  LMCache `finished_sending` completion.
- Batch dispatch can discover one active TD child in nested MultiConnector
  metadata, ignoring empty preceding children and rejecting multiple active
  TD children. This does not establish end-to-end MultiConnector TD support:
  pinned vLLM passes empty allocations to unselected children, and TD rejects
  an empty native allocation rather than compacting without ownership.

## Must-have correctness gates

A revision is a correctness-complete MVP reference only when all of these hold:

1. Valid TD requests retain the engine's graph configuration; scheduled
   TD batches never use runtime FULL. Native NONE/PIECEWISE fallback must
   actually execute the Q capture boundary, and normal-only graph replay
   remains native. Other unsupported configurations still reject before
   TD hooks. In particular, V2 runners, microbatching, non-full-attention KV specs, and
   explicit non-FlashAttention backends fail at scheduler admission. If vLLM
   chooses the attention backend automatically, worker metadata must be
   FlashAttention metadata before installing Q hooks or compacting KV.
2. Normal-only and mixed requests preserve normal KV ownership/slot mapping;
   TD uses L for RoPE, P for KV addressing; no prefix-cache publication or
   LMCache store/restore of compressed TD KV.
3. CUDA BF16/FP16 per-layer/per-head compaction matches an offline reference,
   including repeated compaction, block-boundary permutations, GPU write
   synchronization and native block reclaim.
4. Live serving covers TD-only, normal-only, mixed batch and forced
   preemption/resume on pinned vLLM 0.25.1, across all five engine graph
   configurations, with positive post-RoPE observations, completed compaction,
   native reclaim and subsequent decode. Normal-only FULL replay must be
   verified before and after TD where native eligibility permits it.
5. Pre-write fault injection confirms safe fallback/quarantine; after-write
   fault propagation, Q hook cleanup, stale-generation rejection and normal
   RETRIEVE failure paths are tested.
6. The pinned CI suite and ordinary-request isolation tests execute without
   skips. Measure normal-only latency and throughput with TD disabled and
   enabled against the frozen baseline, and report mixed/TD performance and
   memory overhead. Expanded deployment modes remain outside the MVP.

## Deferred scope

Full-decode CUDA Graph Q-capture, multi-GPU or multiple TD connector
ownership, shared-prefix CoW, compressed KV offload/persistence, chunked
prefill, speculative decoding, and async scheduling are deliberately excluded
from this MVP.

### Future prefix-cache compatibility

Prefix reuse is important for multi-turn serving. The MVP isolates TD from
both vLLM's native prefix cache and external LMCache KV reuse; normal requests
continue to use both. Future support needs to address two independent issues:

- **Out-of-place compaction:** Shared, published prefix blocks can be reused
  for reads. Before compaction mutates KV, gather only the retained entries
  directly into newly allocated private blocks rather than copying the
  entire prompt and then compacting it. The original published prefix
  remains intact and eligible for normal reuse and eviction. The scheduler
  owns allocation and reference-count changes; source references must remain
  live until GPU completion. Destination memory headroom, allocation failure,
  cancellation and preemption need explicit handling.
- **Q availability:** A cache hit skips computation of Q for cached tokens.
  Some algorithms need an observation window at the end of the current
  prompt for prefill-end compression. A future scheduler policy can cap
  the eligible hit length so that the required suffix is recomputed, subject
  to native block alignment. For a prompt of N tokens requiring the last W
  queries, an illustrative cap is max(0, N - max(1, W)). The worker still
  executes the uncached suffix normally. The observation requirement should
  come from the selected algorithm and its configuration rather than a
  hardcoded R-KV window.

The existing separation of selection from KV mutation, logical L from physical
P, and scheduler ownership from worker completion provides the foundation for
this extension. Neither copy-on-write nor observation-aware prefix lookup is
implemented in the MVP; they require a separate design and validation.

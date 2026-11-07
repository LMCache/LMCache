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
allocator, original-position map, or cumulative dropped-token counter.

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
- **CUDA Graph mode NONE or PIECEWISE**. PIECEWISE keeps attention forward eager
  while other compatible computation can remain graphed/compiled;
- **no FULL, FULL_DECODE_ONLY, or FULL_AND_PIECEWISE**, because replay may bypass
  Python post-RoPE Q hooks. Unsupported modes fail admission before TD
  allocator/prefix-publication hooks install;
- no chunked prefill, speculative decoding, cascade attention, async scheduler,
  multi-GPU, shared TD blocks, hybrid/quantized layouts, or TD KV offload/restore.

Normal requests preserve ordinary LMCache and vLLM behavior. TD requests bypass
LMCache lookup/store and vLLM prefix-cache lookup/publication, and require
exclusively owned, unhashed native KV blocks.

## Integration and data flow

All vLLM-private interception lives in
`lmcache/integration/vllm/experimental/token_drop_adapter.py`. Production
LMCache connector changes remain small; nothing is patched in upstream vLLM.

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
3. **S5 post-RoPE Q:** After normal RETRIEVE submission succeeds, a reversible
   instance-level `impl.forward` wrapper captures post-RoPE query tensors.
   Request IDs are resolved to explicit worker rows, not by assuming scheduler
   metadata order matches worker order. The wrapper is removed when the step
   completes, including compaction exceptions.
4. **Per-request algorithm:** During forward the request-local implementation
   receives its selected observation rows via
   `observe_token_queries(layer_queries)`. The request decides whether to
   observe or compact with `should_observe_token_queries` and
   `should_compact_kv`. R-KV's window, buffer, budget and scoring policy are
   plugin settings, not a hardcoded LMCache `budget + 8` trigger.
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
- A MultiConnector may contain one active TD child. Empty preceding children
  are ignored; multiple active TD children are unsupported and fail closed.

## Must-have correctness gates

A revision is a correctness-complete MVP reference only when all of these hold:

1. NONE/PIECEWISE admission accepts valid TD requests; all FULL-decode graph
   modes and other unsupported configurations reject before TD hooks. In
   particular, V2 runners, microbatching, non-full-attention KV specs, and
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
   preemption/resume on pinned vLLM 0.25.1, including PIECEWISE and no missing
   post-RoPE observations.
5. Pre-write fault injection confirms safe fallback/quarantine; after-write
   fault propagation, Q hook cleanup, stale-generation rejection and normal
   RETRIEVE failure paths are tested.
6. The CI pipeline and full ordinary-request isolation tests have documented
   PASS, or a concrete diagnosed blocker. Performance parity and expanded
   deployment modes are separate future review gates, not claimed here.

## Deferred scope

Full-decode CUDA Graph Q-capture, dynamic per-batch graph-mode switching,
multi-GPU or multiple TD connector ownership, shared-prefix COW, compressed KV
offload/persistence, chunked prefill, speculative decoding, async scheduling,
and performance optimization are deliberately excluded from this MVP.

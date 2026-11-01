# MP-mode preemption

When the vLLM scheduler runs out of KV blocks it **preempts** the most recently
admitted running request: every one of its blocks is freed and the request goes
back to the waiting queue to be recomputed from scratch. This document covers
what the MP connector must do about that, and how it is tested.

## What vLLM does

| Event | Behaviour |
|---|---|
| Preempt | Blocks freed immediately, `status = PREEMPTED`, `num_computed_tokens = 0`, request prepended to the waiting queue. Its id appears in `SchedulerOutput.preempted_req_ids` for that step. |
| Victim | FCFS: `running[-1]`. One request can be preempted many times. |
| Block reuse | Freed blocks can be handed to another request inside the *same* `schedule()` call. Nothing waits for a connector store that may still be reading them. |
| Resume | The waiting loop recomputes the local prefix-cache hit over **all** tokens (prompt + generated so far) and calls `get_num_new_matched_tokens()` with `status == PREEMPTED`. |

## The two problems

### 1. Resumed requests recomputed everything

`get_num_new_matched_tokens()` returned `(0, False)` for any `PREEMPTED`
request, so a resumed request never loaded from LMCache. Under the memory
pressure that causes preemption vLLM's own prefix cache is usually evicted
too, so the whole prefix was recomputed.

The fix is to delete that early return. A resumed request already gets a fresh
tracker (`_get_or_create_request_tracker` drops the previous generation's), so
the ordinary path does the right thing: look up over all tokens, retrieve into
the new blocks, and resume the store cursor from the hit.

One related change is needed. The scheduler polls this method repeatedly for
the same request whenever `allocate_slots()` fails -- common under exactly this
memory pressure -- and the adapter caches the lookup answer. Accumulating the
hit into `num_stored_tokens` therefore double-counted it and silently skipped
storing the chunks that followed. The hit is now assigned, not accumulated:
LMCache holds exactly `ret` tokens.

### 2. A store could read blocks that had been reused

On the LMCache-driven path the server copies KV blocks on its own stream after
the forward pass that produced them, and the engine never waits for that copy.
If the request is preempted, vLLM frees those blocks and may hand them to
another request in the same step, whose forward pass overwrites blocks the
server is still reading. The store then commits the wrong KV under the
preempted request's keys -- which looks fine until a later request loads that
prefix.

`TransferContext.flush_inflight_stores()` already exists for this, and the
worker adapter already calls it on steps the scheduler flags as having
preempted or resumed requests. It was implemented for the async engine-driven
context and left as `pass` for the LMCache-driven one, which is the hole. It
now tracks its in-flight store futures and waits for them. A timeout is logged
rather than raised, so a slow server degrades to a possibly stale store rather
than a crashed engine.

## Observability

vLLM cannot report a resume-load. `PrefixCacheStats.record()` routes hits for
any request with `num_preemptions > 0` into `preempted_hits`, which is never
exported or logged, and `prefill_stats` is gated the same way ("track first
scheduled prefill, not post-preemption repeat prefills"). So
`vllm:external_prefix_cache_hits_total` and `vllm:prompt_tokens_cached` both
read zero for exactly the requests this feature serves.

The connector therefore logs each resume itself, where the decision is made:

```
<resume-load> req=<id> apc=<n> lmcache=<n> load=<n>
```

`apc` is what vLLM's own prefix cache still held, `lmcache` what LMCache holds,
and `load` what LMCache contributes beyond it. The scheduler polls the method
repeatedly per admission, so aggregate per request id rather than summing
lines.

## Testing

`.buildkite/k3_tests/multiprocess/scripts/run-preemption-correctness.sh`, a
ladder of agreements over real ShareGPT traffic. ShareGPT has no correct
answers, so each rung is checked for *agreement* with the run below it rather
than for accuracy. The reference is the simplest execution path there is:
plain vLLM at a concurrency low enough that preemption is provably impossible.

| Rung | Server | Concurrency | Adds |
|---|---|---|---|
| ref | plain vLLM | low | nothing -- the ground truth |
| A | plain vLLM | high | preemption |
| B | LMCache | low | the connector, cold cache (writes only) |
| C | LMCache | low | the connector, warm cache (reads) |
| D | LMCache | high | preemption over both |

Both concurrencies are derived from the KV pool, which the pipeline pins with
`NUM_GPU_BLOCKS_OVERRIDE` so it does not depend on the GPU: preemption is
guaranteed when the requests in flight cannot all be resident, and impossible
when the concurrent prompts plus their `max_tokens` fit. Both are asserted
from `vllm:num_preemptions_total`, so a mis-sized pool fails the run instead of
making it vacuous. Rung D additionally fails if no request resumed with a
load, so it cannot pass while the feature is inert.

Exact comparison requires `VLLM_BATCH_INVARIANT=1`, a pinned attention backend
and a model whose kernels vLLM covers (RMSNorm families). The ref-vs-A rung is
what proves the oracle is valid before LMCache is added: if plain vLLM cannot
reproduce its own answers across the two concurrencies, nothing measured above
can be attributed to the connector.

Answers are compared by `.buildkite/correctness/compare_files.py`, keyed by
request id, and kept as files so a reviewer can read them. That matters
because ShareGPT's topic diversity makes failures legible: an answer about
Fiji that starts explaining bash redirection is cross-request contamination,
not numerical drift.

## Async scheduling

`--async-scheduling` is the second axis of the matrix. With a consumer-role
connector vLLM then defers block frees to the end of the in-flight step, so
preemption takes a different path, but nothing above needs to change: the
low-concurrency rungs still see exactly zero preemptions, so the bound that
makes them a valid reference survives.

Measured on Qwen3-14B, 99 ShareGPT requests, a 1024-block pool,
lmcache-driven transfer:

| Rung | sync preemptions | async preemptions |
|---|---|---|
| ref (c3) | 0 | 0 |
| A, plain vLLM (c40) | 64 | 60 |
| B, C (c3) | 0 | 0 |
| D, LMCache (c40) | 59 | 308 |

Every rung reproduces its reference in both modes, and the two references
agree with each other, so scheduling mode does not change the answers.

The difference is the preemption count on rung D: LMCache preempts about as
often as the baseline under sync scheduling and roughly five times as often
under async. That is the vLLM scheduler walking down the running list
preempting victims whose blocks cannot be reused yet because their frees are
deferred, so one shortfall cascades. It is fixed upstream in
vllm-project/vllm#49675; check the pinned vLLM against that commit when
reading these numbers. Wall time was not materially affected here (238 s
against the baseline's 232 s), and resumed requests recovered more KV
precisely because there were more of them: 74 requests and 118k tokens
against 32 and 43k under sync, with vLLM's own prefix cache contributing
nothing in either case.

## Not covered

- Lazy offload under preemption.

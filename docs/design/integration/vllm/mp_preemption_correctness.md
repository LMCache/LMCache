# MP-mode preemption: correctness design and test suite

Status: implemented. This document defines what "correct" means for
`LMCacheMPConnector` under vLLM scheduler preemption, lists the holes found in
the code before this work, and describes the three-layer test suite that now
guards the behaviour. Section 8 records what the suite found and what changed.

Where things live:

| Layer | Path |
|---|---|
| L1 model-based scheduler tests (no GPU) | `tests/v1/mp_preemption/` |
| L2 GPU block-reuse race test | `tests/v1/multiprocess/test_preemption_block_reuse_gpu.py` |
| L3 end-to-end ladder (T0-T3) | `.buildkite/k3_tests/multiprocess/scripts/preemption_correctness.py`, `run-preemption-correctness.sh`; k3 test name `preemption_correctness` |
| Connector / tracker / worker unit tests | `tests/v1/test_mp_connector_preemption.py`, `tests/v1/test_vllm_mp_adapter.py` |

Code references are to `lmcache/integration/vllm/` unless noted, and to the
vLLM checkout at `vllm/v1/core/sched/scheduler.py` for scheduler facts.

---

## 1. Ground truth: what vLLM does on preemption

These are the scheduler behaviours every property below is stated against.

| Event | Scheduler behaviour | Where |
|---|---|---|
| Preempt | `_free_request_blocks(req)` (immediate unless `defer_block_free`), `status = PREEMPTED`, `num_computed_tokens = 0`, `num_preemptions += 1`, request **prepended** to `waiting`, id added to `reset_preempted_req_ids` | `_preempt_request`, scheduler.py:1478 |
| Victim | FCFS: `running[-1]`; priority: max `(priority, arrival)`. A request can be preempted many times (the debug log shows one request preempted 6×) | scheduler.py:749-786 |
| `preempted_req_ids` | Set on the `SchedulerOutput` of the **same** `schedule()` call, cleared in `_update_after_schedule` | scheduler.py:1417, 1569 |
| Block reuse | Freed blocks are back in the pool inside the same `schedule()` call; the waiting loop that runs next can hand them to another request (or to the same request when it resumes in the same step) | `_free_request_blocks`, scheduler.py:2629 |
| Deferred free | Only when `max_concurrent_batches > 1` (async scheduling / PP) **and** `kv_role` is a consumer. k3 CI runs `--no-async-scheduling`, so frees are immediate | scheduler.py:166-172 |
| Resume: lookup | Waiting loop treats `PREEMPTED` like `WAITING`: `_get_local_prefix_cache_hit` (APC over **all** tokens incl. generated) then `connector.get_num_new_matched_tokens(request, block_aligned_local_hit)` with `request.status == PREEMPTED` | scheduler.py:926-940 |
| Resume: alloc | `allocate_slots` then `connector.update_state_after_alloc(request, get_blocks(req) = ALL blocks, num_external)` | scheduler.py:1185-1190 |
| Resume: output | Request is in `scheduled_cached_reqs` with `resumed_req_ids ∋ id` and `new_block_ids` = **all** blocks, `num_computed_tokens` = local + external hit | `_make_cached_request_data`, scheduler.py:1620-1665 |
| Resume: async load | `WAITING_FOR_REMOTE_KVS` → on completion status becomes `PREEMPTED` again (not `WAITING`), re-admitted via the `else` branch with `num_computed_tokens` preset; `get_num_new_matched_tokens` is **not** called again; `update_state_after_alloc` **is** called a second time with all blocks | scheduler.py:2992, 1009-1016 |
| Failed load | `_update_requests_with_invalid_blocks` lowers `num_computed_tokens` to the longest valid prefix (or 0 for hybrid) | scheduler.py:3046 |
| Worker hook | `connector.handle_preemptions(metadata)` at the top of `execute_model`, before `_update_states` and the forward | gpu_model_runner.py:4218 |
| Forced preemption | `reset_prefix_cache` preempts every running request with `drop_stale_output=True` | scheduler.py:2713-2740 |

Two consequences drive everything below:

1. **A preempted request's blocks can be overwritten by the very next forward
   pass.** Nothing in vLLM waits for a connector store that is still reading
   them.
2. **The connector sees the resume as a cached request whose `new_block_ids`
   is the full list**, not a delta, and whose `num_computed_tokens` may be
   non-zero (APC hit on the prompt + generated prefix).

---

## 2. In-process vs MP: what each does today

| Concern | In-process (`vllm_v1_adapter.py`) | MP (`lmcache_mp_connector.py`) |
|---|---|---|
| Lookup on resume | Uses `request.all_token_ids` (covers generated tokens); lookup made idempotent so it can be polled across steps without `update_state_after_alloc` (lines 1372-1420) | Returns `(0, False)` when `status == PREEMPTED` (line 1018). **No LMCache load on resume.** |
| Tracker on resume | `RequestTracker.update(preempted=True)`: block ids replaced wholesale, `num_saved_tokens = lmcache_cached_tokens`, `token_ids = all_token_ids[:needed]` (lines 240-257) | `_get_or_create_request_tracker` **drops** the tracker and creates a fresh one when `status == PREEMPTED` and state ≠ `PREFETCHING` (lines 1610-1622). Fresh tracker: `num_stored_tokens = 0`, `num_vllm_hit_tokens = 0`, `num_lmcache_hit_tokens = 0` |
| Consistency check | Asserts `num_computed_tokens == max(lmcache, vllm)` minus the full-hit −1 (lines 1765-1790); truncates tracker on `invalid_blocks` rollback (1795-1815) | None |
| Block ids for resumed | From `CachedRequestData.new_block_ids` (all blocks) | From `update_state_after_alloc` (all blocks); `_process_cached_requests` skips the append for `resumed_req_ids` (1466-1468) so there is no double count |
| Worker-side race | Save is synchronous inside `wait_for_save` (or layerwise); nothing outlives the step | `need_flush_before_forward = resumed or preempted ids` (1228) → `handle_preemptions` → `transfer_ctx.flush_inflight_stores()`. Implemented only by `AsyncEngineDrivenTransferContext` (async_engine_driven.py:350). `LMCacheDrivenTransferContext.flush_inflight_stores` is `pass` (worker_transfer.py:648) |
| Existing tests | `tests/v1/test_decode_save_and_preemption.py` tests **re-implemented copies** of the slicing/assertion logic, not the adapter | `tests/v1/test_vllm_mp_adapter.py` has no preemption test; k3 `lm_eval_preemption` checks gsm8k score drift < 0.05 and that `<preempted>` appears in the log |

---

## 3. Holes found (hypotheses the proof must confirm or refute)

Ordered by severity. "Silent" means the preempted request itself looks fine.

### H1 (silent, corrupting) — lmcache-driven store races block reuse

Timeline, lmcache-driven path, request R preempted at schedule N+1:

```
wait_for_save(N)       worker: submit_store(R, blocks B, event E_N)   → server queues D2H from B after E_N
schedule(N+1)          scheduler: preempt R → free B → allocate B to R'  (same call)
execute_model(N+1)     worker: handle_preemptions → flush_inflight_stores() == pass
forward(N+1)           vLLM stream writes R' KV into B
server (any time)      D2H copies B → commits under R's chunk keys
```

The server's copy is on another process's stream; vLLM's forward is not
ordered after it. If the copy lands after the forward, R's chunk keys now hold
R' data. R finishes correctly (it recomputes). **Any later request sharing R's
prefix loads garbage.** This is exactly the "looks correct, garbled later"
failure mode. It cannot be caught by comparing the preempted request's output;
only a replay pass against the cache can.

Why the finished-request path does not have this bug: `request_finished`
returns `True` so blocks are held until `get_finished` reports the store done.
Preemption bypasses that (`_free_request_blocks` consults no connector hook
other than `has_pending_block_frees`, which the MP connector does not
implement).

### H2 (coverage, not corruption) — resumed tracker forgets the APC hit

Fresh tracker has `num_vllm_hit_tokens = 0`, so `GetStoreMetadata` computes
`computed_tokens = num_scheduled_tokens + 0` while vLLM actually has
`apc_hit + num_scheduled_tokens` computed. Stores lag by `apc_hit` tokens for
the rest of the request; the tail is never stored. Data stored is still
correct (blocks `[0, k)` hold tokens `[0, k·bs)`). Also `num_stored_tokens = 0`
re-stores chunks already present (idempotent by key; wasted bandwidth).

### H3 (feature gap) — no LMCache load on resume

Every resume recomputes everything beyond vLLM's APC hit, and under the
memory pressure that causes preemption APC is usually evicted. This is the
feature to build; §5 defines its acceptance oracle (P5) so it is written
before the implementation.

### H4 (leak risk once H3 is implemented) — lock/session accounting across regeneration

Today the fresh tracker never takes lookup locks, so nothing leaks. Once
resume issues a second lookup for the same `request_id`, the locks for the
second generation must be freed on admission (`free_lookup_locks`), on
retrieve completion, or on abort (`request_finished` → `cleanup_lookup_result`
/ `end_session`). The adapter caches lookup results by `request_id`
(`_finished_lookup_results`); a stale first-generation result must not be
served to the second generation.

### H5 (unverified) — lazy offload holds GPU block ids across a preemption

`LazyOffloadPendingStore` keeps `(block_id → block_hash)` and later calls
`_gpu_block_pool.touch()` on those ids and compares hashes. After preemption
the ids are free or reassigned; the hash check is the only guard. Touching a
free block is a block-pool invariant question, not just a connector one.

### H6 (unverified) — same-step preempt + resume, repeated preemption

Tracker is dropped and recreated once per generation. `_report_block_allocation_deltas`
computes `start_token = (total_blocks - num_new_blocks) * bs`, which is 0 for a
resumed request (new == all). Needs a scenario, not a hypothesis.

### Out of scope for this proof

Speculative decoding (`num_scheduled_tokens` includes drafts), hybrid/Mamba
groups (recompute-in-full on invalid blocks), P/D disaggregation.

---

## 4. Correctness properties (proof obligations)

Stated so each is mechanically checkable. `bs_g` is the block size of KV
group `g`, `C` the LMCache chunk size, `gen(r)` = `request.num_preemptions`.

**P0 Liveness and hygiene.** Every request reaches a finished state; no
request stays in `WAITING_FOR_REMOTE_KVS` for more than the modelled load
latency; `connector.request_trackers` is empty when all requests are done;
`end_session` is called exactly once per request; no assertion fires.

**P1 Structural op consistency.** For every `LoadStoreOp` emitted by
`build_connector_meta` at step `t` for request `r` over tokens `[s, e)`:

- P1a `op.block_ids[g] == kv_cache_manager.get_block_ids(r)[g][s/bs_g : e/bs_g]`
  evaluated at step `t`, for every group `g`.
- P1b STORE: `e ≤ num_computed_tokens(r) after step t`; `s == tracker.num_stored_tokens`
  before the op; per `(r, gen(r))` the store ranges are disjoint and
  increasing; `e - s` is a multiple of `C`.
- P1c RETRIEVE: `s == align_down(vllm_hit, C)`, `e == lmcache_hit`,
  `e % C == 0`, `e ≤ len(all_token_ids)`, `skip_first_n_tokens == vllm_hit - s`,
  and the op is emitted exactly once per generation.
- P1d Tracker/block-list agreement: after every step,
  `tracker.allocated_block_ids[g] == kv_cache_manager.get_block_ids(r)[g]`
  for every tracked running request.

**P2 No reads of reused blocks (temporal).** For every STORE op emitted at
step `t` covering GPU blocks `B`, no block in `B` changes owner (freed, or
allocated to a different request or a different generation of `r`) before the
store is *complete*. Completion is path-specific and must be pluggable in the
test:

- engine-driven sync: complete at `wait_for_save(t)`.
- engine-driven async: complete at the first later step whose metadata has
  `need_flush_before_forward == True`, or at the store's own completion.
- lmcache-driven: complete when the server's completion event fires; vLLM
  never waits on it. P2 can therefore only hold if the fix either defers the
  free (scheduler side) or blocks in `handle_preemptions` (worker side). The
  test encodes "complete = never before the next forward" for this path and
  is **expected to fail today** (H1).

**P3 Cache content fidelity.** For every chunk key `k` that a RETRIEVE later
consumes, the bytes committed under `k` were produced from blocks that held
exactly the tokens `k` names, and P2 held for the store that produced them.
In L1 this is derived (P1 ∧ P2 ⇒ P3 in the model); in L3 it is measured by
the replay pass.

**P4 Output fidelity.** With temperature 0 and deterministic kernels, the
token ids of every request under (LMCache MP + forced preemption) equal the
no-LMCache baseline, and the token ids of a *replay* pass (LMCache hits, no
preemption) also equal the baseline.

**P5 Resume-load correctness (acceptance for H3).** When a resumed request
has LMCache hit `h` and block-aligned APC hit `a`:

- `get_num_new_matched_tokens` returns `max(0, h - a)` minus 1 iff
  `h == len(all_token_ids)`; repeated calls without an intervening
  `update_state_after_alloc` return the same value and issue no second lookup.
- After admission `request.num_computed_tokens == max(a, h)` (minus the
  full-hit 1), a RETRIEVE op satisfying P1c is emitted for `[align_down(a), h)`
  into the **new** blocks, and `tracker.num_stored_tokens == h` so the next
  STORE starts at `h`.
- A second lookup for the same `request_id` after `cleanup_lookup_result`
  never returns the first generation's cached count.
- Locks taken by the second lookup are released exactly once (admission, retrieve,
  or abort).
- Async-load resume follows `WAITING_FOR_REMOTE_KVS → PREEMPTED → RUNNING`
  with the second `update_state_after_alloc` appending only new blocks.
- Failed resume load (`invalid_block_ids`) drops to local recompute without a
  stuck request and without re-reporting a hit (the existing
  `BYPASS_LMCACHE` path).

---

## 5. Test architecture

Three layers; L1 is the proof, L2 and L3 are the physical confirmations.

### L1 — model-based scheduler test (no GPU, seconds, unit CI)

Drive the **real** vLLM `Scheduler` and the **real** `LMCacheMPConnector`
in `SCHEDULER` role against a simulated LMCache server/worker that doubles as
the oracle.

Components:

- `harness.py`: port of vLLM's `tests/v1/core/utils.py`
  `create_scheduler` / `create_requests` and
  `tests/v1/kv_connector/unit/utils.py` `create_model_runner_output`
  (LMCache tests cannot import vLLM's `tests` package). `kv_connector="LMCacheMPConnector"`,
  tiny `num_blocks` (7–64), `block_size` 2–16, `max_num_seqs` high, APC
  on/off, `async_scheduling` on/off, `scheduling_policy` fcfs/priority.
- `FakeSchedulerAdapter`: monkeypatched in place of
  `lmcache_mp_connector.LMCacheMPSchedulerAdapter` (constructor does a server
  handshake). Implements the nine methods the connector calls:
  `lmcache_tokens_per_chunk`, `maybe_submit_lookup_request`,
  `check_lookup_result` (configurable defer count, like vLLM's
  `MockKVConnector.num_defers_before_matching`), `cleanup_lookup_result`,
  `free_lookup_locks`, `end_session`, `report_block_allocations`,
  `update_pending_store_count`, `is_healthy`. Lookup answer = longest stored
  chunk prefix of the token ids in the cache model. Keeps a per-key lock
  refcount.
- `CacheModel` (the oracle): owner map `block_id → (req_id, gen, token_range)`
  rebuilt every step from the `SchedulerOutput` and from
  `kv_cache_manager.get_block_ids`; in-flight STORE ops with configurable
  latency `L ∈ {0, 1, 2}` steps and a pluggable completion rule (§4 P2);
  chunk table `key → content | POISONED`. A store whose blocks changed owner
  before completion commits `POISONED`. A RETRIEVE of a missing or POISONED
  chunk fails the test. Retrieve completion is reported back through
  `ModelRunnerOutput.kv_connector_output.finished_recving` after a
  configurable delay, and failures through `invalid_block_ids`.
- Step loop: `schedule()` → assert P1 against `kv_cache_manager` → feed
  metadata to `CacheModel` → `update_from_output(create_model_runner_output(...))`
  with a deterministic sampled token `f(req, step)` so `all_token_ids`
  differ per request but are reproducible; run until all requests finish; then
  assert P0.

Scripted scenarios (each a parametrized test, both transfer paths, APC on/off,
lazy offload on/off, async scheduling on/off where the scheduler supports it):

| # | Scenario | Exercises |
|---|---|---|
| S1 | Preempt during chunked prefill, before the first chunk boundary | tracker regeneration with nothing stored |
| S2 | Preempt during decode after ≥ 1 chunk stored; resume with APC hit `0 < a < stored` | H2, P1b `s == num_stored_tokens` |
| S3 | Preempt and resume in the **same** `schedule()` (a running request finishes in the same step) | H6, block ids of the new generation overlapping the old set |
| S4 | Same request preempted 3× | tracker regeneration idempotence, `_report_block_allocation_deltas` |
| S5 | Store submitted at `t`, victim preempted at `t+1`, its blocks handed to a new request at `t+1` | **H1 / P2** — expected red for lmcache-driven with `L ≥ 1` |
| S6 | Resume with full LMCache hit (`h == len(all_token_ids)`) | the −1 rule (P5) |
| S7 | Resume with async load, then `invalid_block_ids` for the loaded blocks | recompute path, no stuck request (P0, P5) |
| S8 | Client aborts a request while it is `PREEMPTED` | `request_finished` with already-freed blocks, session/lock cleanup (P0, H4) |
| S9 | Lazy offload pending store references blocks that are then freed by preemption and reassigned | H5 |
| S10 | `reset_prefix_cache()` preempting every running request | forced path, `drop_stale_output` |
| S11 | Priority policy: victim is a request already scheduled this step | scheduler restores its token budget; connector must not see it as cached |
| S12 | Lookup returns `None` for k steps on a resumed request (deferred lookup) | idempotence of repeated `get_num_new_matched_tokens` (P5) |

Plus a seeded random workload (arrival times, prompt lengths spanning 1–6
chunks with shared prefixes, `max_tokens` 1–40, pool of 16–64 blocks) that
runs for a few thousand steps per seed and asserts P0–P2 continuously. This
is the closest thing to a proof: every reachable interleaving of the
scheduler's preemption logic against the connector's state machine is checked
against an independent model of what the cache should contain.

Expected results **today**: P0 and P1 should pass. P2 fails for
lmcache-driven with `L ≥ 1` (H1). P5 scenarios are skipped/xfail until H3 is
implemented, then flipped to required.

### L2 — worker-side race test (1 GPU, one process, real server)

Physically confirms P2 for both transfer paths without vLLM:

1. Start `lmcache server`; build an `LMCacheMPWorkerAdapter` with a small
   real KV cache (1 layer, few blocks) and register it.
2. Fill blocks `B` with pattern A on the compute stream; `submit_store(B, event)`.
3. Immediately overwrite `B` with pattern A′ on the compute stream (the
   "next forward").
4. Retrieve the key into fresh blocks; assert bytes == A.

Variants: engine-driven async with and without `handle_preemptions(True)`
between steps 2 and 3 (must pass with, documents behaviour without);
lmcache-driven (expected to fail today, becomes the acceptance test for the
H1 fix regardless of whether the fix is a worker-side wait or a scheduler-side
deferred free). Reuse fixtures from `tests/v1/test_vllm_mp_adapter.py` and
`tests/v1/multiprocess/test_engine_driven_transfer.py`.

### L3 — end-to-end differential (1–2 GPUs, k3)

Extends `.buildkite/k3_tests/multiprocess` `lm_eval_preemption`, which today
only checks gsm8k score drift and greps the connector's own `<preempted>`
log line.

Forcing preemption deterministically:

- `--num-gpu-blocks-override N` instead of `--gpu-memory-utilization 0.4`
  (pool size independent of the GPU model).
- Workload: `ignore_eos: true`, fixed `max_tokens`, temperature 0, prompts of
  several chunks with **shared prefixes across requests** (so a poisoned
  chunk is re-hit), concurrency ≥ `max_num_seqs`. Total demand
  `Σ(prompt + max_tokens)` chosen to exceed `N · block_size` by ≥ 2×, so
  preemption is guaranteed rather than probabilistic.
- Assert preemption count from vLLM's `/metrics`
  `vllm:num_preemptions_total` (the connector under test must not be its
  own witness).

Passes, all with the same request set and client-supplied `request_id`:

| Pass | Server | Concurrency | Purpose |
|---|---|---|---|
| 0 | vLLM, no LMCache | same as pass 1 | baseline token ids |
| 1 | vLLM + LMCache MP, cold cache | high (preemption-heavy) | P4 for preempted requests; populates cache under stress |
| 2 | same server as 1, warm | 1–4 (no preemption) | **P3/P4 replay**: everything loads from the cache written in pass 1 |

Pass 2 must also assert a high hit ratio via `cached_token_stats` in
`kv_transfer_params` (the connector already returns
`num_lmcache_cached_tokens`), otherwise the pass proves nothing about cache
content.

Oracle: compare **token ids**, not text (use `return_token_ids` or
`logprobs` with `top_logprobs=0`). With `VLLM_BATCH_INVARIANT=1` (already the
k3 default) require 100 % equality. Fallback when batch invariance is not
available on the target GPU or attention backend: run pass 0 twice at the
two concurrencies to measure the A/A noise floor, require the LMCache mismatch
rate ≤ that floor, and classify each first-divergence position using the
baseline's top-2 logprobs: a near tie (gap < δ) is numerical noise; anything
else is corruption and fails the test.

Axes to add to the matrix: `--async-scheduling` (k3 currently forces it off),
`lmcache_driven` / `engine_driven`, lazy offload on/off.

Fixes to carry over from the debug repo's scripts
(`ApostaC/yihua-lmcache-debug/preemption`): `compare_files.py` compares text
and keys on the server-assigned `chatcmpl-` id; the `*_length.txt` files are
opened in append mode and never cleared, so their counts accumulate across
runs and are not comparable (49 vs 127 in the checked-in samples).

---

## 6. Order of work

1. L1 harness, `FakeSchedulerAdapter`, `CacheModel`, P0–P2 assertions, S1–S11
   and the random workload. Run against current `main`; record which
   assertions fail. Expected: only P2 on lmcache-driven (H1).
2. L2 GPU race test, both paths.
3. L3 script changes: block override, replay pass, metrics-based preemption
   count, token-id compare, hit-ratio assertion.
4. Decide the H1 fix (worker-side wait in `handle_preemptions` for
   lmcache-driven vs scheduler-side deferral via `has_pending_block_frees` /
   a connector-held free list) and pin P2's completion rule to it.
5. Implement resume-load (H3) with S6, S7, S12 and the P5 assertions already
   in place; fix H2 in the same change (carry `num_vllm_hit_tokens` and
   `num_stored_tokens` across the generation instead of zeroing them).

## 7. Open questions

- Target vLLM: `main` only, or also the `0180` / `0201` connector snapshots?
  L1 depends on `SchedulerOutput.preempted_req_ids` and
  `CachedRequestData.resumed_req_ids`, which the snapshots may not have.
- Is `VLLM_BATCH_INVARIANT=1` honoured on every GPU and attention backend we
  run L3 on? If not, the fallback oracle in §5 L3 is required, not optional.
- H1 fix location (worker wait vs scheduler deferral) changes P2's completion
  rule; the test is written with the rule pluggable so this can be decided
  after L1 is green on everything else.

---

## 8. Results and changes

### 8.1 What the suite found before any connector change

Running L1 against the unmodified connector (vLLM 0.28.1) confirmed the
holes above and found two more:

| Id | Finding | Property that caught it |
|---|---|---|
| H1 | lmcache-driven stores read blocks vLLM may already have reallocated | P2 (`SERVER_ASYNC` policy, negative control S5) |
| H3 | no LMCache load on resume | P5 (`resumed_with_retrieve` empty in every seed) |
| **H7** (new) | `get_num_new_matched_tokens` added the cached lookup hit to `num_stored_tokens` on **every** poll. vLLM re-polls whenever `allocate_slots` fails, which is exactly the memory-pressure situation that causes preemption, so the chunks right after the hit were silently never stored. | P1b (`start 16 != cursor 8`) |
| **H8** (new) | A request that finishes while `WAITING_FOR_REMOTE_KVS` (abort, or a failed async load under `kv_load_failure_policy=fail`) is freed by vLLM on `finished_recving`, but `request_finished` also returned `True`, so the worker later reported `finished_sending` for an id vLLM no longer tracked and the scheduler asserted. | P0 (S7 with `fail` policy) |
| H4 | prefetch read locks leaked when a request left with an unconsumed lookup hit | lock ledger (S8, random aborts) |

Everything else (P1a block-id agreement, P1c retrieve geometry, P1d tracker
block lists, the −1 full-hit rule) held on the unmodified code.

### 8.2 Connector changes

`lmcache/integration/vllm/lmcache_mp_connector.py`,
`lmcache_mp_metadata.py`, `vllm_multi_process_adapter.py`:

- **Resume-load (H3).** `get_num_new_matched_tokens` no longer returns
  `(0, False)` for `PREEMPTED` requests. The previous generation's tracker
  is dropped, a fresh lookup runs over prompt + generated tokens, and the
  ordinary `WAITING_FOR_LOAD → RETRIEVE` path loads `[align(a), h)` into the
  new blocks. The full-hit −1 rule and the async promotion sequence are
  unchanged.
- **Idempotent hit (H7).** `LMCacheMPRequestTracker.set_lookup_hit` records
  the hit once per generation instead of accumulating; it raises if stores
  were already emitted (a moved cursor would be a logic error).
- **Lock release (H4).** `request_finished` frees the lookup's read locks
  when the tracker is still `PREFETCHING` with a hit.
- **No double free (H8).** `request_finished` returns `delay_free=False`
  when the current generation never ran a forward pass (nothing can be
  reading its blocks) and records the id in
  `LMCacheMPConnectorMetadata.finished_without_store`. The worker-role
  connector forwards that set on `bind_connector_metadata`, and
  `LMCacheMPWorkerAdapter.skip_finished_sending` guarantees such ids are
  never reported in `finished_sending`.
- **Block-reuse race (H1).** `LMCacheMPWorkerAdapter.handle_preemptions`
  now also waits on every outstanding store future (its device event fires
  when the server's copy is done) on steps the scheduler flagged with
  `need_flush_before_forward`. This is the worker-side option from §6; the
  scheduler-side deferred free was not needed.

### 8.3 Measured results

L1 (`tests/v1/mp_preemption`, 62 tests, ~13 s, no GPU): all invariants hold
across the scenario matrix and 8 seeds × {sync, flush} × {async scheduling
on, off}; every seed resumes at least 3 requests with a RETRIEVE, up to
5 generations of the same request. The `SERVER_ASYNC` negative control
still observes poisoning, so the oracle is sensitive.

L2 (1 GPU, real server, both transfer modes): store → `handle_preemptions`
→ overwrite → lookup → retrieve returns the original KV bit-for-bit.

L3 on this host (facebook/opt-125m, `--num-gpu-blocks-override 256`,
64 requests × up to 768 tokens against a 4096-token pool,
`VLLM_BATCH_INVARIANT=1`):

| Pass | Requests | Preemptions | Divergent vs baseline | LMCache hit fraction |
|---|---|---|---|---|
| baseline (hot) | 64 | 62 | reference | – |
| baseline (replay concurrency, A/A floor) | 64 | 0 | 0 | – |
| lmcache hot (cold cache) | 64 | 71 | 0 | – |
| lmcache replay | 64 | 0 | 0 | 0.91 of prompt tokens |

The same ladder against the engine-driven transfer mode (2 repeats, 95 to
103 preemptions per hot pass, 0.92 replay hit fraction) passed with one
request diverging at output token 49 to the reference's runner-up at a
top-1/top-2 gap of 0.008 nats, classified as a near-tie.

Oracle note: with **random-token** prompts the same setup showed 1/64 A/A
and 3/64 LMCache divergences, all at near-ties in a flat next-token
distribution. Real-text prompts plus the top-2 logprob classifier
(`--near-tie-gap`) remove that noise; the script reports the A/A floor
alongside every run so a future divergence can be read against it.

### 8.4 Still open

- T1 numbers on a production-size model (the k3 `preemption_correctness`
  step prints wall time, p99 and LMCache-served prompt tokens; the slowdown
  gate is opt-in via `PREEMPT_MAX_SLOWDOWN_PERCENT`).
- Lazy offload under preemption (H5) is not in the L1 matrix yet.
- The connector still logs `<preempted>` at WARNING on every affected step;
  the older k3 `lm_eval_preemption` job greps for it, so it was left alone.

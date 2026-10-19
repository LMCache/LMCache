# GPU-native token dropping: clean AIO integration checkpoint

This branch is an **integration/test reference**, not a monolithic PR or a merge-ready implementation.

## Baseline

- Parent/base: `e7476a6fac22ba4ebce907b3fe3dedebb07bed06`. This contains the three frozen #5737 changes squashed into one new commit. The original #5737 branch and SHAs are unchanged.
- `upstream/dev` already includes #5736: use its `TokenDropSpec`, `parse_token_drop_spec` and entry-point plugin factory. No duplicate plugin registry, config parser or GPU KV view.
- The five refreshed features (#5741/#5742/#5743/#5568/#5663) are replayed on the clean base. APC isolation, allocator/reclaim and worker lifecycle are brought in only for the integration reference.
- Pinned vLLM runtime: **0.25.1**. Unsupported TD features fail fast: chunked prefill, speculative decoding, asynchronous scheduling, multiple GPU workers, multi-group KV.

## Runtime behavior and contracts

1. Scheduler identifies TD by the upstream request-local config parser; normal requests retain LMCache lookup/store behavior.
2. TD skips LMCache lookup/offload, and must not read or publish shared vLLM APC blocks; private GPU blocks are required.
3. Scheduler owns authoritative physical resident frontier **P**, separate from logical progress **L** and RoPE/model positions. Allocator accounts for P; worker adaptor uses P only for KV read/write positions and lengths.
4. Request metadata carries algorithm selection, current P, scheduler-owned block IDs, decode facts and a mandatory scheduler-validated exclusive/private block gate.
5. Worker initializes per-request algorithm plugins, optionally captures post-RoPE Q, and selects ordered kept positions in `[kept_tokens, kv_heads]` layout. Reuse frozen `PagedKVSequenceView`; no extra contiguous K/V requirement introduced by LMCache.
6. Validate all layer/head selection shapes and index ranges before the first destructive write. Selection errors can skip before mutation if the CUDA context is healthy. Compaction errors after possible in-place writes must propagate, **not** silently resume decoding.
7. Worker synchronizes its compaction stream before publishing absolute resident-length updates; scheduler commits P and frees only now-unneeded private tail blocks after receiving the worker output.
8. Clean algorithm/request state on finish and preemption; regular requests remain vanilla in mixed batches.

## Maintainer-review constraints

- Kuntai: CUDA-only tests for GPU operations (not cross-device coverage in this MVP).
- Kuntai: no forced `.contiguous()` in the frozen KV accessor; downstream algorithms decide when materialization is needed.
- Kuntai: serving availability matters, but catching exceptions after possible in-place mutation is not a safe fallback. Fall back only before mutation and with a healthy CUDA context; fail fast otherwise.
- Kuntai + ApostaC: code placement under `integration/vllm/experimental` is accepted.
- Merged #5736 already implements the request-local algorithm factory and parser; do not reinvent them.

## Validation gates / release blockers

- **Static**: Ruff lint + formatting, isort, compileall, Git diff whitespace.
- **CUDA unit**: both FP16/BF16; invalid indices prewrite; mixed TD+normal; repeated compactions; independent per-head/layer kept positions; failed GPU operations never publish P. GPU-required tests use CUDA-only skip markers.
- **Pinned vLLM runtime**: import and smoke-test adaptor signatures, worker hook timing, attention Q capture, cache layout, metadata path and scheduling; verify request finish/preemption/ID reuse.
- **End-to-end correctness**: compare normal-only, TD-only, mixed and alternating batches; non-dropped KV matches vanilla, drop moves K/V correctly, next decode uses P, tail reclaimed only after worker barrier; no APC or LMCache offload leakage.
- **Performance**: benchmark vanilla versus mixed TD, GPU compaction microbenchmark (including paged materialization, scoring and relocation), and throughput/latency at matched workload/buffer settings.
- **R-KV plugin compatibility**: check the latest Zefan-Cai/R-KV serving code and pending token-major PR #64; do not assume this branch alone proves plugin API compatibility.
- The connected Windows worktree does not have CUDA, `torch`, `vllm` or `pytest`. Any CUDA/E2E/perf gates not executed there remain explicitly unverified.

This is the reference to refine with the next R-KV serving changes **before** finalizing the BFS-style PR dependency split.

## Executed validation checkpoint — 2026-10-18

Revisions: LMCache AIO feature commit 7038c264; R-KV PR #64 e6c9ccc. GPU: H100 80GB on n4-2, CUDA 13-capable driver, vLLM 0.25.1, Qwen3-0.6B BF16. The original #5737 branch stays frozen.

### Correctness results

- R-KV CUDA serving and scoring tests: 52 PASS; cross-repo frozen paged GPU KV primitive parity against legacy dense R-KV passed for FP16/BF16 and non-contiguous input K.
- Full LMCache token-drop suite: 116 PASS, zero failures. Six added admission tests cover unsupported chunked prefill, speculation, async scheduling, multi-GPU, multi-group KV, and successful adaptor installation. A CUDA-only fault-injection test confirms that an exception raised after a real in-place GPU KV write propagates rather than being swallowed.
- Real vLLM eager E2E: vanilla, normal-only, TD-only, and mixed normal+TD all generate successfully. APC-enabled mixed also passes; in each mixed repeat, normal-request generated tokens match vanilla.
- Long-context APC mixed: 4 requests, 2 normal and 2 TD, 512 prompt + 512 output tokens, 2 repeats, 32 observed compactions, both completed.
- Real PIECEWISE CUDA Graph: mixed 2 requests x 96 output tokens (3 compactions), and TD-only 4 requests x 256 output tokens (32 compactions) both completed with actual compilation and graph capture.
- Mixed-batch crash root cause: engine-finished TD request IDs were erroneously re-reported as LMCache STORE finished_sending; the fix excludes TD completion from LMCache STORE receipts but preserves normal completion. Live failing case reproduced before the fix; live mixed case and new regression tests pass after it.
- Repeated TD-only E2E: 96 compactions over 3 runs. Repeated mixed eager: 48 compactions. Repeated mixed + APC: 48 compactions.
- Important caveat: 1 of 12 normal-only generated outputs differs from vanilla (first repeat, first request), reproducibly. Untouched upstream LMCache commit 663757e3 produces precisely the same output hash, so it is not introduced by AIO; the underlying upstream discrepancy is still open.
- Unit tests cover private allocation, preemption metadata/reset, fail-fast, worker/scheduler P ownership, post-compaction completion barrier, per-layer/per-head movement, and prewrite recoverable errors. A live mixed-batch run with vLLM KV-cache memory capped at 64 MiB actually preempted and resumed normal requests, completed generation, and performed 16 TD compactions. TD-specific preemption was not observed in the separate TD-only forced-cap runs; that specific resume path, end-to-end post-write fault recovery, and long-duration soak remain unverified.

### Performance results and RCA

Exploratory eager BF16 benchmark on the shared H100: four concurrent requests, 128 prompt + 256 generated tokens, three repeats per mode. Median tokens/second:

| Workload | Median tok/s | Relative to vanilla |
| --- | ---: | ---: |
| Vanilla | 412.8 | baseline |
| Normal + LMCache | 393.0 | -4.8% |
| TD-only + R-KV | 331.9 | -19.6% |
| Mixed 2 normal + 2 TD | 350.6 | -15.1% |
| Mixed + APC | 354.9 | -14.0% |

Profiling TD-only 4x256 with instrumentation: 32 request compactions cost 609.5 ms total worker wall time. R-KV selection ~431.9 ms total GPU events, including scoring ~316.0 ms across 896 per-layer calls; paged KV layer compaction ~77.1 ms. These nested measures overlap; do not sum them. K gathering ~23.6 ms. Scoring/selection is the dominant optimization target; R-KV does not read V for scoring. Profiling hooks add overhead and are not used in the throughput table.

Long 512+512 APC mixed: median 376.3 tok/s in 2 runs, a different workload and not directly comparable. PIECEWISE TD-only 4x256: 638.9 tok/s in a single smoke run; compiled vanilla/normal baselines have not been measured, so this is NOT a relative speedup claim.

Reproducible scripts, JSON results and logs are in the scratch workspace on n4-2: /home/sigma/td-aio-gates-20261018/.

### Open gates and next decisions

- Run forced-preemption and fault-injected GPU write failure scenarios if needed for production confidence. Unsupported scheduling/multi-device features remain fail-fast by design.
- Investigate the preexisting upstream normal-only output divergence separately; do not call that parity gate fully green.
- Establish explicit perf acceptance thresholds and obtain matched PIECEWISE control baselines before evaluating throughput acceptance.
- For optimization, prioritize batched/per-layer R-KV scoring and selection overhead; preserve algorithm semantics and the minimum architecture principles.
- Defer breadth-first PR split until AIO correctness and performance requirements are agreed.

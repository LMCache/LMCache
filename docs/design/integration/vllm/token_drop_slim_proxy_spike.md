# Experimental: engine-local token-drop allocator proxy (vLLM 0.25.1)

This alternative to LMCache #5871 **does not install a class-global monkeypatch**.

## How to select it

Pass the vLLM scheduler class explicitly at engine startup:

```shell
vllm serve MODEL --scheduler-cls lmcache.integration.vllm.experimental.token_drop_scheduler_proxy.LMCacheTokenDropScheduler
```

Without this option, nothing from the experimental proxy is installed. vLLM
instantiates our `Scheduler` subclass using its existing `scheduler_cls`
configuration. The subclass calls stock `Scheduler.__init__`, then wraps only
that instance's `kv_cache_manager` with a forwarding proxy.

The proxy delegates all methods except `allocate_slots`. For ordinary requests,
the **original Request** reaches vLLM's original allocator. For compressed
requests, the proxy calls the existing S2 helper with a temporary Request view
whose `num_computed_tokens` property returns the physical resident KV frontier.
It never overwrites the logical Request. The actual allocation code remains
vLLM's unmodified `KVCacheManager.allocate_slots`.

The helper still shares the LMCache resident-length state and preemption guards
with #5871 / S8. This is a *design alternative*, not an extra layer to install
on top of the existing class-global S2 patch.

## Risk assessment — not production-ready

- **Still external allocation semantics.** Stock vLLM CI does not test this
  temporary physical-length Request view or LMCache's GPU compaction.
- **Unstable extension point.** vLLM v0.25.1 explicitly warns `scheduler_cls`
  is not a public stable interface. Startup wiring, class composition and
  scheduler-to-manager references may break on upgrade.
- **P ownership and isolation.** The existing S2/S8 P dict is keyed only by
  request ID, *not by engine*. This proxy being engine-local does **not**
  make the state engine-local. Distinct engines reusing IDs could corrupt each
  other's physical allocation frontier. Multi-engine isolation must be
  designed and tested (or configurations must be rejected) before deployment.
- **Memory integrity and tenant isolation.** If compaction and tail-block
  reclaim are not fenced behind worker completion, newly allocated requests
  could reuse blocks still being read or written. This can yield stale KV,
  cross-request contamination or an out-of-bounds access. The proxy does not
  itself provide the compaction/output/reclaim barrier.
- **Prefix caching / shared blocks.** TD requests must own unshared GPU blocks;
  the S8 ownership checks and connector APC/offload bypass remain required.
- **Lifecycle.** Preemption, request-ID reuse, engine restart, failures between
  reclaim and P commit, S8's same-step preempt/resume, and worker metadata
  correctness require full integration tests.
- **Unsupported modes.** Spec decoding, chunked prefill, async scheduling,
  TP/PP and external KV remain gated in the full MVP integration.

## Evidence and boundaries

- The separate spike ran real vLLM 0.25.1 Scheduler + KVCacheManager CPU
  allocation tests: 3/3 pass (mixed TD/normal batch, physical frontier,
  logical progress, no class-global patch).
- The S2 helper with an engine-local thin proxy passed 7/7 independent
  contract checks (passthrough, guards, failure handling, isolation at the
  method level). These are **not** complete GPU serving tests.
- The prototype is intentionally not installed automatically; full S12 E2E
  remains required before activating in production.

See #5871 for the existing S2 implementation and #6035 for S8 reclaim
semantics. Do not merge both alternatives as separate active adapters.

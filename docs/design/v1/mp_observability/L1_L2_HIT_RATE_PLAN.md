# Design Plan: L1+L2 Token-Level Cache Hit Rate Metric

## Status

Proposal — not yet implemented.

## Motivation

The existing metrics in `METRICS.md` expose throughput counters for L1 (`lmcache_mp_l1_read_keys_total`, etc.), L2 prefetch hits (`lmcache_mp_l2_prefetch_hit_keys_total`), and Storage Manager reads (`lmcache_mp_sm_read_succeed_keys_total`). None of these directly answer the operational question:

> What fraction of tokens requested by a lookup were served from cache (L1 or L2)?

Specifically:

- L1 has no hit/miss counter — only read/write/evict counts.
- L2 prefetch counters track the load phase, not the full lookup decision.
- SM read counters track key-level outcomes but do not tie back to the *requested* lookup size, so they cannot form a proper ratio.

Since the `StorageManager` already combines L1 and L2 in a single prefetch path (L1 is checked first, misses fall through to L2), a single hit counter recorded at lookup completion covers both tiers. L0 (GPU prefix cache) is intentionally excluded — it is vLLM-owned and not observable from LMCache.

## Goal

Expose two Prometheus counters that let an operator compute:

```promql
rate(lmcache_mp_lookup_hit_tokens_total[5m])
/
rate(lmcache_mp_lookup_requested_tokens_total[5m])
```

This ratio is the **L1+L2 token-level hit rate**.

## Non-Goals

- Measuring L0 (GPU) prefix cache hit rate — requires vLLM-side metrics.
- Per-model or per-instance slicing in v1 (can be added via labels later; kept out for cardinality safety).
- Changing any existing metric or event contract (additive only).

---

## Design Overview

A single event — `MP_LOOKUP_PREFETCH_END` — is extended with two new metadata fields so both numerator and denominator arrive together and are attributable to the same completed lookup. A new `LookupMetricsSubscriber` consumes this event and increments two OTel counters.

### Why a single event for both counters

An earlier sketch proposed splitting the counters across `MP_LOOKUP_PREFETCH_START` (denominator) and `MP_LOOKUP_PREFETCH_END` (numerator). This was rejected because:

1. **Abandoned lookups skew the rate.** If a client calls `lookup` but never calls `query_prefetch_status`, the `END` event never fires. With a split design, the denominator grows without the numerator, understating the hit rate. A pending `TODO` at `lmcache/v1/multiprocess/server.py:196–197` already notes that stale `_prefetch_jobs` entries can accumulate in this scenario.
2. **Early-exit paths in `lookup()`** (no matching GPU context at line 630; empty `chunk_hashes` at line 655) emit `MP_LOOKUP_PREFETCH_START` *before* the guard checks, so the START event fires for lookups that can never hit. Attributing a nonzero denominator to these would bias the ratio down.
3. **Emit-time co-location** keeps the subscriber stateless — no correlation by `request_id` across events, no race between concurrent lookups.

### Why emit tokens (not chunks) at the event boundary

The event metadata should carry token counts directly (`requested_tokens`, `hit_tokens`) rather than chunks. Reasons:

- Keeps the subscriber dumb — it does not need to know `chunk_size`.
- The emit site (`query_prefetch_status`) already has `chunk_size` via `self.chunk_size` on the engine, so multiplication there is trivial.
- Consumers of the event (metrics, logging, tracing) all want tokens, not chunks.

---

## Contract Changes

### Event: `MP_LOOKUP_PREFETCH_END` (existing)

**Current metadata** (`lmcache/v1/multiprocess/server.py:783–789`):

```python
metadata={"found_count": found_count}   # found_count is chunks, per-world-size
```

**Proposed metadata**:

| Key | Type | Meaning |
|---|---|---|
| `found_count` | `int` | Existing. Chunk-level prefix hit count, post `// world_size`. |
| `requested_tokens` | `int` | **New.** Number of tokens submitted for lookup; equals `len(chunk_hashes) * chunk_size` computed in `lookup()`. Zero for early-exit paths. |
| `hit_tokens` | `int` | **New.** `found_count * chunk_size`. |

`requested_tokens` must be threaded through `_PrefetchJob` since it is known in `lookup()` but consumed in `query_prefetch_status`.

### Event: `MP_LOOKUP_PREFETCH_START`

**No change.** Emission site and metadata stay as-is. The counters do not depend on it.

### New Counters (OTel → Prometheus)

| Metric Name | Prometheus Name | Type | Unit | Meaning |
|---|---|---|---|---|
| `lmcache_mp.lookup_requested_tokens` | `lmcache_mp_lookup_requested_tokens_total` | Counter | tokens | Total tokens submitted for lookup (denominator). Only counts the chunk-aligned portion; sub-chunk trailing tokens are excluded because they cannot hit by design. |
| `lmcache_mp.lookup_hit_tokens` | `lmcache_mp_lookup_hit_tokens_total` | Counter | tokens | Total tokens found in L1+L2 during lookup (numerator). Counts the contiguous prefix hit only. |

Both counters live in the meter `lmcache.lookup`. No labels in v1.

### New Subscriber: `LookupMetricsSubscriber`

- File: `lmcache/v1/mp_observability/subscribers/metrics/lookup.py`
- Subscribes to: `MP_LOOKUP_PREFETCH_END` only.
- Handler:
  ```
  on MP_LOOKUP_PREFETCH_END(event):
      lookup_requested_tokens.add(event.metadata["requested_tokens"])
      lookup_hit_tokens.add(event.metadata["hit_tokens"])
  ```
- Registration: added to `init_observability()` in `lmcache/v1/mp_observability/config.py` alongside the existing five subscribers (line 300–304).

---

## Call-Site Changes in `server.py`

All changes additive; no existing behavior is altered.

### `_PrefetchJob` dataclass

Add a `requested_tokens: int` field so the denominator survives until `query_prefetch_status` fires `MP_LOOKUP_PREFETCH_END`.

### `lookup()` method

1. Compute `requested_tokens = len(chunk_hashes) * self.chunk_size` once after the `compute_chunk_hashes` call on the happy path.
2. Set `requested_tokens=0` on the two early-exit paths (no layout_desc; empty chunk_hashes) — this keeps the dummy `_PrefetchJob` entries contributing zero to both counters, preserving the ratio.
3. Pass `requested_tokens` into the `_PrefetchJob` constructor at all three call sites.

### `query_prefetch_status()` method

Currently emits (line 783–789):

```python
self._event_bus.publish(
    Event(
        event_type=EventType.MP_LOOKUP_PREFETCH_END,
        session_id=job.request_id,
        metadata={"found_count": found_count},
    )
)
```

Extend metadata to:

```python
metadata={
    "found_count": found_count,
    "requested_tokens": job.requested_tokens,
    "hit_tokens": found_count * self.chunk_size,
}
```

No other logic changes. Exactly-once semantics remain — the job is still popped from `_prefetch_jobs` at line 791–792.

### `query_prefetch_lookup_hits()` method

**No change.** This method is a polling/preview call with no guaranteed exactly-once cardinality and must not emit counter-driving events.

---

## Correctness Properties

1. **Exactly-once numerator and denominator per lookup.** Both come from a single event that fires exactly once per `query_prefetch_status` completion. Even if the client polls `query_prefetch_lookup_hits` repeatedly, no counters are touched there.
2. **Early-exit lookups contribute zero to both.** Dummy `_PrefetchJob` entries created when `layout_desc is None` or `chunk_hashes` is empty have `requested_tokens=0`, and their `found_count` is always 0. Ratio unchanged.
3. **Abandoned lookups contribute to neither.** If a client never calls `query_prefetch_status`, the `_PrefetchJob` leaks (existing behavior, tracked by the `TODO` at `server.py:196–197`). Neither counter increments — the hit rate is unaffected. When the leak is eventually garbage-collected by the future cleanup path, that path should *not* synthesize an `MP_LOOKUP_PREFETCH_END`; leaked lookups are silently dropped from the rate.
4. **Unit consistency.** Both counters are in tokens. The conversion from chunks happens exactly once, at the event emit site.
5. **Prefix semantics.** `found_count` is the contiguous prefix length, not the number of individual keys found anywhere in the range (see the comment at `server.py:777–780`). This matches the semantic definition of "prefix cache hit" and is the correct numerator for KV cache reuse.
6. **World-size normalization.** `found_count` is already divided by `world_size` at line 781 before the event fires, so the tokens math is in per-request terms and independent of tensor-parallelism.

---

## Prometheus Queries

### Overall L1+L2 token hit rate

```promql
rate(lmcache_mp_lookup_hit_tokens_total[5m])
/
rate(lmcache_mp_lookup_requested_tokens_total[5m])
```

### Absolute throughput (for sanity checking)

```promql
# tokens/sec looked up
rate(lmcache_mp_lookup_requested_tokens_total[5m])

# tokens/sec hit
rate(lmcache_mp_lookup_hit_tokens_total[5m])
```

### Comparison against SM-level counters

A cross-check: `lmcache_mp_sm_read_succeed_keys_total * chunk_size` should be of the same order as `lmcache_mp_lookup_hit_tokens_total` (but not identical — SM counts keys submitted to reserve_read, while lookup counts the prefix hit from the client's perspective).

---

## Edge Cases & Open Questions

1. **Partial trailing chunks.** `compute_chunk_hashes` drops sub-chunk-size tails. These tokens appear in `len(key.token_ids)` but not in `len(chunk_hashes) * chunk_size`. Choice: exclude them from the denominator (current proposal). They can never hit at chunk granularity; including them would pull the hit rate artificially down without corresponding numerator opportunity.
2. **Very short lookups.** A lookup with fewer than `chunk_size` tokens yields `chunk_hashes=[]` and goes through the empty-`chunk_hashes` early exit. Both counters increment by 0. The request is effectively invisible in the rate. This is correct — there was nothing to look up.
3. **Error path in storage manager.** If `submit_prefetch_task` raises, `_register_prefetch_job` is never called and no counters fire. If it succeeds but the backend subsequently fails, `query_prefetch_status` will return whatever `found_count` the SM produces (could be 0). Denominator still increments via `requested_tokens`. This correctly reflects the hit rate as "fraction of requested tokens that the cache layer returned," where backend failures count as misses.
4. **Backward compatibility.** Extending `MP_LOOKUP_PREFETCH_END` metadata is additive; existing subscribers ignore unknown keys. No version bump required for the event contract.
5. **Should `query_prefetch_lookup_hits` emit anything?** No. It is a best-effort preview and may be called zero or many times per lookup. Keeping it side-effect-free preserves the cardinality invariant for the counters.

---

## Implementation Steps (for whoever picks this up)

1. **Extend the event contract**
   - Update `docs/design/v1/mp_observability/EVENTS.md` to document the new `requested_tokens` and `hit_tokens` metadata on `MP_LOOKUP_PREFETCH_END`.

2. **Thread the denominator through `_PrefetchJob`**
   - Add `requested_tokens: int` to `_PrefetchJob` in `lmcache/v1/multiprocess/server.py`.
   - Populate at all three construction sites in `lookup()` (two early-exit, one happy-path).

3. **Enrich event emission**
   - In `query_prefetch_status`, extend the `MP_LOOKUP_PREFETCH_END` metadata dict to include `requested_tokens` and `hit_tokens`.

4. **Add the subscriber**
   - Create `lmcache/v1/mp_observability/subscribers/metrics/lookup.py` following the same skeleton as `l2.py` (meter, two counters, one event handler).
   - Name the meter `lmcache.lookup`.

5. **Register the subscriber**
   - In `lmcache/v1/mp_observability/config.py::init_observability`, append `LookupMetricsSubscriber()` registration to the existing block (around line 300–304).

6. **Document the metric**
   - Update `docs/design/v1/mp_observability/METRICS.md` with the two new counter rows.
   - Update the project-level `METRICS_EXPLORATION.md` (if kept in sync) with a new section for the lookup-level hit rate, plus a note in the "Summary Table."

7. **Add tests**
   - Unit test the subscriber with a synthesized `MP_LOOKUP_PREFETCH_END` event.
   - Integration test: issue a lookup with a known token count, poll to completion, scrape `/metrics`, assert both counters advanced by the expected amount.
   - Include the three edge cases: happy path, no-layout-desc, empty-chunk-hashes.

8. **Validate**
   - `pre-pr-check` skill per project standards.
   - Manual: `curl localhost:9090/metrics | grep lookup_` after a benchmark run.

---

## Alternatives Considered

### A. Emit denominator on `MP_LOOKUP_PREFETCH_START`, numerator on `MP_LOOKUP_PREFETCH_END`

**Rejected.** Abandoned lookups and early-exit paths break the symmetry. See "Why a single event" above.

### B. Use SM-level metrics (`sm_read_succeed_keys`, `sm_read_failed_keys`) as the hit rate

**Rejected.** These count keys *per SM call*, not per lookup, and are not per-prefix. They also do not distinguish the first-chunk-missed-so-stop-early case from per-key misses. Semantically wrong for a prefix cache.

### C. Extend an existing subscriber rather than creating a new one

**Considered.** Could be added to `L2MetricsSubscriber` or `SMMetricsSubscriber`. Rejected for single-responsibility — the lookup metric is about the *combined* L1+L2 view, which neither subscriber owns. A new subscriber makes the ownership boundary clear.

### D. Add labels (`model_name`, `instance_id`)

**Deferred.** Useful for multi-model deployments, but raises Prometheus cardinality concerns. Revisit once a second model is deployed or operators request per-model slicing.

---

## References

- Existing events contract: `docs/design/v1/mp_observability/EVENTS.md`
- Existing metrics contract: `docs/design/v1/mp_observability/METRICS.md`
- Event bus design: `docs/design/v1/mp_observability/event-bus.md`
- Subscriber implementations: `lmcache/v1/mp_observability/subscribers/metrics/`
- Call site: `lmcache/v1/multiprocess/server.py::MPCacheEngine.lookup` and `::query_prefetch_status`
- Project-level metrics exploration report: `METRICS_EXPLORATION.md`

# Hybrid cache correctness trace harness

This harness compares cache-disabled or reference execution with L1/L2/hybrid
cache execution at five layers:

1. selected token, log-probability, and top-k overlap;
2. logits max/mean absolute error, KL divergence, and cosine similarity;
3. request generation, accepted length, block table, prefix, and drop-round
   digests;
4. per-group/rank logical spans, physical pages, dtype, shape, stride, content
   digest, and representation revision;
5. ordered store/retrieve/reuse/abort/preempt/resume lifecycle events.

The trace format is canonical JSON with a deterministic SHA-256 identity.
`compare_traces()` reports all differences and orders `first_divergence` by
decode step and then output → logits → request → cache → lifecycle. This makes
an output mismatch actionable without hiding an earlier request/cache fault.
`run_id` and metadata are included in the trace identity for provenance, but
they intentionally do not affect semantic matching. Experiment drivers must
therefore enforce compatible model, hardware, and capture configurations.

## Capture contract

Adapters should capture one `TraceFrame` after each accepted decode step. Do
not add rejected speculative tokens to `accepted_seq_len`. `CacheGroupFrame`
must be emitted for every rank and every member of a hybrid cache family; hash
the actual byte representation with `sha256_digest()`.

Lifecycle `sequence` values must reflect observation order. A backend should
emit separate submitted, complete, and source-reusable events rather than
treating submission as completion.

## Compare two runs

```bash
python benchmarks/hybrid_cache_correctness/trace_harness.py \
  reference.json candidate.json --output first-divergence.json
```

Exit code 0 means all captured evidence matches within the numerical
tolerances. Exit code 1 means a divergence was found. The report remains useful
when full logits are intentionally omitted; logit metrics are then `null`.

Recommended experiment axes are cache off/L1/L2/L1+L2, BF16/FP8, aligned and
unaligned prefixes, concurrency 1/8/32/64, graph mode, preemption/abort, and
first/exact/partial reuse. Results from different hardware or capture contracts
must not be combined as one performance score.

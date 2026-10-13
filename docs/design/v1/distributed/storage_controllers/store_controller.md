# Contributor links for L2 store scheduling

With `LMCACHE_MP_TRACE_CONTEXT=1`, `StoreListener` captures W3C context on L1
reservation callbacks and pairs it with later write completion under its lock.
The store loop creates an independent `mp.l2.store.schedule` span with links
to observed contributors. It does not select one request as the parent of a
shared batch. An empty context clears worker baggage and previous requests;
the original context is restored on success, failure, and cancellation.

```
L1 reservation -> bounded writer summaries -> L1 completion
                                               |
                                   atomic keys + summary pop
                                               |
                            independent scheduling span + links
                                               |
                                  policy + adapter submission
```

Attribution retains at most 10,000 keys, 8 links per key, and 128 links per
batch. Sampling flags accumulate before the per-key and batch link limits.
A sampled ninth writer or 129th contributor still reaches the root sampler.
Only valid, entirely known, unsampled batches get an independent non-recording
context. Missing attribution uses the existing root sampler. It never borrows
the completion thread's parent, and keys remain queued if attribution is lost.

Once key eviction occurs, all newly created writer summaries are marked
incomplete until listener close. An evicted key can be reserved again, so this
conservative policy avoids treating its surviving writers as the full set
without retaining an unbounded set of evicted identities. Deletion and
prefetch-only completion discard per-key attribution; close clears all tracing
metadata. `pop_pending_keys()` keeps its existing keys-only behavior.

L1 callbacks do not identify individual admitted writers. Links describe
observed reservation contributors, not an exact writer audit. This bounded
attribution does not change L1 admission or adapter behavior. Only
`traceparent` and `tracestate` are captured; no payload or baggage is retained.

The span covers synchronous policy selection and adapter submission, not
background I/O completion, GPU operations, or RDMA. Caught adapter submission
failures mark it ERROR while preserving lock cleanup and later submissions.
Exception text, tracebacks, and attributes are not recorded on this span.
No SDK, exporter, or provider is installed. The feature is off by default.

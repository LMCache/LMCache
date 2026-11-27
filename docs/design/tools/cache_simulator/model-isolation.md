# Model isolation in request-trace analysis

`lmcache bench trace-analyze` accepts traces containing several models and
replays them against one token-capacity budget. Model names are opaque cache
namespaces: identical token IDs from different models do not identify reusable
KV. An empty model name is a namespace too; the trace cannot distinguish models
that a producer labels identically.

## Request identity

Both trace-format adapters produce `TraceRecord` values. After optional
`--model` filtering and timestamp sorting, `load_trace_records()` keeps the
earliest record for each `(model, request_id)` pair. Repeated lookups of the same
request count once, while independent models may reuse a request ID.

## Cache identity and eviction

`TokenHasher` continues to hash token IDs without changing the server's hash
format. The replay wraps each hash in `(model, chunk_hash)` when accessing the
single LRU cache. All models compete for the same number of chunk slots;
allocating a separate LRU per model would multiply the configured capacity.

Each model has its own `BlendTokenRangeMatcher` because non-prefix matching
probes token content rather than the LRU key. On LRU eviction, the replay
removes the evicted hash from the matcher of the **evicted model**, which may
differ from the model inserting the new chunk.

The JSON and HTML reports retain their existing aggregate schema. `--model`
selects a model before replay, so excluded models consume no capacity.

## Limits and validation

Capacity remains a token budget, not a byte budget across differently sized
models. The trace does not carry cache salt, KV dtype or model revision, so this
analysis does not infer isolation dimensions absent from its input. Each model
adds a matcher with two fixed-size arrays (about 16 MiB in the current matcher),
plus its indexed-chunk metadata.

Reader regressions cover both producer formats, timestamp ordering and model
filtering. Replay regressions cover prefix and non-prefix isolation, same-model
reuse, shared capacity and eviction from another model's matcher. A CLI
subprocess test verifies that the reader and replay produce consistent JSON
and HTML results together.

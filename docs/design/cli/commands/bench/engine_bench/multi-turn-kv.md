# Mixed-length multi-turn KV qualification

Issue #5848 needs growing conversation prefixes under fixed serving budgets,
without changing prefetch admission or eviction. `MultiTurnKVWorkload` reuses
`BaseWorkload`, `RequestSender`, `StatsCollector`, and `ProgressMonitor`. Its
standalone runner is `examples/benchmarks/multi_turn_kv/benchmark.py`; it is
deliberately separate from the existing workload registry and interactive UI.

## Token contract

Synthetic short/long contexts and turn additions use the existing verified
single-token word pool and a per-session RNG. Requests use raw completion
token IDs, not a chat template. The next prompt is exactly:

```text
previous prompt IDs + server-generated IDs + next turn's input IDs
```

`RequestSender.send_token_request` opts into vLLM's `return_token_ids` protocol
(checked against vLLM v0.27.1). It disables prompt special-token insertion,
echo and SDK retries, verifies returned prompt IDs and usage counts, and
collects delta IDs even when decoded text is empty. Missing/inconsistent IDs
fail the request. There is no text re-encoding fallback. `TokenCompletion`
wraps the unchanged `RequestResult`, preserving text callbacks and CSV schema.
Existing `send_request` and `send_warmup_request` paths are unchanged.

## Scheduling and lifetime

Sessions alternate short/long initial contexts. Each session has at most one
in-flight turn. The global pending-task limit bounds concurrent requests;
round-robin selection skips busy or thinking sessions. Think time begins at
completion, and measured client queueing starts when a session becomes ready.
A failed turn ends that session; peers continue. Unexpected task failures
drain peer requests before propagating, so normal client teardown does not
close their connections early. No request deadlines, transfer cancellation,
retries, cache clearing, or server memory-accounting changes are introduced.

The configuration bounds turns and worst-case context growth, including the
last output reservation. Initial requests are measured; warmup is a no-op.
First/continuation turn labels describe workload phase, not actual cache
residency. Deployment restarts and fresh L2 directories are operator controls.

## Measurement boundary

The runner exports per-turn records, per-session/cohort queueing and TTFT
percentiles, and optional queue-plus-TTFT SLO counts. Failed attempts stay in
queueing/count denominators; unattempted turns are reported separately.
`prefix_overlap_tokens` describes identical input, not a measured cache hit.

Optional pre/post MP counter scrapes sum labels from an isolated server.
Counter resets, failed scrapes and missing final series remain unavailable.
L2 activity requires positive lookup hits, L2 hits and completed L2-load
chunks. This confirms observed L2-to-L1 activity, not a latency benefit or
end-to-end accuracy. Missing transfer-byte, memory-peak and per-token ITL
measurements are explicitly listed, never inferred from token counts or
throughput histograms. Raw snapshots remain in the report for inspection.

The implementation is a workload and run-matrix contribution. Hardware
qualification and cross-policy performance results require actual serving
runs; CPU HTTP fixtures are not evidence of KV-cache correctness or speed.

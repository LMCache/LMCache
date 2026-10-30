# P2P metrics (Prometheus names as exposed on `--http-port /metrics`)

OTel names like `lmcache_mp.lookup_hit_l2` become `lmcache_mp_lookup_hit_l2_tokens_total`
(unit suffix, `_total` for counters). Counters appear only after their first event.

| metric | meaning for P2P |
|---|---|
| `lmcache_mp_lookup_requested_tokens_total` | chunk-aligned tokens looked up (denominator) |
| `lmcache_mp_lookup_hit_l1_tokens_total` | served from this node's own L1 |
| `lmcache_mp_lookup_hit_l2_tokens_total` | added by L2 adapters: with P2P as the only L2 this is **tokens fetched from peers** |
| `lmcache_mp_l2_load_completed_requests_total{l2_name="p2p"}` | P2P load requests completed; the direct proof a peer served data |
| `lmcache_mp_l2_prefetch_load_completed_chunks_total` | chunks loaded from L2 (x chunk bytes = bytes moved) |
| `lmcache_mp_l2_prefetch_failure_chunks_total{reason}` | chunks that fell back to recompute: `l1_oom` (L1 too small for in-flight loads), `l1_contended` (same keys being loaded concurrently) |
| `lmcache_mp_l2_load_throughput_GB_per_second_{sum,count}` | per-(request, adapter) histogram of bytes / submit-to-complete latency; compare configs, not a link rate |
| `lmcache_mp_active_p2p_lookup_jobs` | lookups this node is serving **for peers** right now |
| `lmcache_mp_l1_usage_ratio`, `lmcache_mp_l1_evicted_chunks_total` | L1 fill and evictions; evictions track P2P loads because loaded keys are temporary |
| `lmcache_mp_num_finished_stores_total`, `..._retrieves_total` | engine-side stores and L1-to-GPU retrieves |
| `lmcache_mp_event_bus_dropped_events_total` | must stay 0 (metrics are lossy otherwise) |
| `vllm:num_requests_running/waiting`, `vllm:time_to_first_token_seconds_*` | engine queue and TTFT |

Chunk bytes = chunk_size x layers x kv_heads x head_dim x 2 x dtype bytes (256 tokens of
Qwen3-8B = 37.7 MB). `scripts/tokens_per_gb.py` computes the per-token value.

## PromQL used by monitor.py / export_prom.py (`promql.py`)

```
sum by (node) (rate(lmcache_mp_lookup_hit_l2_tokens_total[10s]))          # tokens served via P2P per s
sum by (node) (rate(lmcache_mp_l2_load_throughput_GB_per_second_sum[10s]))
  / sum by (node) (rate(lmcache_mp_l2_load_throughput_GB_per_second_count[10s]))
sum by (node) (increase(lmcache_mp_l2_prefetch_failure_chunks_total[5m]))  # by reason: add `by (node, reason)`
sum by (node) (lmcache_mp_active_p2p_lookup_jobs)
```

Aggregate P2P bandwidth over a run = prefetch chunks completed x chunk bytes / wall time
(from the stress JSON), or the NIC counter diff from `lmcache-p2p-diagnose/scripts/rail_check.sh`.

`/metrics/reset` (POST) on the HTTP port zeroes the counters between experiments.

# SPDX-License-Identifier: Apache-2.0
"""PromQL expressions shared by monitor.py and export_prom.py.

Every expression is aggregated ``by (node)``; the ``node`` label comes from the
Prometheus config written by ``start_prometheus.sh``. Counters are turned into rates
over ``WINDOW``.
"""

WINDOW = "10s"

# column -> (promql, kind).
# kind: gauge | gauge_pct (x100) | rate | hist_mean (rate(sum) / rate(count))
EXPRS: dict[str, tuple[str, str]] = {
    "lookup_tok/s": (
        f"sum by (node) (rate(lmcache_mp_lookup_requested_tokens_total[{WINDOW}]))",
        "rate",
    ),
    "hit_l1_tok/s": (
        f"sum by (node) (rate(lmcache_mp_lookup_hit_l1_tokens_total[{WINDOW}]))",
        "rate",
    ),
    "hit_l2_tok/s": (
        f"sum by (node) (rate(lmcache_mp_lookup_hit_l2_tokens_total[{WINDOW}]))",
        "rate",
    ),
    "p2p_jobs": ("sum by (node) (lmcache_mp_active_p2p_lookup_jobs)", "gauge"),
    "pf_lookup/s": (
        f"sum by (node) (rate(lmcache_mp_l2_prefetch_lookup_requests_total[{WINDOW}]))",
        "rate",
    ),
    "pf_chunks/s": (
        f"sum by (node) "
        f"(rate(lmcache_mp_l2_prefetch_load_completed_chunks_total[{WINDOW}]))",
        "rate",
    ),
    "pf_fail/s": (
        f"sum by (node) (rate(lmcache_mp_l2_prefetch_failure_chunks_total[{WINDOW}]))",
        "rate",
    ),
    "load_GB/s": (
        f"sum by (node) "
        f"(rate(lmcache_mp_l2_load_throughput_GB_per_second_sum[{WINDOW}])) / "
        f"sum by (node) "
        f"(rate(lmcache_mp_l2_load_throughput_GB_per_second_count[{WINDOW}]))",
        "hist_mean",
    ),
    "l1_use%": ("sum by (node) (lmcache_mp_l1_usage_ratio)", "gauge_pct"),
    "l1_evict/s": (
        f"sum by (node) (rate(lmcache_mp_l1_evicted_chunks_total[{WINDOW}]))",
        "rate",
    ),
    "stores/s": (
        f"sum by (node) (rate(lmcache_mp_num_finished_stores_total[{WINDOW}]))",
        "rate",
    ),
    "retr/s": (
        f"sum by (node) (rate(lmcache_mp_num_finished_retrieves_total[{WINDOW}]))",
        "rate",
    ),
    "vllm_run": ("sum by (node) (vllm:num_requests_running)", "gauge"),
    "vllm_wait": ("sum by (node) (vllm:num_requests_waiting)", "gauge"),
    "ttft_mean_s": (
        f"sum by (node) (rate(vllm:time_to_first_token_seconds_sum[{WINDOW}])) / "
        f"sum by (node) (rate(vllm:time_to_first_token_seconds_count[{WINDOW}]))",
        "hist_mean",
    ),
    "vllm_tok/s": (f"sum by (node) (rate(vllm:prompt_tokens_total[{WINDOW}]))", "rate"),
}


def parse_vector(result: dict) -> dict[str, float]:
    """Map an instant-query result to ``{node: value}``, dropping NaN/Inf samples."""
    out = {}
    for s in result["data"]["result"]:
        v = float(s["value"][1])
        if v == v and v not in (float("inf"), float("-inf")):
            out[s["metric"].get("node") or s["metric"].get("instance", "?")] = v
    return out


def parse_matrix(result: dict) -> dict[tuple[float, str], float]:
    """Map a range-query result to ``{(ts, node): value}``, dropping NaN/Inf samples."""
    out = {}
    for s in result["data"]["result"]:
        node = s["metric"].get("node", "?")
        for ts, v in s["values"]:
            if v not in ("NaN", "+Inf", "-Inf"):
                out[(float(ts), node)] = float(v)
    return out

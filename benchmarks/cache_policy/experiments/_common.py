# SPDX-License-Identifier: Apache-2.0
"""
Shared plumbing for the direction-finding experiment modules.

Not part of the shipped simulator (``lmcache/tools/cache_policy_bench/``) --
kept separate so experimental code never has to reach into that module's
private (underscore-prefixed) internals.
"""

# Standard
from typing import Optional
import time

# First Party
from lmcache.tools.cache_policy_bench.cost_model import CostModel
from lmcache.tools.cache_policy_bench.runner import BenchResult
from lmcache.tools.cache_policy_bench.workloads import Request
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy


class ChunkObj:
    """Stand-in for a ``MemoryObj``, exposing only what cache policies read."""

    __slots__ = ("can_evict", "observed_recompute_tokens", "_physical_size")

    def __init__(
        self, physical_size_bytes: int, observed_recompute_tokens: Optional[float]
    ) -> None:
        self._physical_size = physical_size_bytes
        self.observed_recompute_tokens = observed_recompute_tokens
        self.can_evict = True

    def get_physical_size(self) -> int:
        return self._physical_size


def run_policy_instance_workload(
    policy: BaseCachePolicy,
    result_label: str,
    requests: list[Request],
    cache_capacity_bytes: int,
    kv_bytes_per_chunk: int,
    cost_model: CostModel,
    workload_name: str = "",
) -> BenchResult:
    """
    Single-tier simulation loop, identical in behavior to
    :func:`lmcache.tools.cache_policy_bench.runner.run_workload` but taking
    an already-constructed policy *instance* rather than a name registered
    in ``get_cache_policy``'s ``POLICY_MAPPING`` -- lets experimental
    ``CostAwareEvictionPolicy`` subclasses (``variant_policies.py``) be
    benchmarked without registering them in production code.

    Args:
        policy: Pre-constructed policy instance to drive.
        result_label: Value recorded as ``BenchResult.policy_name``.
        requests: Request sequence to replay, in order.
        cache_capacity_bytes: Total simulated cache capacity in bytes.
        kv_bytes_per_chunk: Simulated bytes occupied by one cached chunk.
        cost_model: Modeled latency function.
        workload_name: Label recorded on the result for reporting.

    Returns:
        Aggregated :class:`BenchResult` for this run.

    Raises:
        ValueError: If ``cache_capacity_bytes`` or ``kv_bytes_per_chunk`` is
            non-positive.
    """
    if cache_capacity_bytes <= 0 or kv_bytes_per_chunk <= 0:
        raise ValueError("cache_capacity_bytes and kv_bytes_per_chunk must be positive")

    capacity_chunks = max(1, cache_capacity_bytes // kv_bytes_per_chunk)
    cache_dict: dict[str, ChunkObj] = policy.init_mutable_mapping()
    eviction_count = 0

    latencies: list[float] = []
    total_tokens = 0
    total_hit_tokens = 0
    wall_start = time.perf_counter()

    for req in requests:
        hit_prefix = 0
        for h in req.chunk_hashes:
            if h in cache_dict:
                hit_prefix += 1
            else:
                break

        for h in req.chunk_hashes[:hit_prefix]:
            policy.update_on_hit(h, cache_dict)

        hit_tokens = hit_prefix * req.chunk_size
        recompute_tokens = req.total_tokens - hit_tokens
        latencies.append(cost_model.total_latency(hit_prefix, recompute_tokens))
        total_tokens += req.total_tokens
        total_hit_tokens += hit_tokens

        for i in range(hit_prefix, len(req.chunk_hashes)):
            key = req.chunk_hashes[i]
            if key in cache_dict:
                continue
            chunk_start = i * req.chunk_size
            observed_recompute = max(1, req.total_tokens - chunk_start)

            while len(cache_dict) >= capacity_chunks and cache_dict:
                victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
                if not victims:
                    break
                for v in victims:
                    if cache_dict.pop(v, None) is not None:
                        eviction_count += 1
                    policy.update_on_force_evict(v)

            obj = ChunkObj(
                kv_bytes_per_chunk, observed_recompute_tokens=observed_recompute
            )
            cache_dict[key] = obj
            policy.update_on_put_with_metadata(
                key, cache_obj=obj, observed_recompute_tokens=observed_recompute
            )

    wall_clock = time.perf_counter() - wall_start
    latencies.sort()
    n = len(latencies)
    latency_mean = sum(latencies) / n if n else 0.0

    def _pct(pct: float) -> float:
        if not latencies:
            return 0.0
        idx = min(int(pct / 100 * n), n - 1)
        return latencies[idx]

    return BenchResult(
        policy_name=result_label,
        workload_name=workload_name,
        cache_capacity_bytes=cache_capacity_bytes,
        num_requests=len(requests),
        total_tokens=total_tokens,
        total_hit_tokens=total_hit_tokens,
        token_hit_rate=(total_hit_tokens / total_tokens if total_tokens else 0.0),
        eviction_count=eviction_count,
        wall_clock_seconds=wall_clock,
        requests_per_second=(len(requests) / wall_clock if wall_clock > 0 else 0.0),
        tokens_per_second=(total_tokens / wall_clock if wall_clock > 0 else 0.0),
        rss_delta_bytes=None,
        latency_mean_seconds=latency_mean,
        latency_p50_seconds=_pct(50),
        latency_p95_seconds=_pct(95),
        latency_p99_seconds=_pct(99),
        extra_params={},
    )

# SPDX-License-Identifier: Apache-2.0
"""
Two-tier (small fast + large slow) hierarchical cache simulation.

Direction 4 of the direction-finding experiment. Investigation of
``lmcache/v1/storage_backend/storage_manager.py`` confirmed LMCache's real
multi-tier storage today is write-through (every configured tier is
populated independently at insert time) with no promotion/demotion between
tiers on eviction -- so this models a mechanism that doesn't exist yet in
production, to check whether it's worth building.

The real-data evaluation (Finding 4) diagnosed the failure mode this
targets directly: under real high-fan-out traffic, a single tier's
capacity is far smaller than the concurrently-live working set, so chunks
get evicted (and fully lost) long before a same-conversation second round
could reuse them. Here, a tier1 eviction demotes into tier2 instead of
being dropped; a tier1-miss/tier2-hit promotes back to tier1 and costs a
latency between a tier1 hit and a full recompute -- so a second round that
arrives "too late" for tier1 can still hit tier2 instead of missing
entirely.
"""

# Standard
from typing import Optional
import time

# First Party
from benchmarks.cache_policy.experiments._common import ChunkObj
from lmcache.tools.cache_policy_bench.cost_model import CostModel
from lmcache.tools.cache_policy_bench.runner import BenchResult
from lmcache.tools.cache_policy_bench.workloads import Request
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy


class _Tier:
    """One capacity-bounded tier with its own policy and mutable mapping."""

    def __init__(self, policy: BaseCachePolicy, capacity_chunks: int) -> None:
        self.policy = policy
        self.capacity_chunks = max(1, capacity_chunks)
        self.cache_dict: dict[str, ChunkObj] = policy.init_mutable_mapping()
        self.eviction_count = 0

    def contains(self, key: str) -> bool:
        return key in self.cache_dict

    def touch(self, key: str) -> None:
        self.policy.update_on_hit(key, self.cache_dict)

    def pop(self, key: str) -> Optional[ChunkObj]:
        obj = self.cache_dict.pop(key, None)
        if obj is not None:
            self.policy.update_on_force_evict(key)
        return obj

    def admit(self, key: str, obj: ChunkObj) -> list[tuple[str, ChunkObj]]:
        """Insert ``obj``, evicting (and returning) victims if over capacity."""
        evicted: list[tuple[str, ChunkObj]] = []
        if key in self.cache_dict:
            return evicted
        while len(self.cache_dict) >= self.capacity_chunks and self.cache_dict:
            victims = self.policy.get_evict_candidates(
                self.cache_dict, num_candidates=1
            )
            if not victims:
                break
            for v in victims:
                victim_obj = self.cache_dict.pop(v, None)
                self.policy.update_on_force_evict(v)
                if victim_obj is not None:
                    evicted.append((v, victim_obj))
                    self.eviction_count += 1
        self.cache_dict[key] = obj
        self.policy.update_on_put_with_metadata(
            key, cache_obj=obj, observed_recompute_tokens=obj.observed_recompute_tokens
        )
        return evicted


def run_hierarchical_workload(
    tier1_policy_name: str,
    tier2_policy_name: str,
    requests: list[Request],
    tier1_capacity_bytes: int,
    tier2_capacity_bytes: int,
    kv_bytes_per_chunk: int,
    cost_model: CostModel,
    tier2_hit_multiplier: float = 5.0,
    workload_name: str = "",
) -> BenchResult:
    """
    Replay ``requests`` through a two-tier cache: small/fast tier1 backed
    by a larger/slower tier2 that tier1 evictions demote into.

    Args:
        tier1_policy_name: Eviction policy for the fast tier.
        tier2_policy_name: Eviction policy for the slow (demotion) tier.
        requests: Request sequence to replay, in order.
        tier1_capacity_bytes: Fast-tier capacity in bytes.
        tier2_capacity_bytes: Slow-tier capacity in bytes.
        kv_bytes_per_chunk: Simulated bytes occupied by one cached chunk.
        cost_model: Modeled hit/miss latency function; tier1 hits use
            ``cost_model.hit_latency``, tier2 hits use that scaled by
            ``tier2_hit_multiplier``.
        tier2_hit_multiplier: How much more expensive a tier2 hit is than a
            tier1 hit (still cheaper than a full recompute, modeling e.g. a
            disk/remote fetch vs. a full prefill).
        workload_name: Label recorded on the result.

    Returns:
        A :class:`~lmcache.tools.cache_policy_bench.runner.BenchResult`
        with ``cache_capacity_bytes`` set to the combined tier1+tier2
        budget, for apples-to-apples comparison against single-tier runs
        at the same total memory footprint.

    Raises:
        ValueError: If any capacity argument is non-positive.
    """
    if (
        tier1_capacity_bytes <= 0
        or tier2_capacity_bytes <= 0
        or kv_bytes_per_chunk <= 0
    ):
        raise ValueError("tier capacities and kv_bytes_per_chunk must be positive")

    tier1 = _Tier(
        get_cache_policy(tier1_policy_name), tier1_capacity_bytes // kv_bytes_per_chunk
    )
    tier2 = _Tier(
        get_cache_policy(tier2_policy_name), tier2_capacity_bytes // kv_bytes_per_chunk
    )

    latencies: list[float] = []
    total_tokens = 0
    total_hit_tokens = 0
    wall_start = time.perf_counter()

    for req in requests:
        hit_prefix = 0
        tier1_hits = 0
        tier2_hits = 0
        tier1_hit_keys: list[str] = []
        promotions: list[tuple[str, ChunkObj]] = []

        for h in req.chunk_hashes:
            if tier1.contains(h):
                hit_prefix += 1
                tier1_hits += 1
                tier1_hit_keys.append(h)
            elif tier2.contains(h):
                hit_prefix += 1
                tier2_hits += 1
                obj = tier2.pop(h)
                if obj is not None:
                    promotions.append((h, obj))
            else:
                break

        for h in tier1_hit_keys:
            tier1.touch(h)
        for key, obj in promotions:
            for victim_key, victim_obj in tier1.admit(key, obj):
                tier2.admit(victim_key, victim_obj)

        hit_tokens = hit_prefix * req.chunk_size
        recompute_tokens = req.total_tokens - hit_tokens
        latency = (
            tier1_hits * cost_model.config.retrieval_cost_per_chunk_seconds
            + tier2_hits
            * cost_model.config.retrieval_cost_per_chunk_seconds
            * tier2_hit_multiplier
            + cost_model.miss_latency(recompute_tokens)
        )
        latencies.append(latency)
        total_tokens += req.total_tokens
        total_hit_tokens += hit_tokens

        for i in range(hit_prefix, len(req.chunk_hashes)):
            key = req.chunk_hashes[i]
            chunk_start = i * req.chunk_size
            observed_recompute = max(1, req.total_tokens - chunk_start)
            obj = ChunkObj(
                kv_bytes_per_chunk, observed_recompute_tokens=observed_recompute
            )
            for victim_key, victim_obj in tier1.admit(key, obj):
                tier2.admit(victim_key, victim_obj)

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
        policy_name=f"hier[{tier1_policy_name}+{tier2_policy_name}]",
        workload_name=workload_name,
        cache_capacity_bytes=tier1_capacity_bytes + tier2_capacity_bytes,
        num_requests=len(requests),
        total_tokens=total_tokens,
        total_hit_tokens=total_hit_tokens,
        token_hit_rate=(total_hit_tokens / total_tokens if total_tokens else 0.0),
        eviction_count=tier1.eviction_count + tier2.eviction_count,
        wall_clock_seconds=wall_clock,
        requests_per_second=(len(requests) / wall_clock if wall_clock > 0 else 0.0),
        tokens_per_second=(total_tokens / wall_clock if wall_clock > 0 else 0.0),
        rss_delta_bytes=None,
        latency_mean_seconds=latency_mean,
        latency_p50_seconds=_pct(50),
        latency_p95_seconds=_pct(95),
        latency_p99_seconds=_pct(99),
        extra_params={
            "tier1_capacity_bytes": tier1_capacity_bytes,
            "tier2_capacity_bytes": tier2_capacity_bytes,
            "tier2_hit_multiplier": tier2_hit_multiplier,
            "tier1_evictions": tier1.eviction_count,
            "tier2_evictions": tier2.eviction_count,
        },
    )

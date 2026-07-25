# SPDX-License-Identifier: Apache-2.0
"""
TinyLFU-style admission control, wrapping any existing cache policy.

Direction 3 of the direction-finding experiment. The real-data evaluation
(Finding 4) found that under real, high-fan-out traffic, most chunks are
requested once and never again -- so *which* resident gets evicted matters
less than whether a low-value one-shot chunk gets admitted at all, evicting
a potentially more valuable resident to make room. This tests rejecting
low-estimated-value admissions outright (like Caffeine/TinyLFU's admission
filter) instead of only ranking eviction candidates.

This intentionally reimplements a small simulation loop (not
``lmcache.tools.cache_policy_bench.runner.run_workload``) because admission
control needs to observe every access attempt -- hit or miss -- to build a
useful frequency estimate, and needs an admit/reject decision point
``run_workload`` doesn't have a hook for. It still produces the same
``BenchResult`` shape for apples-to-apples comparison.
"""

# Standard
import time

# First Party
from benchmarks.cache_policy.experiments._common import ChunkObj
from lmcache.tools.cache_policy_bench.cost_model import CostModel
from lmcache.tools.cache_policy_bench.runner import BenchResult
from lmcache.tools.cache_policy_bench.workloads import Request
from lmcache.v1.storage_backend.cache_policy import get_cache_policy


class _FrequencySketch:
    """Approximate, decaying per-key request-frequency counter.

    Not a real Count-Min Sketch (no hashing/collisions modeled) -- an exact
    dict counter with periodic halving to bound memory and let old
    popularity decay, which is sufficient at benchmark scale and keeps the
    TinyLFU comparison honest about the *policy*, not sketch-accuracy
    artifacts.
    """

    def __init__(self, halve_every: int = 20_000) -> None:
        self._counts: dict[str, int] = {}
        self._halve_every = halve_every
        self._increments = 0

    def increment(self, key: str) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1
        self._increments += 1
        if self._increments % self._halve_every == 0:
            for k in list(self._counts):
                halved = self._counts[k] // 2
                if halved > 0:
                    self._counts[k] = halved
                else:
                    del self._counts[k]

    def estimate(self, key: str) -> int:
        return self._counts.get(key, 0)


def run_admission_controlled_workload(
    policy_name: str,
    requests: list[Request],
    cache_capacity_bytes: int,
    kv_bytes_per_chunk: int,
    cost_model: CostModel,
    workload_name: str = "",
    **policy_kwargs: object,
) -> BenchResult:
    """
    Replay ``requests`` through ``policy_name`` with TinyLFU-style
    admission control gating insertions.

    Args, returns: mirror
    :func:`lmcache.tools.cache_policy_bench.runner.run_workload` exactly,
    so results are directly comparable.
    """
    if cache_capacity_bytes <= 0 or kv_bytes_per_chunk <= 0:
        raise ValueError("cache_capacity_bytes and kv_bytes_per_chunk must be positive")

    policy = get_cache_policy(policy_name, **policy_kwargs)
    capacity_chunks = max(1, cache_capacity_bytes // kv_bytes_per_chunk)
    cache_dict: dict[str, ChunkObj] = policy.init_mutable_mapping()
    sketch = _FrequencySketch()

    eviction_count = 0
    rejected_admissions = 0
    latencies: list[float] = []
    total_tokens = 0
    total_hit_tokens = 0

    wall_start = time.perf_counter()

    for req in requests:
        hit_prefix = 0
        for h in req.chunk_hashes:
            sketch.increment(h)
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
            obj = ChunkObj(
                kv_bytes_per_chunk, observed_recompute_tokens=observed_recompute
            )

            if len(cache_dict) >= capacity_chunks:
                victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
                if not victims:
                    continue
                victim_key = victims[0]
                if sketch.estimate(key) <= sketch.estimate(victim_key):
                    rejected_admissions += 1
                    continue
                if cache_dict.pop(victim_key, None) is not None:
                    eviction_count += 1
                policy.update_on_force_evict(victim_key)

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
        policy_name=f"admission[{policy_name}]",
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
        extra_params={"rejected_admissions": rejected_admissions, **policy_kwargs},
    )

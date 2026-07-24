# SPDX-License-Identifier: Apache-2.0
"""
CPU-only, policy-pluggable cache-eviction-policy benchmark runner.

Replays synthetic :mod:`lmcache.tools.cache_policy_bench.workloads` request
sequences through any
:class:`~lmcache.v1.storage_backend.cache_policy.base_policy.BaseCachePolicy`
(driven through the exact same public calls the real storage backend makes
-- see ``lmcache/v1/storage_backend/local_cpu_backend.py``), and reports
hit-rate, modeled-latency, throughput, and process-memory metrics.

No GPU or model inference is involved; see
:mod:`lmcache.tools.cache_policy_bench.cost_model` for what "latency" means
here.

Usage (module mode)::

    python -m lmcache.tools.cache_policy_bench.runner --sweep -o results/
"""

# Standard
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
import argparse
import csv
import json
import time

# First Party
from lmcache.tools.cache_policy_bench.cost_model import CostModel, CostModelConfig
from lmcache.tools.cache_policy_bench.workloads import WORKLOAD_REGISTRY, Request
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

try:
    # Third Party
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_MIB = 2**20

DEFAULT_POLICIES: list[str] = ["LRU", "LFU", "FIFO", "MRU", "COST_AWARE"]
DEFAULT_CACHE_SIZES_MIB: list[float] = [50.0, 100.0, 200.0]
DEFAULT_KV_BYTES_PER_CHUNK = 256 * 1024  # 256 KiB/chunk, a typical KV-chunk size


class _SimulatedChunkObj:
    """Stand-in for a ``MemoryObj``, exposing only what cache policies read."""

    __slots__ = ("can_evict", "observed_recompute_tokens", "_physical_size")

    def __init__(
        self,
        physical_size_bytes: int,
        observed_recompute_tokens: Optional[float] = None,
    ) -> None:
        self._physical_size = physical_size_bytes
        self.observed_recompute_tokens = observed_recompute_tokens
        self.can_evict = True

    def get_physical_size(self) -> int:
        return self._physical_size


class _PolicyCache:
    """Minimal capacity-bounded cache wrapper driving a ``BaseCachePolicy``
    through the same call sequence as ``LocalCPUBackend``."""

    def __init__(self, policy: BaseCachePolicy, capacity_chunks: int) -> None:
        self.policy = policy
        self.capacity_chunks = max(1, capacity_chunks)
        self.cache_dict: dict[str, Any] = policy.init_mutable_mapping()
        self.eviction_count = 0

    def contains(self, key: str) -> bool:
        return key in self.cache_dict

    def touch(self, key: str) -> None:
        self.policy.update_on_hit(key, self.cache_dict)

    def put(self, key: str, obj: _SimulatedChunkObj) -> None:
        if key in self.cache_dict:
            return
        self._ensure_capacity()
        self.cache_dict[key] = obj
        self.policy.update_on_put_with_metadata(
            key,
            cache_obj=obj,
            observed_recompute_tokens=obj.observed_recompute_tokens,
        )

    def _ensure_capacity(self) -> None:
        while len(self.cache_dict) >= self.capacity_chunks and self.cache_dict:
            victims = self.policy.get_evict_candidates(
                self.cache_dict, num_candidates=1
            )
            if not victims:
                break
            for k in victims:
                if self.cache_dict.pop(k, None) is not None:
                    self.eviction_count += 1
                self.policy.update_on_force_evict(k)


@dataclass
class BenchResult:
    """Aggregate metrics from one (policy, workload, cache-size) run."""

    policy_name: str
    workload_name: str
    cache_capacity_bytes: int
    num_requests: int
    total_tokens: int
    total_hit_tokens: int
    token_hit_rate: float
    eviction_count: int
    wall_clock_seconds: float
    requests_per_second: float
    tokens_per_second: float
    rss_delta_bytes: Optional[int]
    latency_mean_seconds: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    latency_p99_seconds: float
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten this result to a JSON/CSV-serializable dict."""
        d = asdict(self)
        extra = d.pop("extra_params")
        d.update({f"param_{k}": v for k, v in extra.items()})
        return d


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(pct / 100 * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


def _rss_bytes() -> Optional[int]:
    if not _HAS_PSUTIL:
        return None
    return psutil.Process().memory_info().rss


def run_workload(
    policy_name: str,
    requests: list[Request],
    cache_capacity_bytes: int,
    kv_bytes_per_chunk: int,
    cost_model: CostModel,
    workload_name: str = "",
    **policy_kwargs: Any,
) -> BenchResult:
    """Replay ``requests`` through ``policy_name`` and collect metrics.

    Args:
        policy_name: Name understood by
            :func:`lmcache.v1.storage_backend.cache_policy.get_cache_policy`
            (e.g. ``"LRU"``, ``"COST_AWARE"``).
        requests: Request sequence to replay, in order.
        cache_capacity_bytes: Total simulated cache capacity in bytes.
        kv_bytes_per_chunk: Simulated bytes occupied by one cached chunk.
        cost_model: Modeled latency function (see :mod:`cost_model`).
        workload_name: Label recorded on the result for reporting.
        **policy_kwargs: Extra constructor kwargs forwarded to
            ``get_cache_policy`` (e.g. ``half_life_seconds`` for
            ``COST_AWARE``; ignored by policies that don't accept them).

    Returns:
        Aggregated :class:`BenchResult` for this run.

    Raises:
        ValueError: If ``cache_capacity_bytes`` or ``kv_bytes_per_chunk`` is
            non-positive.
    """
    if cache_capacity_bytes <= 0 or kv_bytes_per_chunk <= 0:
        raise ValueError("cache_capacity_bytes and kv_bytes_per_chunk must be positive")

    policy = get_cache_policy(policy_name, **policy_kwargs)
    capacity_chunks = max(1, cache_capacity_bytes // kv_bytes_per_chunk)
    cache = _PolicyCache(policy, capacity_chunks)

    latencies: list[float] = []
    total_tokens = 0
    total_hit_tokens = 0

    rss_before = _rss_bytes()
    wall_start = time.perf_counter()

    for req in requests:
        hit_prefix = 0
        for h in req.chunk_hashes:
            if cache.contains(h):
                hit_prefix += 1
            else:
                break

        for h in req.chunk_hashes[:hit_prefix]:
            cache.touch(h)

        hit_tokens = hit_prefix * req.chunk_size
        recompute_tokens = req.total_tokens - hit_tokens
        latencies.append(cost_model.total_latency(hit_prefix, recompute_tokens))

        total_tokens += req.total_tokens
        total_hit_tokens += hit_tokens

        for i in range(hit_prefix, len(req.chunk_hashes)):
            chunk_start = i * req.chunk_size
            observed_recompute = max(1, req.total_tokens - chunk_start)
            cache.put(
                req.chunk_hashes[i],
                _SimulatedChunkObj(
                    kv_bytes_per_chunk, observed_recompute_tokens=observed_recompute
                ),
            )

    wall_clock = time.perf_counter() - wall_start
    rss_after = _rss_bytes()
    rss_delta = (
        None if rss_before is None or rss_after is None else rss_after - rss_before
    )

    latencies.sort()
    n = len(latencies)
    latency_mean = sum(latencies) / n if n else 0.0

    return BenchResult(
        policy_name=policy_name,
        workload_name=workload_name,
        cache_capacity_bytes=cache_capacity_bytes,
        num_requests=len(requests),
        total_tokens=total_tokens,
        total_hit_tokens=total_hit_tokens,
        token_hit_rate=(total_hit_tokens / total_tokens if total_tokens else 0.0),
        eviction_count=cache.eviction_count,
        wall_clock_seconds=wall_clock,
        requests_per_second=(len(requests) / wall_clock if wall_clock > 0 else 0.0),
        tokens_per_second=(total_tokens / wall_clock if wall_clock > 0 else 0.0),
        rss_delta_bytes=rss_delta,
        latency_mean_seconds=latency_mean,
        latency_p50_seconds=_percentile(latencies, 50),
        latency_p95_seconds=_percentile(latencies, 95),
        latency_p99_seconds=_percentile(latencies, 99),
        extra_params=dict(policy_kwargs),
    )


def run_sweep(
    policy_names: list[str],
    workloads: dict[str, list[Request]],
    cache_sizes_bytes: list[int],
    kv_bytes_per_chunk: int = DEFAULT_KV_BYTES_PER_CHUNK,
    cost_model: Optional[CostModel] = None,
) -> list[BenchResult]:
    """Run every (policy x workload x cache-size) combination.

    Args:
        policy_names: Policy names to benchmark.
        workloads: Mapping of workload label to pre-generated request list.
        cache_sizes_bytes: Cache capacities (bytes) to sweep.
        kv_bytes_per_chunk: Simulated bytes per cached chunk.
        cost_model: Latency model; defaults to
            ``CostModel(CostModelConfig())`` if omitted.

    Returns:
        One :class:`BenchResult` per combination, in sweep order.
    """
    model = cost_model or CostModel(CostModelConfig())
    results: list[BenchResult] = []
    for policy_name in policy_names:
        for workload_name, requests in workloads.items():
            for capacity_bytes in cache_sizes_bytes:
                results.append(
                    run_workload(
                        policy_name,
                        requests,
                        capacity_bytes,
                        kv_bytes_per_chunk,
                        model,
                        workload_name=workload_name,
                    )
                )
    return results


def to_csv(results: list[BenchResult], path: Path) -> None:
    """Write ``results`` to a CSV file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in results]
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_json(results: list[BenchResult], path: Path) -> None:
    """Write ``results`` to a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def _default_workloads(quick: bool) -> dict[str, list[Request]]:
    n = 200 if quick else 3000
    if quick:
        return {
            "repetitive_short": WORKLOAD_REGISTRY["repetitive_short"](n, vocab_size=15),
            "novel_long": WORKLOAD_REGISTRY["novel_long"](n // 4),
            "mixed_zipfian": WORKLOAD_REGISTRY["mixed_zipfian"](n, unique_prefixes=40),
            "multi_round_chat": WORKLOAD_REGISTRY["multi_round_chat"](
                10, rounds_per_session=6
            ),
        }
    return {
        "repetitive_short": WORKLOAD_REGISTRY["repetitive_short"](n, vocab_size=100),
        "novel_long": WORKLOAD_REGISTRY["novel_long"](n // 4),
        "mixed_zipfian": WORKLOAD_REGISTRY["mixed_zipfian"](n, unique_prefixes=300),
        "multi_round_chat": WORKLOAD_REGISTRY["multi_round_chat"](
            40, rounds_per_session=12
        ),
    }


def main() -> None:
    """CLI entry point: ``python -m lmcache.tools.cache_policy_bench.runner``."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the cache-eviction-policy benchmark sweep and write CSV/JSON results."
        )
    )
    parser.add_argument("--sweep", action="store_true", help="Run the full sweep")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small request counts (for CI smoke runs)",
    )
    parser.add_argument(
        "--policies", nargs="+", default=DEFAULT_POLICIES, metavar="NAME"
    )
    parser.add_argument(
        "--cache-sizes-mib",
        nargs="+",
        type=float,
        default=DEFAULT_CACHE_SIZES_MIB,
        metavar="MIB",
    )
    parser.add_argument("-o", "--output-dir", default="benchmarks/cache_policy/results")
    args = parser.parse_args()

    if not args.sweep:
        parser.error("Pass --sweep to run the benchmark (no other mode implemented)")

    workloads = _default_workloads(quick=args.quick)
    cache_sizes_bytes = [int(mib * _MIB) for mib in args.cache_sizes_mib]

    print(
        f"Running sweep: policies={args.policies} "
        f"cache_sizes_mib={args.cache_sizes_mib} "
        f"workloads={list(workloads.keys())}"
    )
    results = run_sweep(args.policies, workloads, cache_sizes_bytes)

    out_dir = Path(args.output_dir)
    to_csv(results, out_dir / "sweep_results.csv")
    to_json(results, out_dir / "sweep_results.json")
    print(f"Wrote {len(results)} result rows to {out_dir}/sweep_results.{{csv,json}}")


if __name__ == "__main__":
    main()

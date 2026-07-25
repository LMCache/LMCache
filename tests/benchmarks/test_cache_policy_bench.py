# SPDX-License-Identifier: Apache-2.0
"""Cache-eviction-policy performance benchmarks.

CPU-only, no GPU/model required. The fast, parametrized ``test_bench_*``
cases below are what CI runs on every PR (see ``.github/workflows/test.yml``);
the full parameter sweep across cache sizes lives in ``test_full_sweep``,
marked ``slow`` and run by the nightly workflow instead
(``.github/workflows/cache_policy_benchmark_nightly.yml``).

Run with: ``pytest tests/benchmarks/test_cache_policy_bench.py --benchmark-only``
"""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.tools.cache_policy_bench.cost_model import CostModel, CostModelConfig
from lmcache.tools.cache_policy_bench.runner import (
    DEFAULT_KV_BYTES_PER_CHUNK,
    run_sweep,
    run_workload,
    to_csv,
    to_json,
)
from lmcache.tools.cache_policy_bench.workloads import (
    WORKLOAD_REGISTRY,
    mixed_zipfian,
    multi_round_chat,
    novel_long,
    repetitive_short,
)

POLICIES = ["LRU", "LFU", "FIFO", "MRU", "COST_AWARE"]
# Smoke-test-only coverage; kept separate from POLICIES since that list
# also drives test_full_sweep's scope (nightly), which this isn't meant
# to expand.
_FAST_TEST_POLICIES = [*POLICIES, "ADMISSION_LRU"]

_SMALL_CACHE_BYTES = 4 * 1024 * 1024  # 4 MiB


def _small_workload(name: str) -> list:
    if name == "repetitive_short":
        return repetitive_short(150, vocab_size=15, seed=1)
    if name == "novel_long":
        return novel_long(40, seed=1)
    if name == "mixed_zipfian":
        return mixed_zipfian(150, unique_prefixes=25, seed=1)
    if name == "multi_round_chat":
        return multi_round_chat(6, rounds_per_session=5)
    raise ValueError(f"Unknown workload {name!r}")


@pytest.mark.benchmark(group="cache_policy")
@pytest.mark.parametrize("policy_name", _FAST_TEST_POLICIES)
@pytest.mark.parametrize("workload_name", list(WORKLOAD_REGISTRY))
def test_bench_policy_workload(benchmark, policy_name, workload_name):
    """Fast smoke benchmark: one policy x one workload x one cache size.

    Asserts the run completes and produces sane aggregate metrics; this is
    a regression guard (crashes, nonsensical hit rates), not a correctness
    check -- correctness is covered by ``tests/v1/test_cache_policy.py``.
    """
    requests = _small_workload(workload_name)
    cost_model = CostModel(CostModelConfig())

    def run():
        return run_workload(
            policy_name,
            requests,
            _SMALL_CACHE_BYTES,
            DEFAULT_KV_BYTES_PER_CHUNK,
            cost_model,
            workload_name=workload_name,
        )

    result = benchmark(run)

    assert 0.0 <= result.token_hit_rate <= 1.0
    assert result.eviction_count >= 0
    assert result.num_requests == len(requests)
    assert result.latency_p99_seconds >= result.latency_p50_seconds >= 0.0


@pytest.mark.slow
def test_full_sweep(tmp_path: Path):
    """Full cache-size sweep across all policies and workloads.

    Not run on every PR (see module docstring) -- this generates the
    CSV/JSON artifacts that back the evaluation report.
    """
    workloads = {
        "repetitive_short": repetitive_short(3000, vocab_size=100, seed=1),
        "novel_long": novel_long(750, seed=1),
        "mixed_zipfian": mixed_zipfian(3000, unique_prefixes=300, seed=1),
        "multi_round_chat": multi_round_chat(40, rounds_per_session=12),
    }
    cache_sizes_bytes = [int(mib * 2**20) for mib in (50, 100, 200)]

    results = run_sweep(POLICIES, workloads, cache_sizes_bytes)

    assert len(results) == len(POLICIES) * len(workloads) * len(cache_sizes_bytes)
    for r in results:
        assert 0.0 <= r.token_hit_rate <= 1.0

    to_csv(results, tmp_path / "sweep_results.csv")
    to_json(results, tmp_path / "sweep_results.json")
    assert (tmp_path / "sweep_results.csv").exists()
    assert (tmp_path / "sweep_results.json").exists()

# SPDX-License-Identifier: Apache-2.0
"""Edge-case / adversarial stress tests for cache policies on real ShareGPT data.

Unlike ``tests/benchmarks/test_cache_policy_bench.py`` (synthetic, always
runs), these tests require a real, locally-prepared corpus and are
opt-in only -- they are not wired into any CI workflow (large one-time
download + tokenizer fetch; see ``benchmarks/cache_policy/README.md``).

To run locally, prepare the corpus once (see
``lmcache/tools/cache_policy_bench/sharegpt_workload.py`` module docstring),
then::

    LMCACHE_SHAREGPT_PATH=benchmarks/multi_round_qa/ShareGPT.json \\
        pytest tests/benchmarks/test_cache_policy_bench_real_data.py -v

Without the environment variable set, every test in this module is skipped.
"""

# Standard
from pathlib import Path
import os

# Third Party
import pytest

# First Party
from lmcache.tools.cache_policy_bench.cost_model import CostModel, CostModelConfig
from lmcache.tools.cache_policy_bench.runner import (
    DEFAULT_KV_BYTES_PER_CHUNK,
    run_workload,
)
from lmcache.tools.cache_policy_bench.sharegpt_workload import (
    load_sharegpt_conversations,
    requests_from_conversations,
)

_SHAREGPT_PATH = os.environ.get("LMCACHE_SHAREGPT_PATH")

pytestmark = pytest.mark.skipif(
    not _SHAREGPT_PATH,
    reason="LMCACHE_SHAREGPT_PATH not set -- real-data tests are opt-in local-only",
)

POLICIES = ["LRU", "LFU", "FIFO", "MRU", "COST_AWARE"]
_MIB = 2**20


@pytest.fixture(scope="module")
def conversations() -> list[dict]:
    assert _SHAREGPT_PATH is not None  # narrowed by pytestmark skip
    return load_sharegpt_conversations(Path(_SHAREGPT_PATH))


@pytest.fixture(scope="module")
def cost_model() -> CostModel:
    return CostModel(CostModelConfig())


def test_near_empty_cache(conversations, cost_model):
    """Cache far below one conversation's footprint: no crash, thrash-heavy."""
    requests = requests_from_conversations(
        conversations, chunk_size=256, max_conversations=200, seed=0
    )
    tiny_cache_bytes = 64 * 1024  # far smaller than one conversation's chunks

    for policy_name in POLICIES:
        result = run_workload(
            policy_name,
            requests,
            tiny_cache_bytes,
            DEFAULT_KV_BYTES_PER_CHUNK,
            cost_model,
            workload_name="sharegpt_near_empty",
        )
        assert 0.0 <= result.token_hit_rate <= 1.0
        assert result.eviction_count >= 0
        assert result.token_hit_rate < 0.2, (
            f"{policy_name}: expected near-zero hit rate under a "
            f"far-too-small cache, got {result.token_hit_rate}"
        )


def test_hit_rate_nondecreasing_with_cache_size(conversations, cost_model):
    """Capacity-cliff check: for a fixed sample, more cache should never hurt."""
    requests = requests_from_conversations(
        conversations, chunk_size=256, max_conversations=1500, seed=0
    )
    cache_sizes_bytes = [int(mib * _MIB) for mib in (10, 15, 25, 50, 100)]

    for policy_name in POLICIES:
        hit_rates = []
        for cache_bytes in cache_sizes_bytes:
            result = run_workload(
                policy_name,
                requests,
                cache_bytes,
                DEFAULT_KV_BYTES_PER_CHUNK,
                cost_model,
                workload_name="sharegpt_capacity_cliff",
            )
            hit_rates.append(result.token_hit_rate)
        assert hit_rates == sorted(hit_rates), (
            f"{policy_name}: hit rate should be non-decreasing as cache "
            f"capacity grows, got {hit_rates} for sizes {cache_sizes_bytes}"
        )


def test_pathologically_long_conversations(conversations, cost_model):
    """Replay only the longest real conversations; must not crash or misbehave."""
    longest = sorted(
        conversations,
        key=lambda c: c.get("max_gpt_token", 0),
        reverse=True,
    )[:100]
    requests = requests_from_conversations(
        longest, chunk_size=256, max_conversations=None, seed=0
    )
    assert requests, "expected at least one request from the longest conversations"
    assert max(r.total_tokens for r in requests) > 1000, (
        "sanity check: the 'longest conversations' selection should "
        "actually contain long requests"
    )

    for policy_name in POLICIES:
        result = run_workload(
            policy_name,
            requests,
            50 * _MIB,
            DEFAULT_KV_BYTES_PER_CHUNK,
            cost_model,
            workload_name="sharegpt_long_conversations",
        )
        assert 0.0 <= result.token_hit_rate <= 1.0
        assert result.eviction_count >= 0
        assert result.num_requests == len(requests)


def test_high_fanout_degrades_hit_rate_vs_low_fanout(conversations, cost_model):
    """
    Many concurrent distinct conversations (round-robin) should thrash a
    fixed-size cache harder than few conversations run through -- this
    encodes an empirical finding from the real-data evaluation as a
    regression-checkable invariant, not just a one-off observation.
    """
    cache_bytes = 100 * _MIB

    low_fanout = requests_from_conversations(
        conversations, chunk_size=256, max_conversations=20, seed=1
    )
    high_fanout = requests_from_conversations(
        conversations, chunk_size=256, max_conversations=3000, seed=1
    )

    for policy_name in POLICIES:
        low_result = run_workload(
            policy_name,
            low_fanout,
            cache_bytes,
            DEFAULT_KV_BYTES_PER_CHUNK,
            cost_model,
            workload_name="sharegpt_low_fanout",
        )
        high_result = run_workload(
            policy_name,
            high_fanout,
            cache_bytes,
            DEFAULT_KV_BYTES_PER_CHUNK,
            cost_model,
            workload_name="sharegpt_high_fanout",
        )
        assert low_result.token_hit_rate >= high_result.token_hit_rate, (
            f"{policy_name}: expected low concurrent fan-out ({len(low_fanout)} "
            f"requests) to hit the cache at least as well as high fan-out "
            f"({len(high_fanout)} requests) at a fixed cache size, got "
            f"{low_result.token_hit_rate} vs {high_result.token_hit_rate}"
        )

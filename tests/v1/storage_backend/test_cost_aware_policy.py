# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import MagicMock

# Ensure lmcache.v1.storage_backend.cache_policy.base_policy is loaded directly
base_policy_path = Path(__file__).parents[3] / "lmcache" / "v1" / "storage_backend" / "cache_policy" / "base_policy.py"
spec_base = importlib.util.spec_from_file_location("lmcache.v1.storage_backend.cache_policy.base_policy", str(base_policy_path))
mod_base = importlib.util.module_from_spec(spec_base)
sys.modules["lmcache.v1.storage_backend.cache_policy.base_policy"] = mod_base
spec_base.loader.exec_module(mod_base)

# Mock lmcache.logging if needed
if "lmcache.logging" not in sys.modules:
    mock_logging = MagicMock()
    mock_logging.init_logger.return_value = MagicMock()
    sys.modules["lmcache.logging"] = mock_logging

# Load cost_aware_policy directly
cost_policy_path = Path(__file__).parents[3] / "lmcache" / "v1" / "storage_backend" / "cache_policy" / "cost_aware_policy.py"
spec_cost = importlib.util.spec_from_file_location("lmcache.v1.storage_backend.cache_policy.cost_aware_policy", str(cost_policy_path))
mod_cost = importlib.util.module_from_spec(spec_cost)
sys.modules["lmcache.v1.storage_backend.cache_policy.cost_aware_policy"] = mod_cost
spec_cost.loader.exec_module(mod_cost)

CostAwareEvictionPolicy = mod_cost.CostAwareEvictionPolicy


@dataclass
class MockCacheObject:
    can_evict: bool = True
    chunk_length: int = 256
    storage_tier: str = "CPU"


class TestCostAwareEvictionPolicy(unittest.TestCase):
    def test_cost_aware_score_formula(self):
        policy = CostAwareEvictionPolicy(w1=2.0, w2=0.5, w3=1.0)
        t0 = time.monotonic()

        # Store metadata for key1
        policy.put(key="chunk1", chunk_length=512, storage_tier="RAM")
        # Score = (2.0 * 512) + (0.5 * 10.0) - (1.0 * time_elapsed)
        score = policy.calculate_score("chunk1", current_time=t0 + 2.0)
        expected_score = (2.0 * 512) + (0.5 * 10.0) - (1.0 * 2.0)
        self.assertAlmostEqual(score, expected_score, delta=1e-3)

    def test_eviction_lowest_score_selected(self):
        policy = CostAwareEvictionPolicy(w1=1.0, w2=1.0, w3=1.0)

        # Chunk A: short chunk in GPU (low compute, fast tier -> lower score)
        policy.put("chunkA", chunk_length=64, storage_tier="GPU")

        # Chunk B: long chunk in DISK (high compute, slow tier -> higher score)
        policy.put("chunkB", chunk_length=1024, storage_tier="DISK")

        # Lowest score should be evicted first (chunkA)
        evicted = policy.remove_next()
        self.assertEqual(evicted, "chunkA")

        evicted_second = policy.remove_next()
        self.assertEqual(evicted_second, "chunkB")

    def test_recency_decay(self):
        policy = CostAwareEvictionPolicy(w1=1.0, w2=1.0, w3=10.0)

        t0 = time.monotonic()
        policy.put("old_chunk", chunk_length=512, storage_tier="CPU")
        policy.metadata["old_chunk"].last_access_time = t0 - 100.0

        policy.put("new_chunk", chunk_length=512, storage_tier="CPU")
        policy.metadata["new_chunk"].last_access_time = t0

        # Old chunk should decay heavily and have a lower score -> evicted first
        evicted = policy.remove_next()
        self.assertEqual(evicted, "old_chunk")

    def test_custom_tier_cost_and_tunable_weights(self):
        custom_tier_costs = {"GPU": 1.0, "CPU": 5.0, "SSD": 50.0, "CLOUD": 500.0}
        policy = CostAwareEvictionPolicy(
            w1=0.1,
            w2=2.0,
            w3=0.01,
            storage_tier_cost=custom_tier_costs,
        )

        policy.put("k1", chunk_length=100, storage_tier="SSD")
        policy.put("k2", chunk_length=100, storage_tier="CLOUD")

        # k1 in SSD (cost 50) vs k2 in CLOUD (cost 500). k1 has lower tier cost -> lower score
        evicted = policy.remove_next()
        self.assertEqual(evicted, "k1")

    def test_base_cache_policy_interface(self):
        policy = CostAwareEvictionPolicy()
        cache_dict = policy.init_mutable_mapping()

        obj1 = MockCacheObject(can_evict=True, chunk_length=128, storage_tier="RAM")
        obj2 = MockCacheObject(can_evict=True, chunk_length=1024, storage_tier="DISK")

        cache_dict["k1"] = obj1
        cache_dict["k2"] = obj2

        policy.update_on_put("k1")
        policy.update_on_put("k2")

        # Pass cache_dict to update metadata from objects
        policy.put("k1", value=obj1)
        policy.put("k2", value=obj2)

        candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
        self.assertEqual(candidates, ["k1"])

        policy.update_on_hit("k1", cache_dict)
        policy.update_on_force_evict("k1")
        self.assertNotIn("k1", policy.metadata)


if __name__ == "__main__":
    unittest.main()

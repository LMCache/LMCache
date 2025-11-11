# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    BloomFilter,
    ChunkStatisticsLookupClient,
)


class MockLookupClient(LookupClientInterface):
    """Mock lookup client for testing."""

    def __init__(self):
        self.chunk_size = 256
        self.lookup_calls = []

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        self.lookup_calls.append((token_ids, lookup_id, request_configs))
        return len(token_ids) // 2

    def clear_lookup_status(self, lookup_id: str) -> None:
        pass

    def supports_producer_reuse(self) -> bool:
        return True

    def close(self) -> None:
        pass


class TestBloomFilter:
    """Test suite for BloomFilter functionality."""

    def test_basic_operations(self):
        """Test basic add, contains and clear operations."""
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)

        bf.add("test_item_1")
        bf.add("test_item_2")

        assert bf.contains("test_item_1")
        assert bf.contains("test_item_2")
        assert not bf.contains("test_item_3")

        bf.clear()
        assert not bf.contains("test_item_1")

    def test_false_positive_rate(self):
        """Test false positive rate is within expected range."""
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)

        for i in range(10000):
            bf.add(f"item_{i}")

        false_positives = 0
        test_count = 1000
        for i in range(10000, 10000 + test_count):
            if bf.contains(f"item_{i}"):
                false_positives += 1

        fp_rate = false_positives / test_count
        assert fp_rate < 0.05, f"False positive rate {fp_rate} is too high"

    def test_memory_metrics(self):
        """Test memory usage metrics."""
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)

        stats = bf.get_statistics()
        assert "size_mb" in stats
        assert "hash_count" in stats
        assert "item_count" in stats
        assert "bits_set" in stats
        assert "fill_rate" in stats
        assert stats["size_mb"] > 0


class TestChunkStatisticsBasic:
    """Test suite for basic chunk statistics functionality."""

    def test_basic_statistics_collection(self):
        """Test basic statistics collection with multiple requests."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(
            chunk_size=256,
            chunk_statistics_expected_chunks=1000,
            chunk_statistics_false_positive_rate=0.01,
        )
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()

        token_ids_1 = list(range(512))
        token_ids_2 = list(range(256))

        result1 = stats_client.lookup(token_ids_1, "req_1")
        result2 = stats_client.lookup(token_ids_2, "req_2")

        assert result1 == 256
        assert result2 == 128

        stats = stats_client.get_statistics()
        assert stats["enabled"] is True
        assert stats["total_requests"] == 2
        assert stats["total_chunks"] == 3
        assert stats["unique_chunks"] == 2

    def test_preemption_handling(self):
        """Test that preempted requests are skipped."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()

        token_ids = list(range(256))
        stats_client.lookup(token_ids, "req_1")
        stats1 = stats_client.get_statistics()

        stats_client.lookup(token_ids, "req_1")
        stats2 = stats_client.get_statistics()

        assert stats1["total_requests"] == stats2["total_requests"]
        assert stats1["total_chunks"] == stats2["total_chunks"]

    def test_torch_tensor_input(self):
        """Test statistics with torch.Tensor input."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()

        token_ids = torch.arange(512)
        stats_client.lookup(token_ids, "req_1")

        stats = stats_client.get_statistics()
        assert stats["total_requests"] == 1
        assert stats["total_chunks"] == 2

    def test_disabled_statistics(self):
        """Test that statistics are not collected when disabled."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.lookup(list(range(256)), "req_1")

        stats = stats_client.get_statistics()
        assert stats["enabled"] is False
        assert stats["total_requests"] == 0

    def test_interface_delegation(self):
        """Test delegation of interface methods."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()
        stats_client.lookup(list(range(256)), "req_1")

        stats_client.clear_lookup_status("req_1")
        assert stats_client.supports_producer_reuse() is True
        stats_client.close()


class TestChunkStatisticsMetrics:
    """Test suite for chunk statistics metrics and calculations."""

    def test_reuse_rate_calculation(self):
        """Test chunk reuse rate calculation."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()

        stats_client.lookup(list(range(512)), "req_1")
        stats_client.lookup(list(range(256)), "req_2")

        stats = stats_client.get_statistics()
        assert stats["total_chunks"] == 3
        assert stats["reuse_rate"] >= 0.0

    def test_detailed_metrics(self):
        """Test detailed statistics metrics including Bloom Filter info."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(
            chunk_size=256,
            chunk_statistics_expected_chunks=5000,
            chunk_statistics_false_positive_rate=0.01,
        )
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()

        stats_client.lookup(list(range(512)), "req_1")
        stats_client.lookup(list(range(256)), "req_2")

        stats = stats_client.get_statistics()

        assert "enabled" in stats
        assert "total_requests" in stats
        assert "total_chunks" in stats
        assert "unique_chunks" in stats
        assert "duplicate_chunks" in stats
        assert "reuse_rate" in stats
        assert "bloom_filter" in stats

        bf_stats = stats["bloom_filter"]
        assert "size_mb" in bf_stats
        assert "hash_count" in bf_stats
        assert "item_count" in bf_stats
        assert "bits_set" in bf_stats
        assert "fill_rate" in bf_stats
        assert "expected_elements" in bf_stats
        assert "false_positive_rate" in bf_stats

        assert stats["total_requests"] == 2
        assert stats["total_chunks"] == 3
        assert stats["duplicate_chunks"] >= 0
        assert 0.0 <= stats["reuse_rate"] <= 1.0
        assert bf_stats["expected_elements"] == 5000
        assert bf_stats["false_positive_rate"] == 0.01

    def test_progressive_metrics(self):
        """Test metrics update progressively with more requests."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()

        stats_client.lookup(list(range(256)), "req_1")
        stats1 = stats_client.get_statistics()
        assert stats1["total_requests"] == 1
        assert stats1["total_chunks"] == 1

        stats_client.lookup(list(range(256, 512)), "req_2")
        stats2 = stats_client.get_statistics()
        assert stats2["total_requests"] == 2
        assert stats2["total_chunks"] == 2

        stats_client.lookup(list(range(256)), "req_3")
        stats3 = stats_client.get_statistics()
        assert stats3["total_requests"] == 3
        assert stats3["total_chunks"] == 3
        assert stats3["duplicate_chunks"] > 0

    def test_memory_efficiency(self):
        """Test memory efficiency of Bloom Filter."""
        config = LMCacheEngineConfig(
            chunk_size=256,
            chunk_statistics_expected_chunks=100000,
            chunk_statistics_false_positive_rate=0.01,
        )
        mock_client = MockLookupClient()
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()

        for i in range(100):
            stats_client.lookup(list(range(i * 256, (i + 1) * 256)), f"req_{i}")

        stats = stats_client.get_statistics()
        bf_stats = stats["bloom_filter"]

        assert bf_stats["size_mb"] < 1.0
        assert stats["total_requests"] == 100
        assert stats["total_chunks"] == 100


class TestChunkStatisticsLifecycle:
    """Test suite for statistics lifecycle management."""

    def test_reset_statistics(self):
        """Test statistics reset."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()
        stats_client.lookup(list(range(256)), "req_1")

        stats_client.reset_statistics()
        stats = stats_client.get_statistics()

        assert stats["total_requests"] == 0
        assert stats["total_chunks"] == 0
        assert stats["unique_chunks"] == 0

    def test_auto_start_configuration(self):
        """Test auto_start_statistics configuration."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(
            chunk_size=256,
            enable_chunk_statistics=True,
            chunk_statistics_auto_start_statistics=True,
        )
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats = stats_client.get_statistics()
        assert stats["enabled"] is True
        assert stats["total_requests"] == 0

    def test_auto_exit_configuration(self):
        """Test auto exit configuration."""
        mock_client = MockLookupClient()

        config = LMCacheEngineConfig(
            chunk_size=256,
            enable_chunk_statistics=True,
            chunk_statistics_auto_exit_timeout_hours=1.0,
        )
        stats_client = ChunkStatisticsLookupClient(mock_client, config)
        assert stats_client.enable_auto_exit is True
        assert stats_client.timeout_hours == 1.0

        config2 = LMCacheEngineConfig(
            chunk_size=256,
            enable_chunk_statistics=True,
            chunk_statistics_auto_exit_timeout_hours=0.0,
        )
        stats_client2 = ChunkStatisticsLookupClient(mock_client, config2)
        assert stats_client2.enable_auto_exit is False

    def test_timing_statistics(self):
        """Test timing statistics collection."""
        mock_client = MockLookupClient()
        config = LMCacheEngineConfig(chunk_size=256)
        stats_client = ChunkStatisticsLookupClient(mock_client, config)

        stats_client.start_statistics()
        stats_client.lookup(list(range(512)), "req_1")
        stats_client.lookup(list(range(256)), "req_2")

        stats = stats_client.get_statistics()

        assert "timing" in stats
        timing = stats["timing"]
        assert "lookup_time_seconds" in timing
        assert "record_statistics_time_seconds" in timing
        assert "check_exit_conditions_time_seconds" in timing
        assert "total_time_seconds" in timing
        assert "overhead_time_seconds" in timing
        assert "overhead_percentage" in timing

        assert timing["lookup_time_seconds"] > 0
        assert timing["total_time_seconds"] > 0
        assert timing["overhead_time_seconds"] >= 0
        assert 0 <= timing["overhead_percentage"] <= 100

        total = (
            timing["lookup_time_seconds"]
            + timing["record_statistics_time_seconds"]
            + timing["check_exit_conditions_time_seconds"]
        )
        assert abs(timing["total_time_seconds"] - total) < 0.001

        overhead = (
            timing["record_statistics_time_seconds"]
            + timing["check_exit_conditions_time_seconds"]
        )
        assert abs(timing["overhead_time_seconds"] - overhead) < 0.001

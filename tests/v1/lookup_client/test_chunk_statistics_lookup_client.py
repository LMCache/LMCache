# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.record_strategies import _get_strategies
from lmcache.v1.utils.bloom_filter import BloomFilter


class BaseMockClient(LookupClientInterface):
    """Base mock client with common functionality."""

    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size

    def clear_lookup_status(self, lookup_id: str) -> None:
        pass

    def supports_producer_reuse(self) -> bool:
        return True

    def close(self) -> None:
        pass


class MockLookupClient(BaseMockClient):
    """Mock lookup client for testing."""

    def __init__(self):
        super().__init__()
        self.lookup_calls = []

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        self.lookup_calls.append((token_ids, lookup_id, request_configs))
        return len(token_ids) // 2


class FastMissLookupClient(BaseMockClient):
    """Mock lookup client that returns immediately on first chunk miss."""

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        time.sleep(0.008)  # Sleep for 8ms to simulate actual lookup time
        return 0


class BaseTestCase:
    """Base test case with common functionality for chunk statistics tests."""

    def create_stats_client(self, **kwargs):
        """Create a ChunkStatisticsLookupClient with given configuration."""
        config = LMCacheEngineConfig(**kwargs)
        mock_client = MockLookupClient()
        return ChunkStatisticsLookupClient(mock_client, config)

    def setup_stats_client(self, **kwargs):
        """Setup statistics client with default configuration and start statistics."""
        default_kwargs = {
            "enable_chunk_statistics": True,
            "chunk_statistics_strategy": "memory_bloom_filter",
            "chunk_statistics_expected_chunks": 1000,
            "chunk_statistics_false_positive_rate": 0.01,
        }
        default_kwargs.update(kwargs)

        stats_client = self.create_stats_client(**default_kwargs)
        stats_client.start_statistics()
        return stats_client, stats_client.actual_lookup_client


class TestStrategyDiscovery:
    """Test suite for strategy discovery functionality."""

    def test_get_strategies_discovers_all_strategies(self):
        """Test that _get_strategies discovers all available strategies."""
        strategies = _get_strategies()

        assert isinstance(strategies, dict)
        assert len(strategies) >= 2

        assert "file_hash" in strategies
        assert "memory_bloom_filter" in strategies

        assert strategies["file_hash"].name() == "file_hash"
        assert strategies["memory_bloom_filter"].name() == "memory_bloom_filter"


class TestBloomFilter:
    """Test suite for BloomFilter functionality."""

    def test_bloom_filter_operations(self):
        """Test basic BloomFilter operations."""
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)

        # Test add/contains
        bf.add("test_item_1")
        bf.add("test_item_2")
        assert bf.contains("test_item_1")
        assert bf.contains("test_item_2")
        assert not bf.contains("test_item_3")

        # Test clear
        bf.clear()
        assert not bf.contains("test_item_1")

    def test_false_positive_rate(self):
        """Test false positive rate is within expected range."""
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)

        # Add 10000 items
        for i in range(10000):
            bf.add(f"item_{i}")

        # Test 1000 non-existent items
        false_positives = sum(
            1 for i in range(10000, 11000) if bf.contains(f"item_{i}")
        )
        fp_rate = false_positives / 1000
        assert fp_rate < 0.05, f"False positive rate {fp_rate} is too high"

    def test_memory_metrics(self):
        """Test memory usage metrics."""
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)
        stats = bf.get_statistics()

        required_metrics = [
            "size_mb",
            "hash_count",
            "item_count",
            "bits_set",
            "fill_rate",
        ]
        for metric in required_metrics:
            assert metric in stats
        assert stats["size_mb"] > 0


class TestChunkStatisticsBasic(BaseTestCase):
    """Test suite for basic chunk statistics functionality."""

    def test_preemption_handling(self):
        """Test that preempted requests are skipped."""
        stats_client, _ = self.setup_stats_client()

        token_ids = list(range(256))
        stats_client.lookup(token_ids, "req_1")
        stats1 = stats_client.get_statistics()

        # Same request ID should be skipped
        stats_client.lookup(token_ids, "req_1")
        stats2 = stats_client.get_statistics()

        assert stats1["total_requests"] == stats2["total_requests"]
        assert stats1["total_chunks"] == stats2["total_chunks"]

    def test_disabled_statistics(self):
        """Test that statistics are not collected when disabled."""
        stats_client = self.create_stats_client()
        stats_client.lookup(list(range(256)), "req_1")

        stats = stats_client.get_statistics()
        assert stats["enabled"] is False
        assert stats["total_requests"] == 0


class TestChunkStatisticsMetrics(BaseTestCase):
    """Test suite for chunk statistics metrics and calculations."""

    def test_detailed_metrics(self):
        """Test detailed statistics metrics including Bloom Filter info."""
        stats_client, _ = self.setup_stats_client(
            chunk_statistics_expected_chunks=5000,
            chunk_statistics_false_positive_rate=0.01,
        )

        stats_client.lookup(list(range(512)), "req_1")
        stats_client.lookup(list(range(256)), "req_2")

        stats = stats_client.get_statistics()

        # Check top-level metrics
        required_stats = [
            "enabled",
            "total_requests",
            "total_chunks",
            "unique_chunks",
            "duplicate_chunks",
            "reuse_rate",
            "bloom_filter",
            "timing",
        ]
        for stat in required_stats:
            assert stat in stats

        # Check Bloom Filter metrics
        bf_stats = stats["bloom_filter"]
        required_bf_stats = [
            "size_mb",
            "hash_count",
            "item_count",
            "bits_set",
            "fill_rate",
            "expected_elements",
            "false_positive_rate",
        ]
        for stat in required_bf_stats:
            assert stat in bf_stats

        assert stats["total_requests"] == 2
        assert stats["total_chunks"] == 3
        assert stats["duplicate_chunks"] >= 0
        assert 0.0 <= stats["reuse_rate"] <= 1.0
        assert bf_stats["expected_elements"] == 5000
        assert bf_stats["false_positive_rate"] == 0.01

        timing = stats["timing"]
        required_timing_fields = [
            "lookup_time_seconds",
            "record_statistics_time_seconds",
            "check_exit_conditions_time_seconds",
            "total_time_seconds",
            "overhead_time_seconds",
            "overhead_percentage",
        ]

        for field in required_timing_fields:
            assert field in timing
            assert timing[field] >= 0

        assert 0 <= timing["overhead_percentage"] <= 100

        # Verify timing calculations
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

    def test_progressive_metrics(self):
        """Test metrics update progressively with more requests."""
        stats_client, _ = self.setup_stats_client()

        # First request
        stats_client.lookup(list(range(256)), "req_1")
        stats1 = stats_client.get_statistics()
        assert stats1["total_requests"] == 1
        assert stats1["total_chunks"] == 1

        # Second request
        stats_client.lookup(list(range(256, 512)), "req_2")
        stats2 = stats_client.get_statistics()
        assert stats2["total_requests"] == 2
        assert stats2["total_chunks"] == 2

        # Third request with duplicate chunk
        # Test with torch.Tensor
        stats_client.lookup(torch.arange(256), "req_3")
        stats3 = stats_client.get_statistics()
        assert stats3["total_requests"] == 3
        assert stats3["total_chunks"] == 3
        assert stats3["duplicate_chunks"] > 0

    def test_memory_efficiency(self):
        """Test memory efficiency of Bloom Filter."""
        stats_client, _ = self.setup_stats_client(
            chunk_statistics_expected_chunks=100000,
            chunk_statistics_false_positive_rate=0.01,
        )

        # Add 100 unique chunks
        for i in range(100):
            stats_client.lookup(list(range(i * 256, (i + 1) * 256)), f"req_{i}")

        stats = stats_client.get_statistics()
        bf_stats = stats["bloom_filter"]

        assert bf_stats["size_mb"] < 1.0  # Should be memory efficient
        assert stats["total_requests"] == 100
        assert stats["total_chunks"] == 100


class TestChunkStatisticsLifecycle(BaseTestCase):
    """Test suite for statistics lifecycle management."""

    def test_reset_statistics(self):
        """Test statistics reset."""
        stats_client, _ = self.setup_stats_client()
        stats_client.lookup(list(range(256)), "req_1")

        stats_client.reset_statistics()
        stats = stats_client.get_statistics()

        assert stats["total_requests"] == 0
        assert stats["total_chunks"] == 0
        assert stats["unique_chunks"] == 0

    def test_auto_exit_configuration(self):
        """Test auto exit configuration."""
        # Test with timeout enabled
        stats_client = self.create_stats_client(
            enable_chunk_statistics=True,
            chunk_statistics_auto_start_statistics=True,
            chunk_statistics_auto_exit_timeout_hours=1.0,
        )
        stats = stats_client.get_statistics()
        assert stats["enabled"] is True
        assert stats["total_requests"] == 0

        assert stats_client.enable_auto_exit is True
        assert stats_client.timeout_hours == 1.0

        # Test with timeout disabled
        stats_client2 = self.create_stats_client(
            enable_chunk_statistics=True,
            chunk_statistics_auto_exit_timeout_hours=0.0,
        )
        assert stats_client2.enable_auto_exit is False


class TestChunkStatisticsPerformance:
    """Test suite for chunk statistics performance validation."""

    @pytest.mark.parametrize(
        "strategy_name,async_enabled,async_preprocess_chunks",
        [
            ("memory_bloom_filter", False, False),
            ("memory_bloom_filter", True, True),
            ("memory_bloom_filter", True, False),
            ("file_hash", False, False),
            ("file_hash", True, True),
            ("file_hash", True, False),
        ],
    )
    def test_worst_case_overhead(
        self, strategy_name, async_enabled, async_preprocess_chunks
    ):
        """Test worst case performance with different strategies and configs."""
        self._run_performance_test(
            strategy_name=strategy_name,
            async_enabled=async_enabled,
            async_preprocess_chunks=async_preprocess_chunks,
        )

    def _run_performance_test(
        self,
        strategy_name: str,
        async_enabled: bool,
        async_preprocess_chunks: bool,
    ):
        """
        Test worst case performance with different configurations.

        Simulates the scenario where:
        - Large request with 32K tokens (128 chunks with chunk_size=256)
        - Actual lookup returns immediately on first chunk miss
        - Statistics recording still processes all chunks

        Validates that overhead stays within x% in realistic workload.
        """

        # Determine test configuration description
        if not async_enabled:
            mode_desc = f"{strategy_name} (Sync)"
        elif async_preprocess_chunks:
            mode_desc = f"{strategy_name} (Async Preprocessed)"
        else:
            mode_desc = f"{strategy_name} (Async Raw Tokens)"

        # Create temporary directory for file_hash strategy
        temp_dir = None
        if strategy_name == "file_hash":
            temp_dir = tempfile.mkdtemp(prefix="lmcache_test_")

        try:
            mock_client = FastMissLookupClient()
            config_kwargs = {
                "chunk_size": 256,
                "chunk_statistics_strategy": strategy_name,
                "chunk_statistics_expected_chunks": 100000,
                "chunk_statistics_false_positive_rate": 0.01,
                "chunk_statistics_async_enabled": async_enabled,
                "chunk_statistics_async_preprocess_chunks": async_preprocess_chunks,
            }
            if temp_dir:
                config_kwargs["chunk_statistics_file_output_dir"] = temp_dir

            config = LMCacheEngineConfig(**config_kwargs)
            stats_client = ChunkStatisticsLookupClient(mock_client, config)
            stats_client.start_statistics()

            # Test with 32K tokens (128 chunks) - more realistic large request
            token_count = 32 * 1024
            token_ids = list(range(token_count))

            # Warm up
            stats_client.lookup(token_ids, "warmup")
            stats_client.reset_statistics()
            stats_client.start_statistics()

            # Run test with multiple requests to get stable measurements
            num_requests = 30
            for i in range(num_requests):
                stats_client.lookup(token_ids, f"req_{i}")

            # Wait for async processing to complete
            if async_enabled:
                assert stats_client.wait_for_async_processing(timeout=10.0), (
                    "Async processing timeout"
                )

            stats = stats_client.get_statistics()
            timing = stats["timing"]

            # Calculate metrics
            overhead_percentage = timing["overhead_percentage"]
            record_time = timing["record_statistics_time_seconds"]
            avg_record_ms = record_time / num_requests * 1000
            avg_lookup_ms = timing["lookup_time_seconds"] / num_requests * 1000

            # Print performance statistics (always print, even when test passes)
            print("\n" + "=" * 80)
            print(f"Performance Test Results - {mode_desc}:")
            print("=" * 80)
            print("Configuration:")
            print(f"  Strategy: {strategy_name}")
            print(f"  Async enabled: {async_enabled}")
            print(f"  Async preprocess: {async_preprocess_chunks}")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Token count per request: {token_count}")
            print(f"  Chunks per request: {token_count // 256}")
            if async_enabled:
                async_queue = stats.get("async_queue", {})
                print(f"  Queue max size: {async_queue.get('max_size_reached', 'N/A')}")
                print(f"  Queue full blocks: {async_queue.get('full_blocks', 'N/A')}")
            print("-" * 80)
            print(f"Lookup time: {timing['lookup_time_seconds']:.6f}s")
            print(f"  Avg per request: {avg_lookup_ms:.2f}ms")
            print(f"  Expected (sleep): {8.0 * num_requests:.2f}ms")
            print(f"Record time: {timing['record_statistics_time_seconds']:.6f}s")
            print(f"  Avg per request: {avg_record_ms:.2f}ms")
            print(
                f"Check exit time: {timing['check_exit_conditions_time_seconds']:.6f}s"
            )
            print(f"Total time: {timing['total_time_seconds']:.6f}s")
            print("-" * 80)
            print(f"Overhead time: {timing['overhead_time_seconds']:.6f}s")
            print(f"Overhead percentage: {overhead_percentage:.2f}%")
            print("=" * 80 + "\n")

            # Validate statistics
            assert stats["total_requests"] == num_requests
            expected_chunks = num_requests * (token_count // 256)
            assert stats["total_chunks"] == expected_chunks

            # Validate overhead is within 40%
            assert overhead_percentage <= 40.0, (
                f"Overhead {overhead_percentage:.2f}% exceeds 40% threshold "
                f"{mode_desc}. "
                f"Lookup time: {timing['lookup_time_seconds']:.6f}s, "
                f"Record time: {timing['record_statistics_time_seconds']:.6f}s, "
                f"Check exit time: "
                f"{timing['check_exit_conditions_time_seconds']:.6f}s"
            )

            # Additional validation: record time should be reasonable
            avg_record_time_per_request = (
                timing["record_statistics_time_seconds"] / num_requests
            )
            assert avg_record_time_per_request < 0.01, (
                f"Average record time per request "
                f"{avg_record_time_per_request:.6f}s "
                f"is too high (should be < 10ms) for {mode_desc}"
            )

            # Additional async-specific validations
            if async_enabled:
                async_queue = stats.get("async_queue", {})
                # Only validate async_queue if strategy provides it
                if async_queue:
                    assert async_queue.get("enabled") is True
                    assert (
                        async_queue.get("capacity")
                        == config.chunk_statistics_async_queue_capacity
                    )
                    # Queue should not have been full in our test
                    assert async_queue.get("full_blocks", 0) == 0
        finally:
            # Clean up temporary directory for file_hash strategy
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass


class TestFileHashStrategy:
    """Test suite for file_hash strategy."""

    def test_file_hash_basic(self):
        """Test file_hash strategy basic functionality."""
        temp_dir = tempfile.mkdtemp()
        try:
            config = LMCacheEngineConfig.from_dict(
                {
                    "chunk_statistics_enabled": True,
                    "chunk_statistics_strategy": "file_hash",
                    "chunk_statistics_file_output_dir": temp_dir,
                }
            )

            mock_client = MockLookupClient()
            client = ChunkStatisticsLookupClient(mock_client, config)
            client.start_statistics()

            token_ids = list(range(512))
            client.lookup(token_ids, "test_request_1")

            client.wait_for_async_processing(timeout=2.0)

            # Standard
            from pathlib import Path
            import json

            output_files = list(Path(temp_dir).glob("*.jsonl"))
            assert len(output_files) > 0

            with open(output_files[0], "r") as f:
                line = f.readline()
                data = json.loads(line)

                assert "chunk_hashes" in data
                assert "lookup_id" in data
                assert "timestamp" in data
                assert len(data["chunk_hashes"]) == 2

            client.close()
        finally:
            shutil.rmtree(temp_dir)

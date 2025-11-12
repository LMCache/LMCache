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


class FastMissLookupClient(LookupClientInterface):
    """Mock lookup client that returns immediately on first chunk miss."""

    def __init__(self):
        self.chunk_size = 256

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        # Sleep for 8ms to simulate actual lookup time
        # This ensures lookup dominates the timing to measure overhead accurately
        time.sleep(0.008)
        return 0

    def clear_lookup_status(self, lookup_id: str) -> None:
        pass

    def supports_producer_reuse(self) -> bool:
        return True

    def close(self) -> None:
        pass


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

    def test_get_strategies_returns_strategy_classes(self):
        """Test that discovered strategies are proper RecordStrategy classes."""
        strategies = _get_strategies()

        for strategy_name, strategy_class in strategies.items():
            assert hasattr(strategy_class, "name")
            assert hasattr(strategy_class, "record")
            assert hasattr(strategy_class, "get_statistics")
            assert hasattr(strategy_class, "reset")
            assert hasattr(strategy_class, "wait_for_async_processing")
            assert hasattr(strategy_class, "close")

            assert callable(strategy_class.name)
            assert strategy_class.name() == strategy_name

    def test_get_strategies_caching(self):
        """Test that _get_strategies returns cached results."""
        strategies1 = _get_strategies()
        strategies2 = _get_strategies()

        assert strategies1 is strategies2


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

        # Wait for async processing to complete
        assert stats_client.wait_for_async_processing(timeout=2.0), (
            "Async processing timeout"
        )

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

        assert timing["lookup_time_seconds"] >= 0
        assert timing["total_time_seconds"] >= 0
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

        Validates that overhead stays within 15% in realistic workload.
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

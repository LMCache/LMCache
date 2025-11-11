# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional, Union
import hashlib
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


class BloomFilter:
    """
    A simple Bloom Filter implementation for memory-efficient set membership testing.

    Args:
        expected_elements: Expected number of elements to store
        false_positive_rate: Desired false positive rate (default 0.01 = 1%)
    """

    def __init__(
        self, expected_elements: int = 1000000, false_positive_rate: float = 0.01
    ):
        # Standard

        # Calculate optimal bit array size and number of hash functions
        self.size = self._optimal_size(expected_elements, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_elements)
        self.bit_array = [False] * self.size
        self.item_count = 0  # Track number of items added
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

        logger.info(
            "BloomFilter initialized with size=%d bits "
            "(%.2f MB), "
            "hash_count=%d, "
            "expected_elements=%d, "
            "false_positive_rate=%f",
            self.size,
            self.size / 8 / 1024 / 1024,
            self.hash_count,
            expected_elements,
            false_positive_rate,
        )

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        # Standard
        import math

        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        # Standard
        import math

        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _hashes(self, item: Union[str, int]) -> list[int]:
        """Generate multiple hash values for an item."""
        result = []
        if isinstance(item, int):
            for i in range(self.hash_count):
                h = hashlib.sha256(
                    item.to_bytes(8, "big", signed=True) + i.to_bytes(4, "big")
                ).digest()
                result.append(int.from_bytes(h[:4], "big") % self.size)
        else:
            for i in range(self.hash_count):
                h = hashlib.sha256(f"{item}_{i}".encode()).digest()
                result.append(int.from_bytes(h[:4], "big") % self.size)
        return result

    def add(self, item: Union[str, int]) -> None:
        """Add an item to the Bloom Filter."""
        for pos in self._hashes(item):
            self.bit_array[pos] = True
        self.item_count += 1

    def contains(self, item: Union[str, int]) -> bool:
        """Check if an item might be in the set (may have false positives)."""
        return all(self.bit_array[pos] for pos in self._hashes(item))

    def clear(self) -> None:
        """Clear all items from the Bloom Filter."""
        self.bit_array = [False] * self.size
        self.item_count = 0

    def get_memory_usage_bytes(self) -> int:
        """Get the memory usage of the Bloom Filter in bytes."""
        # Each boolean in Python list takes ~28 bytes (object overhead)
        # But bit_array is a list of booleans, actual memory is size / 8
        return self.size // 8

    def get_statistics(self) -> dict:
        """Get Bloom Filter statistics."""
        bits_set = sum(self.bit_array)
        fill_rate = bits_set / self.size if self.size > 0 else 0.0
        size_bytes = self.get_memory_usage_bytes()
        return {
            "size_mb": size_bytes / 1024 / 1024,
            "hash_count": self.hash_count,
            "item_count": self.item_count,
            "bits_set": bits_set,
            "fill_rate": fill_rate,
            "expected_elements": self.expected_elements,
            "false_positive_rate": self.false_positive_rate,
        }


class ChunkStatisticsLookupClient(LookupClientInterface):
    """
    A wrapper lookup client that tracks chunk reuse statistics.

    This client wraps an actual lookup client and records statistics about
    chunk hits and reuse patterns. It uses a Bloom Filter for memory-efficient
    tracking of unique chunks per request.

    Key features:
    - Tracks unique chunks per request using Bloom Filter (90%+ accuracy)
    - Detects and skips preempted requests (same request arriving multiple times)
    - Records chunk hit statistics
    - Provides statistics query and export capabilities
    """

    def __init__(
        self,
        actual_lookup_client: LookupClientInterface,
        config: LMCacheEngineConfig,
    ):
        expected_chunks = config.chunk_statistics_expected_chunks
        false_positive_rate = config.chunk_statistics_false_positive_rate
        self.actual_lookup_client = actual_lookup_client
        self.lock = threading.RLock()
        self.chunk_size = config.chunk_size

        # Statistics tracking
        self.enabled = False
        self.request_seen: set[str] = set()  # Track request IDs to detect preemption
        # Global Bloom Filter for tracking unique chunks across all requests
        self.global_bloom = BloomFilter(expected_chunks, false_positive_rate)
        self.total_chunks = 0
        self.unique_chunks_count = 0

        # Timing statistics
        self.lookup_time = 0.0  # Time spent in actual lookup
        self.record_time = 0.0  # Time spent recording statistics
        self.check_exit_time = 0.0  # Time spent checking exit conditions

        # Bloom Filter parameters
        self.expected_chunks = expected_chunks
        self.false_positive_rate = false_positive_rate

        # Auto exit condition tracking
        self.start_time = 0.0
        self.timeout_hours = config.chunk_statistics_auto_exit_timeout_hours
        self.target_unique_chunks = (
            config.chunk_statistics_auto_exit_target_unique_chunks
        )
        self.enable_auto_exit = (
            self.timeout_hours > 0.0 or self.target_unique_chunks is not None
        )

        # Auto start statistics if configured
        auto_start_statistics = config.chunk_statistics_auto_start_statistics
        if auto_start_statistics:
            self.start_statistics()

        logger.info(
            "ChunkStatisticsLookupClient initialized with "
            "expected_chunks=%d, "
            "false_positive_rate=%f, "
            "auto_start_statistics=%s, "
            "auto_exit_timeout_hours=%f, "
            "auto_exit_target_unique_chunks=%s",
            expected_chunks,
            false_positive_rate,
            auto_start_statistics,
            self.timeout_hours,
            self.target_unique_chunks,
        )

    def start_statistics(self) -> None:
        """Start collecting statistics."""
        with self.lock:
            self.enabled = True
            self.start_time = 0.0  # Initialize start time lazily on first lookup
            logger.info("Chunk statistics collection started")

    def stop_statistics(self) -> None:
        """Stop collecting statistics."""
        with self.lock:
            self.enabled = False
            logger.info("Chunk statistics collection stopped")

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        with self.lock:
            self.request_seen.clear()
            self.global_bloom.clear()
            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.lookup_time = 0.0
            self.record_time = 0.0
            self.check_exit_time = 0.0
            logger.info("Chunk statistics reset")

    def get_statistics(self) -> dict:
        """
        Get current statistics.

        Returns:
            Dictionary containing:
            - enabled: Whether statistics collection is enabled
            - total_requests: Total number of unique requests processed
            - total_chunks: Total number of chunks processed
            - unique_chunks: Number of unique chunks (estimated)
            - duplicate_chunks: Number of duplicate chunks (estimated)
            - reuse_rate: Chunk reuse rate (0.0 to 1.0)
            - bloom_filter: Bloom Filter statistics including memory usage
            - timing: Timing statistics (if enabled)
        """
        with self.lock:
            duplicate_chunks = self.total_chunks - self.unique_chunks_count
            reuse_rate = (
                duplicate_chunks / self.total_chunks if self.total_chunks > 0 else 0.0
            )

            total_time = self.lookup_time + self.record_time + self.check_exit_time
            overhead_time = self.record_time + self.check_exit_time
            overhead_percentage = (
                (overhead_time / total_time * 100.0) if total_time > 0 else 0.0
            )

            return {
                "enabled": self.enabled,
                "total_requests": len(self.request_seen),
                "total_chunks": self.total_chunks,
                "unique_chunks": self.unique_chunks_count,
                "duplicate_chunks": duplicate_chunks,
                "reuse_rate": reuse_rate,
                "bloom_filter": self.global_bloom.get_statistics(),
                "timing": {
                    "lookup_time_seconds": self.lookup_time,
                    "record_statistics_time_seconds": self.record_time,
                    "check_exit_conditions_time_seconds": self.check_exit_time,
                    "total_time_seconds": total_time,
                    "overhead_time_seconds": overhead_time,
                    "overhead_percentage": overhead_percentage,
                },
            }

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        # Call actual lookup client and track time
        start_time = time.time()
        result = self.actual_lookup_client.lookup(token_ids, lookup_id, request_configs)
        with self.lock:
            self.lookup_time += time.time() - start_time

        # Record statistics if enabled
        if self.enabled:
            start_time = time.time()
            self._record_statistics(token_ids, lookup_id)
            with self.lock:
                self.record_time += time.time() - start_time

            start_time = time.time()
            self._check_exit_conditions()
            with self.lock:
                self.check_exit_time += time.time() - start_time

        return result

    def _record_statistics(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
    ) -> None:
        """Record statistics for a lookup operation."""
        with self.lock:
            # Check if this is a preempted request (seen before)
            if lookup_id in self.request_seen:
                logger.debug("Skipping statistics for preempted request: %s", lookup_id)
                return

            # Mark request as seen
            self.request_seen.add(lookup_id)

            # Convert token_ids to list if needed
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()

            # Get chunk size from actual client or use default
            num_chunks = (len(token_ids) + self.chunk_size - 1) // self.chunk_size

            # Calculate cumulative prefix hash for each chunk
            prefix_hash = 0
            unique_chunks_in_request = 0
            for i in range(num_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, len(token_ids))
                chunk_tokens = tuple(token_ids[start_idx:end_idx])

                # Cumulative prefix hash: hash(prefix_hash, current_chunk)
                prefix_hash = hash((prefix_hash, chunk_tokens))

                # Check if this chunk is globally unique
                if not self.global_bloom.contains(prefix_hash):
                    self.global_bloom.add(prefix_hash)
                    unique_chunks_in_request += 1

            # Update statistics
            self.total_chunks += num_chunks
            self.unique_chunks_count += unique_chunks_in_request

            # Update observability metrics
            self._update_observability_metrics()

            logger.debug(
                "Request %s: %d chunks, %d unique globally",
                lookup_id,
                num_chunks,
                unique_chunks_in_request,
            )

    def _update_observability_metrics(self) -> None:
        """Update observability metrics with current statistics."""
        try:
            monitor = LMCStatsMonitor.GetOrCreate()
            stats = self.get_statistics()
            monitor.update_chunk_statistics(stats)
        except Exception as e:
            logger.debug("Failed to update observability metrics: %s", e)

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Clear lookup status."""
        self.actual_lookup_client.clear_lookup_status(lookup_id)

    def supports_producer_reuse(self) -> bool:
        return self.actual_lookup_client.supports_producer_reuse()

    def close(self) -> None:
        self.actual_lookup_client.close()

    def _check_exit_conditions(self) -> None:
        """Check exit conditions and trigger statistics stop if any condition is met."""
        if not self.enable_auto_exit:
            return

        # Initialize start time on first check
        if self.start_time == 0.0:
            self.start_time = time.time()

        stop_reason = None

        # Check timeout
        if self.timeout_hours > 0.0:
            elapsed_hours = (time.time() - self.start_time) / 3600.0
            if elapsed_hours >= self.timeout_hours:
                stop_reason = "Timeout reached: %.2fh >= %.2fh" % (
                    elapsed_hours,
                    self.timeout_hours,
                )

        # Check unique chunks target
        if self.target_unique_chunks is not None:
            if self.unique_chunks_count >= self.target_unique_chunks:
                stop_reason = "Target unique chunks reached: %d >= %d" % (
                    self.unique_chunks_count,
                    self.target_unique_chunks,
                )

        if stop_reason:
            logger.warning("LMCache auto-stop triggered: %s", stop_reason)
            self._trigger_stop(stop_reason)

    def _trigger_stop(self, reason: str) -> None:
        """Trigger statistics stop when exit conditions are met."""
        logger.warning("LMCache auto-stop triggered: %s", reason)

        # Stop statistics collection if active
        if self.enabled:
            self.stop_statistics()
            logger.info(
                "Chunk statistics collection automatically stopped due to "
                "exit conditions"
            )

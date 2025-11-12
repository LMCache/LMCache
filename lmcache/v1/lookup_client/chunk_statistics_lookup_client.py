# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional, Union
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.record_strategies import (
    RecordStrategy,
    create_record_strategy,
)

logger = init_logger(__name__)


class ChunkStatisticsLookupClient(LookupClientInterface):
    """
    A wrapper lookup client that tracks chunk reuse statistics.

    This client wraps an actual lookup client and records statistics about
    chunk hits and reuse patterns using a configurable recording strategy.

    Key features:
    - Configurable recording strategy (memory, file, etc.)
    - Detects and skips preempted requests (same request arriving multiple times)
    - Records chunk hit statistics
    - Provides statistics query and export capabilities
    """

    def __init__(
        self,
        actual_lookup_client: LookupClientInterface,
        config: LMCacheEngineConfig,
        record_strategy: Optional[RecordStrategy] = None,
    ):
        self.actual_lookup_client = actual_lookup_client
        self.lock = threading.RLock()
        self.chunk_size = config.chunk_size

        # Statistics tracking
        self.enabled = False
        self.request_seen: set[str] = set()  # Track request IDs to detect preemption

        # Timing statistics
        self.lookup_time = 0.0  # Time spent in actual lookup
        self.record_time = 0.0  # Time spent recording statistics
        self.check_exit_time = 0.0  # Time spent checking exit conditions

        # Auto exit condition tracking
        self.start_time = 0.0
        self.timeout_hours = config.chunk_statistics_auto_exit_timeout_hours
        self.target_unique_chunks = (
            config.chunk_statistics_auto_exit_target_unique_chunks
        )
        self.enable_auto_exit = (
            self.timeout_hours > 0.0 or self.target_unique_chunks is not None
        )

        # Initialize recording strategy - explicitly typed as RecordStrategy
        self.record_strategy: RecordStrategy
        if record_strategy is None:
            # Use factory function to create strategy based on configuration
            self.record_strategy = create_record_strategy(
                strategy_name=config.chunk_statistics_strategy,
                chunk_size=config.chunk_size,
                config=config,
            )
        else:
            self.record_strategy = record_strategy

        # Auto start statistics if configured
        auto_start_statistics = config.chunk_statistics_auto_start_statistics
        if auto_start_statistics:
            self.start_statistics()

        logger.info(
            "ChunkStatisticsLookupClient initialized with "
            "record_strategy=%s, "
            "auto_start_statistics=%s, "
            "auto_exit_timeout_hours=%f, "
            "auto_exit_target_unique_chunks=%s",
            type(self.record_strategy).__name__,
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
        # Wait for async processing to complete before reset
        self.record_strategy.wait_for_async_processing(timeout=5.0)

        with self.lock:
            self.request_seen.clear()
            self.record_strategy.reset()

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
            - bloom_filter: Bloom Filter statistics
            - timing: Timing statistics (if enabled)
            - async_queue: Async queue statistics (if enabled)
        """
        # Wait for async processing to complete before returning statistics
        self.record_strategy.wait_for_async_processing(timeout=5.0)

        with self.lock:
            strategy_stats = self.record_strategy.get_statistics()

            total_time = self.lookup_time + self.record_time + self.check_exit_time
            overhead_time = self.record_time + self.check_exit_time
            overhead_percentage = (
                (overhead_time / total_time * 100.0) if total_time > 0 else 0.0
            )

            # Build backward compatible statistics structure
            result = {
                "enabled": self.enabled,
                "total_requests": len(self.request_seen),
                "timing": {
                    "lookup_time_seconds": self.lookup_time,
                    "record_statistics_time_seconds": self.record_time,
                    "check_exit_conditions_time_seconds": self.check_exit_time,
                    "total_time_seconds": total_time,
                    "overhead_time_seconds": overhead_time,
                    "overhead_percentage": overhead_percentage,
                },
                "total_chunks": strategy_stats.get("total_chunks", 0),
                "unique_chunks": strategy_stats.get("unique_chunks", 0),
                "duplicate_chunks": strategy_stats.get("duplicate_chunks", 0),
                "reuse_rate": strategy_stats.get("reuse_rate", 0.0),
                **{
                    k: v
                    for k, v in strategy_stats.items()
                    if k in ("bloom_filter", "async_queue", "file_hash")
                },
            }

            return result

    def wait_for_async_processing(self, timeout: float = 5.0) -> bool:
        """
        Wait for async processing to complete.

        This is a proxy method that delegates to the record strategy.
        Maintained for backward compatibility with existing tests.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if processing is complete, False if timeout
        """
        return self.record_strategy.wait_for_async_processing(timeout)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        # Call actual lookup client and track time
        start_time = time.time()
        result = self.actual_lookup_client.lookup(token_ids, lookup_id, request_configs)
        end_time = time.time()
        with self.lock:
            self.lookup_time += end_time - start_time

        # Record statistics if enabled
        if self.enabled:
            start_time = time.time()
            # Quick check for preempted request
            with self.lock:
                if lookup_id in self.request_seen:
                    logger.debug(
                        "Skipping statistics for preempted request: %s", lookup_id
                    )
                    return result
                self.request_seen.add(lookup_id)

            # Use strategy to record statistics
            self.record_strategy.record(token_ids, lookup_id)
            end_time = time.time()
            with self.lock:
                self.record_time += end_time - start_time

            start_time = time.time()
            self._check_exit_conditions()
            end_time = time.time()
            with self.lock:
                self.check_exit_time += end_time - start_time

        return result

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Clear lookup status."""
        self.actual_lookup_client.clear_lookup_status(lookup_id)

    def supports_producer_reuse(self) -> bool:
        return self.actual_lookup_client.supports_producer_reuse()

    def close(self) -> None:
        # Stop statistics and wait for strategy to finish
        if self.enabled:
            self.stop_statistics()
        self.record_strategy.close()
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
            strategy_stats = self.record_strategy.get_statistics()
            unique_chunks = strategy_stats.get("unique_chunks", 0)
            if unique_chunks >= self.target_unique_chunks:
                stop_reason = "Target unique chunks reached: %d >= %d" % (
                    unique_chunks,
                    self.target_unique_chunks,
                )

        if stop_reason:
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

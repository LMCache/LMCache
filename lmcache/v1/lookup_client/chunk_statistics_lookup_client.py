# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional, Union
import array
import hashlib
import queue
import struct
import threading
import time

# Third Party
import torch
import xxhash

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.utils.bloom_filter import BloomFilter

logger = init_logger(__name__)


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

        # Async queue configuration
        self.async_enabled = config.chunk_statistics_async_enabled
        self.async_queue_capacity = config.chunk_statistics_async_queue_capacity
        self.async_preprocess_chunks = config.chunk_statistics_async_preprocess_chunks
        self.async_queue: Optional[queue.Queue] = None
        self.async_worker_thread: Optional[threading.Thread] = None
        self.async_shutdown = False
        self.queue_full_blocks = 0  # Track how many times queue was full
        self.queue_max_size = 0  # Track maximum queue size reached

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
            "auto_exit_target_unique_chunks=%s, "
            "async_enabled=%s, "
            "async_queue_capacity=%d, "
            "async_preprocess_chunks=%s",
            expected_chunks,
            false_positive_rate,
            auto_start_statistics,
            self.timeout_hours,
            self.target_unique_chunks,
            self.async_enabled,
            self.async_queue_capacity,
            self.async_preprocess_chunks,
        )

    def start_statistics(self) -> None:
        """Start collecting statistics."""
        with self.lock:
            self.enabled = True
            self.start_time = 0.0  # Initialize start time lazily on first lookup

            # Start async worker if enabled
            if self.async_enabled and self.async_worker_thread is None:
                self.async_queue = queue.Queue(maxsize=self.async_queue_capacity)
                self.async_shutdown = False
                self.async_worker_thread = threading.Thread(
                    target=self._async_worker, daemon=True, name="ChunkStatisticsWorker"
                )
                self.async_worker_thread.start()
                logger.info(
                    "Chunk statistics async worker started with queue capacity=%d",
                    self.async_queue_capacity,
                )

            logger.info("Chunk statistics collection started")

    def stop_statistics(self) -> None:
        """Stop collecting statistics."""
        with self.lock:
            self.enabled = False

            # Stop async worker if running
            if self.async_worker_thread is not None:
                self.async_shutdown = True
                # Put sentinel value to wake up worker
                if self.async_queue is not None:
                    try:
                        self.async_queue.put(None, block=False)
                    except queue.Full:
                        pass

            logger.info("Chunk statistics collection stopped")

        # Wait for worker thread to finish (outside lock)
        if self.async_worker_thread is not None:
            self.async_worker_thread.join(timeout=5.0)
            if self.async_worker_thread.is_alive():
                logger.warning("Async worker thread did not stop within timeout")
            else:
                logger.info("Async worker thread stopped")
            self.async_worker_thread = None
            self.async_queue = None

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        # Wait for async processing to complete before reset
        if self.async_enabled and self.async_queue is not None:
            self.wait_for_async_processing(timeout=5.0)

        with self.lock:
            self.request_seen.clear()
            self.global_bloom.clear()
            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.lookup_time = 0.0
            self.record_time = 0.0
            self.check_exit_time = 0.0
            self.queue_full_blocks = 0
            self.queue_max_size = 0

            # Clear async queue if exists
            if self.async_queue is not None:
                while not self.async_queue.empty():
                    try:
                        self.async_queue.get_nowait()
                    except queue.Empty:
                        break

            logger.info("Chunk statistics reset")

    def wait_for_async_processing(self, timeout: float = 5.0) -> bool:
        """
        Wait for async queue to be processed.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if queue is empty, False if timeout
        """
        if not self.async_enabled or self.async_queue is None:
            return True

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.async_queue.empty():
                # Give a bit more time for worker to finish processing
                time.sleep(0.01)
                if self.async_queue.empty():
                    return True
            time.sleep(0.01)

        return self.async_queue.empty()

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
        """
        # Wait for async processing to complete before returning statistics
        if self.async_enabled and self.async_queue is not None:
            self.wait_for_async_processing(timeout=5.0)

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

            # Get async queue metrics
            queue_size = 0
            if self.async_queue is not None:
                queue_size = self.async_queue.qsize()
                self.queue_max_size = max(self.queue_max_size, queue_size)

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
                "async_queue": {
                    "enabled": self.async_enabled,
                    "capacity": self.async_queue_capacity,
                    "current_size": queue_size,
                    "max_size_reached": self.queue_max_size,
                    "full_blocks": self.queue_full_blocks,
                    "utilization": queue_size / self.async_queue_capacity
                    if self.async_queue_capacity > 0
                    else 0.0,
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
        end_time = time.time()
        with self.lock:
            self.lookup_time += end_time - start_time

        # Record statistics if enabled
        if self.enabled:
            start_time = time.time()
            if self.async_enabled:
                self._record_statistics_async(token_ids, lookup_id)
            else:
                self._record_statistics(token_ids, lookup_id)
            end_time = time.time()
            with self.lock:
                self.record_time += end_time - start_time

            start_time = time.time()
            self._check_exit_conditions()
            end_time = time.time()
            with self.lock:
                self.check_exit_time += end_time - start_time

        return result

    def _record_statistics_async(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
    ) -> None:
        """Record statistics asynchronously by queuing the request."""
        # Quick check for preempted request (with lock)
        with self.lock:
            if lookup_id in self.request_seen:
                logger.debug("Skipping statistics for preempted request: %s", lookup_id)
                return
            self.request_seen.add(lookup_id)

        # Convert token_ids to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Choose queuing strategy based on configuration
        if self.async_preprocess_chunks:
            # Strategy 1: Pre-process chunks and compute hashes before queuing
            # This reduces queue memory usage by chunk_size factor
            self._queue_preprocessed_chunks(token_ids, lookup_id)
        else:
            # Strategy 2: Queue raw token_ids for processing in background
            # This minimizes main thread overhead but uses more queue memory
            self._queue_raw_tokens(token_ids, lookup_id)

    def _queue_preprocessed_chunks(
        self,
        token_ids: list[int],
        lookup_id: str,
    ) -> None:
        """Pre-process chunks and queue hash positions (memory efficient)."""
        token_count = len(token_ids)
        num_chunks = (token_count + self.chunk_size - 1) // self.chunk_size
        chunk_data_list = []

        # Hybrid hash optimization: try fastest method first
        # Method 1: Python built-in hash (3.4x faster than sha256, minimal memory)
        try:
            prefix_hash = 0
            # For small chunks, built-in hash is fastest and most memory efficient
            if self.chunk_size <= 512:
                for i in range(num_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, token_count)
                    chunk_slice = token_ids[start_idx:end_idx]

                    # Use built-in hash for maximum speed
                    chunk_hash = (
                        hash((prefix_hash, tuple(chunk_slice))) & 0xFFFFFFFFFFFFFFFF
                    )
                    prefix_hash = chunk_hash
                    positions = self.global_bloom._hashes(chunk_hash)
                    chunk_data_list.append(positions)
            else:
                # Fall through to xxhash for large chunks
                raise ImportError("Chunk too large for built-in hash optimization")

        except (ImportError, TypeError):
            # Method 2: xxhash for better performance
            try:
                prefix_hash_int = 0

                for i in range(num_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, token_count)
                    chunk_slice = token_ids[start_idx:end_idx]

                    # Use xxhash for faster hashing
                    h = xxhash.xxh64()
                    h.update(prefix_hash_int.to_bytes(8, "big", signed=False))

                    # Use array.array for much faster byte conversion
                    token_array = array.array("i", chunk_slice)
                    h.update(token_array.tobytes())

                    prefix_hash_int = h.intdigest()
                    positions = self.global_bloom._hashes(prefix_hash_int)
                    chunk_data_list.append(positions)

            except ImportError:
                # Method 3: Fallback to optimized sha256 implementation
                prefix_hash_bytes = b""
                # Pre-allocate format string for better performance
                chunk_format = f">{self.chunk_size}i"

                for i in range(num_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, token_count)
                    chunk_slice = token_ids[start_idx:end_idx]

                    # Reuse hasher object when possible
                    h = hashlib.sha256()
                    h.update(prefix_hash_bytes)

                    # Optimize struct.pack for full chunks
                    if len(chunk_slice) == self.chunk_size:
                        h.update(struct.pack(chunk_format, *chunk_slice))
                    else:
                        h.update(struct.pack(f">{len(chunk_slice)}i", *chunk_slice))

                    digest = h.digest()
                    prefix_hash_bytes = digest[:8]

                    prefix_hash = int.from_bytes(prefix_hash_bytes, "big", signed=False)
                    positions = self.global_bloom._hashes(prefix_hash)
                    chunk_data_list.append(positions)

        # Queue the pre-processed chunks for async processing
        if self.async_queue is not None:
            try:
                self.async_queue.put(
                    (chunk_data_list, lookup_id), block=True, timeout=10.0
                )
            except queue.Full:
                with self.lock:
                    self.queue_full_blocks += 1
                logger.warning(
                    "Async queue full (capacity=%d), blocking until space available",
                    self.async_queue_capacity,
                )
                # Block until space is available
                self.async_queue.put((chunk_data_list, lookup_id), block=True)

    def _queue_raw_tokens(
        self,
        token_ids: list[int],
        lookup_id: str,
    ) -> None:
        """Queue raw token_ids for background processing (min main thread overhead)."""
        # Queue the raw token_ids for async processing
        if self.async_queue is not None:
            try:
                self.async_queue.put((token_ids, lookup_id), block=True, timeout=10.0)
            except queue.Full:
                with self.lock:
                    self.queue_full_blocks += 1
                logger.warning(
                    "Async queue full (capacity=%d), blocking until space available",
                    self.async_queue_capacity,
                )
                # Block until space is available
                self.async_queue.put((token_ids, lookup_id), block=True)

    def _async_worker(self) -> None:
        """Background worker that processes statistics asynchronously."""
        logger.info("Async statistics worker started")

        if self.async_queue is None:
            return

        while not self.async_shutdown:
            try:
                # Get item from queue with timeout
                item = self.async_queue.get(timeout=0.1)

                # Check for sentinel value (shutdown signal)
                if item is None:
                    break

                # Determine data format based on configuration
                if self.async_preprocess_chunks:
                    # Pre-processed chunk data format: (chunk_data_list, lookup_id)
                    chunk_data_list, lookup_id = item
                    self._process_chunk_data(chunk_data_list, lookup_id)
                else:
                    # Raw token data format: (token_ids, lookup_id)
                    token_ids, lookup_id = item
                    self._process_statistics(token_ids, lookup_id)

                self.async_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error in async statistics worker: %s", e, exc_info=True)

        # Process remaining items in queue before shutdown
        while not self.async_queue.empty():
            try:
                item = self.async_queue.get_nowait()
                if item is not None:
                    if self.async_preprocess_chunks:
                        chunk_data_list, lookup_id = item
                        self._process_chunk_data(chunk_data_list, lookup_id)
                    else:
                        token_ids, lookup_id = item
                        self._process_statistics(token_ids, lookup_id)
                self.async_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error("Error processing remaining items: %s", e)

        logger.info("Async statistics worker stopped")

    def _process_chunk_data(
        self,
        chunk_data_list: list[list[int]],
        lookup_id: str,
    ) -> None:
        """Process pre-computed chunk data (called by async worker)."""
        num_chunks = len(chunk_data_list)

        # Acquire lock and update bloom filter and statistics
        unique_chunks_in_request = 0
        with self.lock:
            bit_array = self.global_bloom.bit_array
            for positions in chunk_data_list:
                # Inline contains check for better performance
                is_new = False
                for pos in positions:
                    idx = pos >> 5
                    bit = pos & 31
                    if not (bit_array[idx] & (1 << bit)):
                        is_new = True
                        break

                if is_new:
                    # Add to bloom filter
                    for pos in positions:
                        idx = pos >> 5
                        bit = pos & 31
                        bit_array[idx] |= 1 << bit
                    unique_chunks_in_request += 1

            # Update item count
            self.global_bloom.item_count += unique_chunks_in_request
            self.total_chunks += num_chunks
            self.unique_chunks_count += unique_chunks_in_request

        # Update observability metrics (skip for performance)
        if num_chunks > 100 or unique_chunks_in_request > 50:
            self._update_observability_metrics()

        logger.debug(
            "Request %s: %d chunks, %d unique globally",
            lookup_id,
            num_chunks,
            unique_chunks_in_request,
        )

    def _process_statistics(
        self,
        token_ids: list[int],
        lookup_id: str,
    ) -> None:
        """Process statistics for a lookup operation (called by async worker)."""
        # Calculate all hashes and bloom filter positions
        token_count = len(token_ids)
        num_chunks = (token_count + self.chunk_size - 1) // self.chunk_size
        chunk_bloom_positions = []
        prefix_hash_bytes = b""

        # Use struct for faster byte packing
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, token_count)
            chunk_slice = token_ids[start_idx:end_idx]

            # Fast hash: pack all tokens at once using struct
            h = hashlib.sha256()
            h.update(prefix_hash_bytes)
            h.update(struct.pack(f">{len(chunk_slice)}i", *chunk_slice))

            digest = h.digest()
            prefix_hash_bytes = digest[:8]

            # Pre-compute bloom filter positions using first 8 bytes as hash
            prefix_hash = int.from_bytes(prefix_hash_bytes, "big", signed=False)
            positions = self.global_bloom._hashes(prefix_hash)
            chunk_bloom_positions.append(positions)

        # Now acquire lock and update bloom filter and statistics
        unique_chunks_in_request = 0
        with self.lock:
            bit_array = self.global_bloom.bit_array
            for positions in chunk_bloom_positions:
                # Inline contains check for better performance
                is_new = False
                for pos in positions:
                    idx = pos >> 5
                    bit = pos & 31
                    if not (bit_array[idx] & (1 << bit)):
                        is_new = True
                        break

                if is_new:
                    # Add to bloom filter
                    for pos in positions:
                        idx = pos >> 5
                        bit = pos & 31
                        bit_array[idx] |= 1 << bit
                    unique_chunks_in_request += 1

            # Update item count
            self.global_bloom.item_count += unique_chunks_in_request
            self.total_chunks += num_chunks
            self.unique_chunks_count += unique_chunks_in_request

        self._update_observability_metrics()

        logger.debug(
            "Request %s: %d chunks, %d unique globally",
            lookup_id,
            num_chunks,
            unique_chunks_in_request,
        )

    def _record_statistics(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
    ) -> None:
        """Record statistics for a lookup operation (synchronous version)."""
        # Quick check for preempted request (with lock)
        with self.lock:
            if lookup_id in self.request_seen:
                logger.debug("Skipping statistics for preempted request: %s", lookup_id)
                return
            self.request_seen.add(lookup_id)

        # Convert token_ids to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Process statistics synchronously
        self._process_statistics(token_ids, lookup_id)

    def _update_observability_metrics(self) -> None:
        """Update observability metrics with current statistics."""
        try:
            monitor = LMCStatsMonitor.GetOrCreate()
            # Only update basic metrics, avoid expensive bloom filter statistics
            with self.lock:
                duplicate_chunks = self.total_chunks - self.unique_chunks_count
                reuse_rate = (
                    duplicate_chunks / self.total_chunks
                    if self.total_chunks > 0
                    else 0.0
                )
                stats = {
                    "enabled": self.enabled,
                    "total_requests": len(self.request_seen),
                    "total_chunks": self.total_chunks,
                    "unique_chunks": self.unique_chunks_count,
                    "duplicate_chunks": duplicate_chunks,
                    "reuse_rate": reuse_rate,
                }
            monitor.update_chunk_statistics(stats)
        except Exception as e:
            logger.debug("Failed to update observability metrics: %s", e)

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Clear lookup status."""
        self.actual_lookup_client.clear_lookup_status(lookup_id)

    def supports_producer_reuse(self) -> bool:
        return self.actual_lookup_client.supports_producer_reuse()

    def close(self) -> None:
        # Stop statistics and wait for async worker to finish
        if self.enabled:
            self.stop_statistics()
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

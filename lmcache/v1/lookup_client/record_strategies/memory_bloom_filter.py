# SPDX-License-Identifier: Apache-2.0

# Standard
import hashlib
import queue
import struct

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.record_strategies.base import RecordStrategy
from lmcache.v1.utils.bloom_filter import BloomFilter

logger = init_logger(__name__)


class MemoryBloomFilterStrategy(RecordStrategy):
    """
    Memory-based recording strategy using Bloom Filter for chunk deduplication.

    This is the original implementation extracted into a strategy.
    """

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this strategy."""
        return "memory_bloom_filter"

    def __init__(self, config, chunk_size: int):
        """Initialize from configuration object."""
        super().__init__(
            chunk_size=chunk_size,
            async_enabled=config.chunk_statistics_async_enabled,
            async_queue_capacity=config.chunk_statistics_async_queue_capacity,
        )

        self.async_preprocess_chunks = config.chunk_statistics_async_preprocess_chunks

        # Bloom filter for tracking unique chunks
        self.global_bloom = BloomFilter(
            config.chunk_statistics_expected_chunks,
            config.chunk_statistics_false_positive_rate,
        )
        self.total_chunks = 0
        self.unique_chunks_count = 0
        self.queue_max_size = 0

    def _record_async(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics asynchronously."""
        if self.async_preprocess_chunks:
            chunk_data_list = self._preprocess_chunks(token_ids)
            self._queue_item((chunk_data_list, lookup_id))
        else:
            self._queue_item((token_ids, lookup_id))

    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics synchronously."""
        self._process_statistics(token_ids, lookup_id)

    def _compute_chunk_hash(
        self, prefix_hash_bytes: bytes, chunk_slice: list[int]
    ) -> bytes:
        """Compute hash for a single chunk."""
        h = hashlib.sha256()
        h.update(prefix_hash_bytes)
        h.update(struct.pack(f">{len(chunk_slice)}i", *chunk_slice))
        return h.digest()[:8]

    def _preprocess_chunks(self, token_ids: list[int]) -> list[list[int]]:
        """Pre-process chunks and return hash positions (memory efficient)."""
        token_count = len(token_ids)
        num_chunks = (token_count + self.chunk_size - 1) // self.chunk_size
        chunk_data_list = []
        prefix_hash_bytes = b""

        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, token_count)
            chunk_slice = token_ids[start_idx:end_idx]

            prefix_hash_bytes = self._compute_chunk_hash(prefix_hash_bytes, chunk_slice)
            prefix_hash = int.from_bytes(prefix_hash_bytes, "big", signed=False)
            positions = self.global_bloom._hashes(prefix_hash)
            chunk_data_list.append(positions)

        return chunk_data_list

    def _process_queue_item(self, item) -> None:
        """Process a single item from the queue."""
        if self.async_preprocess_chunks:
            chunk_data_list, lookup_id = item
            self._process_chunk_data(chunk_data_list, lookup_id)
        else:
            token_ids, lookup_id = item
            self._process_statistics(token_ids, lookup_id)

    def _async_worker(self) -> None:
        """Background worker that processes statistics asynchronously."""
        logger.info("Async statistics worker started")

        if self.async_queue is None:
            return

        while not self.async_shutdown:
            try:
                item = self.async_queue.get(timeout=0.1)
                if item is None:
                    break
                self._process_queue_item(item)
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
                    self._process_queue_item(item)
                self.async_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error("Error processing remaining items: %s", e)

        logger.info("Async statistics worker stopped")

    def _update_bloom_filter(self, positions_list: list[list[int]]) -> int:
        """Update bloom filter with positions and return unique count."""
        unique_count = 0
        bit_array = self.global_bloom.bit_array

        for positions in positions_list:
            # Check if chunk is new
            is_new = any(
                not (bit_array[pos >> 5] & (1 << (pos & 31))) for pos in positions
            )

            if is_new:
                # Add to bloom filter
                for pos in positions:
                    bit_array[pos >> 5] |= 1 << (pos & 31)
                unique_count += 1

        return unique_count

    def _process_chunk_data(
        self,
        chunk_data_list: list[list[int]],
        lookup_id: str,
    ) -> None:
        """Process pre-computed chunk data (called by async worker)."""
        num_chunks = len(chunk_data_list)

        with self.lock:
            unique_chunks_in_request = self._update_bloom_filter(chunk_data_list)
            self.global_bloom.item_count += unique_chunks_in_request
            self.total_chunks += num_chunks
            self.unique_chunks_count += unique_chunks_in_request

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
        """Process statistics for a lookup operation (synchronous version)."""
        chunk_bloom_positions = self._preprocess_chunks(token_ids)
        num_chunks = len(chunk_bloom_positions)

        with self.lock:
            unique_chunks_in_request = self._update_bloom_filter(chunk_bloom_positions)
            self.global_bloom.item_count += unique_chunks_in_request
            self.total_chunks += num_chunks
            self.unique_chunks_count += unique_chunks_in_request

        logger.debug(
            "Request %s: %d chunks, %d unique globally",
            lookup_id,
            num_chunks,
            unique_chunks_in_request,
        )

    def get_statistics(self) -> dict:
        """Get current statistics from this strategy."""
        with self.lock:
            duplicate_chunks = self.total_chunks - self.unique_chunks_count
            reuse_rate = (
                duplicate_chunks / self.total_chunks if self.total_chunks > 0 else 0.0
            )

            # Get async queue metrics
            queue_size = 0
            if self.async_queue is not None:
                queue_size = self.async_queue.qsize()
                self.queue_max_size = max(self.queue_max_size, queue_size)

            return {
                "total_chunks": self.total_chunks,
                "unique_chunks": self.unique_chunks_count,
                "duplicate_chunks": duplicate_chunks,
                "reuse_rate": reuse_rate,
                "bloom_filter": self.global_bloom.get_statistics(),
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

    def reset(self) -> None:
        """Reset all statistics."""
        self.wait_for_async_processing(timeout=5.0)

        with self.lock:
            self.global_bloom.clear()
            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.queue_full_blocks = 0
            self.queue_max_size = 0
            self._clear_async_queue()

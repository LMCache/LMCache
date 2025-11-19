# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.record_strategies.base import RecordStrategy
from lmcache.v1.utils.bloom_filter import BloomFilter

logger = init_logger(__name__)


class MemoryBloomFilterStrategy(RecordStrategy):
    """Memory-based strategy using Bloom Filter."""

    def __init__(self, config, chunk_size: int):
        super().__init__(
            chunk_size=chunk_size,
            async_enabled=config.chunk_statistics_async_enabled,
            # Maximum number of items in async processing queue
            async_queue_capacity=config.get_extra_config_value(
                "chunk_statistics_async_queue_capacity", 100000
            ),
            # Whether to preprocess chunks before adding to async queue
            async_preprocess_chunks=config.get_extra_config_value(
                "chunk_statistics_async_preprocess_chunks", False
            ),
        )
        self.global_bloom = BloomFilter(
            # Expected number of chunks for bloom filter capacity planning
            config.get_extra_config_value(
                "chunk_statistics_mem_bf_expected_chunks", 20000000
            ),
            # Target false positive rate for bloom filter
            config.get_extra_config_value(
                "chunk_statistics_mem_bf_false_positive_rate", 0.01
            ),
        )

    def _preprocess_for_async(self, token_ids: list[int]) -> list[list[int]]:
        return self._preprocess_chunks(token_ids)

    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        chunk_bloom_positions = self._preprocess_chunks(token_ids)
        self._process_chunk_data(chunk_bloom_positions)

    def _preprocess_chunks(self, token_ids: list[int]) -> list[list[int]]:
        """Preprocess token IDs into bloom filter hash positions.

        Args:
            token_ids: List of token IDs to process

        Returns:
            List of bloom filter hash positions for each chunk, where each inner list
            contains the hash_count separate hash results for that chunk.

            Example structure:
            [
                [hash1_pos1, hash1_pos2, ..., hash1_posN],  # chunk 1 hashes
                [hash2_pos1, hash2_pos2, ..., hash2_posN],  # chunk 2 hashes
                ...
            ]
            where N is the bloom filter's hash_count
        """
        chunk_data_list = []
        for prefix_hash in self._compute_chunk_hashes(token_ids):
            if prefix_hash < 0:
                prefix_hash = prefix_hash & ((1 << 64) - 1)
            chunk_data_list.append(self.global_bloom._hashes(prefix_hash))
        return chunk_data_list

    def _process_queue_item(self, item) -> None:
        data, lookup_id = item
        if self.async_preprocess_chunks:
            chunk_bloom_positions = data
        else:
            chunk_bloom_positions = self._preprocess_chunks(data)
        self._process_chunk_data(chunk_bloom_positions)

    def _process_chunk_data(self, chunk_bloom_positions: list[list[int]]) -> None:
        with self.lock:
            unique = self.global_bloom.add_batch_with_hashes_and_check(
                chunk_bloom_positions
            )
            self.total_chunks += len(chunk_bloom_positions)
            self.unique_chunks_count += unique

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats.update({"bloom_filter": self.global_bloom.get_statistics()})
        return stats

    def setup_metrics(self, prometheus_logger) -> None:
        """Setup bloom filter specific metrics."""
        super().setup_metrics(prometheus_logger)
        prometheus_logger.chunk_statistics_bloom_filter_size_mb.set_function(
            lambda: self.global_bloom.get_memory_usage_bytes() / (1024 * 1024)
        )
        prometheus_logger.chunk_statistics_bloom_filter_fill_rate.set_function(
            lambda: self.global_bloom.get_fill_rate()
        )

    def reset(self) -> None:
        self.wait_for_async_processing(timeout=5.0)
        with self.lock:
            self.global_bloom.clear()
            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.queue_full_blocks = 0
            self.queue_max_size = 0
            self._clear_async_queue()

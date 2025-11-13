# SPDX-License-Identifier: Apache-2.0

# Standard

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.record_strategies.base import RecordStrategy
from lmcache.v1.utils.bloom_filter import BloomFilter

logger = init_logger(__name__)


class MemoryBloomFilterStrategy(RecordStrategy):
    """Memory-based strategy using Bloom Filter."""

    @classmethod
    def name(cls) -> str:
        return "memory_bloom_filter"

    def __init__(self, config, chunk_size: int):
        super().__init__(
            chunk_size=chunk_size,
            async_enabled=config.chunk_statistics_async_enabled,
            async_queue_capacity=config.chunk_statistics_async_queue_capacity,
            async_preprocess_chunks=config.chunk_statistics_async_preprocess_chunks,
        )
        self.global_bloom = BloomFilter(
            config.chunk_statistics_expected_chunks,
            config.chunk_statistics_false_positive_rate,
        )

    def _preprocess_for_async(self, token_ids: list[int]) -> list[list[int]]:
        return self._preprocess_chunks(token_ids)

    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        self._process_statistics(token_ids, lookup_id)

    def _preprocess_chunks(self, token_ids: list[int]) -> list[list[int]]:
        chunk_data_list = []
        for prefix_hash in self._compute_chunk_hashes(token_ids):
            if prefix_hash < 0:
                prefix_hash = prefix_hash & ((1 << 64) - 1)
            chunk_data_list.append(self.global_bloom._hashes(prefix_hash))
        return chunk_data_list

    def _process_queue_item(self, item) -> None:
        if self.async_preprocess_chunks:
            self._process_chunk_data(item[0], item[1])
        else:
            self._process_statistics(item[0], item[1])

    def _update_bloom_filter(self, positions_list: list[list[int]]) -> int:
        unique_count = 0
        bit_array = self.global_bloom.bit_array
        for positions in positions_list:
            is_new = any(
                not (bit_array[pos >> 5] & (1 << (pos & 31))) for pos in positions
            )
            if is_new:
                for pos in positions:
                    bit_array[pos >> 5] |= 1 << (pos & 31)
                unique_count += 1
        return unique_count

    def _process_chunk_data(
        self, chunk_data_list: list[list[int]], lookup_id: str
    ) -> None:
        with self.lock:
            unique = self._update_bloom_filter(chunk_data_list)
            self.global_bloom.item_count += unique
            self.total_chunks += len(chunk_data_list)
            self.unique_chunks_count += unique

    def _process_statistics(self, token_ids: list[int], lookup_id: str) -> None:
        chunk_bloom_positions = self._preprocess_chunks(token_ids)
        with self.lock:
            unique = self._update_bloom_filter(chunk_bloom_positions)
            self.global_bloom.item_count += unique
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
            lambda: sum(bin(val).count("1") for val in self.global_bloom.bit_array)
            / self.global_bloom.size
            if self.global_bloom.size > 0
            else 0.0
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

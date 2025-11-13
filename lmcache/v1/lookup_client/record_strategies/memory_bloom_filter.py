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
        prefix_hash_bytes = b""
        for chunk_slice in self._iterate_chunks(token_ids):
            prefix_hash_bytes = self._compute_chunk_hash(prefix_hash_bytes, chunk_slice)
            prefix_hash = int.from_bytes(prefix_hash_bytes, "big", signed=False)
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

    def _get_strategy_specific_statistics(self) -> dict:
        return {"bloom_filter": self.global_bloom.get_statistics()}

    def reset(self) -> None:
        self.wait_for_async_processing(timeout=5.0)
        with self.lock:
            self.global_bloom.clear()
            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.queue_full_blocks = 0
            self.queue_max_size = 0
            self._clear_async_queue()

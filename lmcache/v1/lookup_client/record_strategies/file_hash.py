# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Optional, cast
import io
import json
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.record_strategies.base import RecordStrategy

logger = init_logger(__name__)


class FileHashStrategy(RecordStrategy):
    """File-based strategy that writes chunk hashes to disk."""

    def __init__(self, config, chunk_size: int):
        super().__init__(
            chunk_size=chunk_size,
            async_enabled=config.chunk_statistics_async_enabled,
            async_queue_capacity=config.chunk_statistics_async_queue_capacity,
            async_preprocess_chunks=config.chunk_statistics_async_preprocess_chunks,
        )
        self.file_rotation_size = config.chunk_statistics_file_rotation_size
        self.file_max_count = config.chunk_statistics_file_max_count
        self.output_dir = Path(config.chunk_statistics_file_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_count = 0
        self.current_file_size = 0
        self.current_file: Optional[Path] = None
        self.current_file_handle: Optional[io.TextIOWrapper] = None
        self.file_list: list[Path] = []

    def _preprocess_for_async(self, token_ids: list[int]) -> list[str]:
        return self._compute_chunk_hashes_hex(token_ids)

    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        self._write_data_to_file(self._compute_chunk_hashes_hex(token_ids), lookup_id)

    def _write_data_to_file(self, chunk_hashes: list[str], lookup_id: str) -> None:
        with self.lock:
            if (
                self.current_file is None
                or self.current_file_size >= self.file_rotation_size
            ):
                self._rotate_file()
            data = {
                "timestamp": time.time(),
                "lookup_id": lookup_id,
                "chunk_hashes": chunk_hashes,
            }
            if self.current_file_handle is not None:
                file_handle = cast(io.TextIOWrapper, self.current_file_handle)
                line = json.dumps(data) + "\n"
                file_handle.write(line)
                file_handle.flush()
                self.current_file_size += len(line)
            self.total_chunks += len(chunk_hashes)
            self.unique_chunks_count += len(set(chunk_hashes))

    def _rotate_file(self) -> None:
        if self.current_file_handle is not None:
            cast(io.TextIOWrapper, self.current_file_handle).close()
        timestamp = int(time.time())
        self.current_file = (
            self.output_dir / f"chunk_hashes_{timestamp}_{self.file_count:06d}.jsonl"
        )
        self.current_file_handle = open(self.current_file, "w")
        self.current_file_size = 0
        self.file_count += 1
        self.file_list.append(self.current_file)
        if len(self.file_list) > self.file_max_count:
            oldest_file = self.file_list.pop(0)
            try:
                if oldest_file.exists():
                    oldest_file.unlink()
            except Exception as e:
                logger.error("Failed to delete file %s: %s", oldest_file, e)

    def _process_queue_item(self, item) -> None:
        data, lookup_id = item
        if self.async_preprocess_chunks:
            chunk_hashes = data
        else:
            chunk_hashes = self._compute_chunk_hashes_hex(data)
        self._write_data_to_file(chunk_hashes, lookup_id)

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats.update(
            {
                "file_hash": {
                    "file_count": self.file_count,
                    "current_file_size": self.current_file_size,
                    "file_max_count": self.file_max_count,
                    "output_dir": str(self.output_dir),
                }
            }
        )
        return stats

    def setup_metrics(self, prometheus_logger) -> None:
        """Setup file hash specific metrics."""
        super().setup_metrics(prometheus_logger)
        prometheus_logger.chunk_statistics_file_count.set_function(
            lambda: self.file_count
        )
        prometheus_logger.chunk_statistics_current_file_size.set_function(
            lambda: self.current_file_size
        )
        prometheus_logger.chunk_statistics_file_max_count.set_function(
            lambda: self.file_max_count
        )

    def reset(self) -> None:
        self.wait_for_async_processing(timeout=5.0)
        with self.lock:
            if self.current_file_handle is not None:
                cast(io.TextIOWrapper, self.current_file_handle).close()
                self.current_file_handle = None
                self.current_file = None
            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.file_count = 0
            self.current_file_size = 0
            self.file_list.clear()
            self._clear_async_queue()

    def close(self) -> None:
        super().close()
        if self.current_file_handle is not None:
            cast(io.TextIOWrapper, self.current_file_handle).close()
            self.current_file_handle = None
            self.current_file = None

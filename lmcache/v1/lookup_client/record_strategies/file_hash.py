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
    """
    File-based recording strategy that writes chunk hashes to disk.

    This strategy computes chunk hashes and writes them to files for persistence
    and later analysis. Supports both synchronous and asynchronous modes.
    """

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this strategy."""
        return "file_hash"

    def __init__(self, config, chunk_size: int):
        """Initialize from configuration object."""
        super().__init__(
            chunk_size=chunk_size,
            async_enabled=config.chunk_statistics_async_enabled,
            async_queue_capacity=config.chunk_statistics_async_queue_capacity,
            async_preprocess_chunks=config.chunk_statistics_async_preprocess_chunks,
        )

        self.file_rotation_size = config.chunk_statistics_file_rotation_size
        self.file_max_count = config.chunk_statistics_file_max_count
        self.store_full_tokens = getattr(
            config, "chunk_statistics_store_full_tokens", False
        )

        # Setup output directory
        self.output_dir = Path(config.chunk_statistics_file_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File-specific statistics
        self.file_count = 0
        self.current_file_size = 0

        # Current file handle
        self.current_file: Optional[Path] = None
        self.current_file_handle: Optional[io.TextIOWrapper] = None
        self.file_list: list[Path] = []  # Track created files for rotation

        logger.info(
            "FileHashStrategy initialized with output_dir=%s, async_enabled=%s",
            self.output_dir,
            self.async_enabled,
        )

    def _record_async(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics asynchronously."""
        if self.async_preprocess_chunks:
            chunk_hashes = self._compute_chunk_hashes(token_ids)
            if self.store_full_tokens:
                self._queue_item((chunk_hashes, token_ids, lookup_id, True))
            else:
                self._queue_item((chunk_hashes, None, lookup_id, True))
        else:
            self._queue_item((token_ids, lookup_id, False))

    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics synchronously."""
        chunk_hashes = self._compute_chunk_hashes(token_ids)
        self._write_data_to_file(chunk_hashes, token_ids, lookup_id)

    def _write_data_to_file(
        self,
        chunk_hashes: list[str],
        token_ids: Optional[list[int]],
        lookup_id: str,
    ) -> None:
        """Write chunk hashes and optionally full token ids to file."""
        with self.lock:
            # Rotate file if needed
            if (
                self.current_file is None
                or self.current_file_size >= self.file_rotation_size
            ):
                self._rotate_file()

            # Write data
            data = {
                "timestamp": time.time(),
                "lookup_id": lookup_id,
                "chunk_hashes": chunk_hashes,
            }
            if self.store_full_tokens and token_ids is not None:
                data["token_ids"] = token_ids

            if self.current_file_handle is not None:
                file_handle = cast(io.TextIOWrapper, self.current_file_handle)
                file_handle.write(json.dumps(data) + "\n")
                file_handle.flush()
                self.current_file_size += len(json.dumps(data)) + 1

            # Update statistics
            self.total_chunks += len(chunk_hashes)
            self.unique_chunks_count += len(set(chunk_hashes))

    def _rotate_file(self) -> None:
        """Rotate to a new output file."""
        # Close current file if open
        if self.current_file_handle is not None:
            file_handle = cast(io.TextIOWrapper, self.current_file_handle)
            file_handle.close()

        # Create new file
        timestamp = int(time.time())
        self.current_file = (
            self.output_dir / f"chunk_hashes_{timestamp}_{self.file_count:06d}.jsonl"
        )
        self.current_file_handle = open(self.current_file, "w")
        self.current_file_size = 0
        self.file_count += 1
        self.file_list.append(self.current_file)

        # Remove oldest file if exceeding max count
        if len(self.file_list) > self.file_max_count:
            oldest_file = self.file_list.pop(0)
            try:
                if oldest_file.exists():
                    oldest_file.unlink()
                    logger.info("Deleted oldest file: %s", oldest_file)
            except Exception as e:
                logger.error("Failed to delete oldest file %s: %s", oldest_file, e)

        logger.info("Rotated to new file: %s", self.current_file)

    def _process_queue_item(self, item) -> None:
        """Process a single item from the queue."""
        if len(item) == 4:
            chunk_hashes, token_ids, lookup_id, is_preprocessed = item
        elif len(item) == 3:
            data, lookup_id, is_preprocessed = item
            if is_preprocessed:
                chunk_hashes = data
                token_ids = None
            else:
                token_ids = data
                chunk_hashes = self._compute_chunk_hashes(token_ids)
        else:
            token_ids, lookup_id = item
            chunk_hashes = self._compute_chunk_hashes(token_ids)

        self._write_data_to_file(chunk_hashes, token_ids, lookup_id)

    def _get_strategy_specific_statistics(self) -> dict:
        """Get strategy-specific statistics."""
        return {
            "file_hash": {
                "file_count": self.file_count,
                "current_file_size": self.current_file_size,
                "output_dir": str(self.output_dir),
                "file_max_count": self.file_max_count,
                "file_rotation_size": self.file_rotation_size,
            },
        }

    def reset(self) -> None:
        """Reset all statistics and close current file."""
        self.wait_for_async_processing(timeout=5.0)

        with self.lock:
            if self.current_file_handle is not None:
                file_handle = cast(io.TextIOWrapper, self.current_file_handle)
                file_handle.close()
                self.current_file_handle = None
                self.current_file = None

            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.file_count = 0
            self.current_file_size = 0
            self.file_list.clear()
            self._clear_async_queue()

    def close(self) -> None:
        """Clean up resources."""
        super().close()

        if self.current_file_handle is not None:
            file_handle = cast(io.TextIOWrapper, self.current_file_handle)
            file_handle.close()
            self.current_file_handle = None
            self.current_file = None

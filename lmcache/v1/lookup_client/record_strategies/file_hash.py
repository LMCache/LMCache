# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Optional, cast
import hashlib
import io
import json
import queue
import struct
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
        )

        self.file_rotation_size = config.chunk_statistics_file_rotation_size

        # Setup output directory
        self.output_dir = Path(config.chunk_statistics_file_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.total_chunks = 0
        self.unique_chunks_count = 0
        self.file_count = 0
        self.current_file_size = 0

        # Current file handle
        self.current_file: Optional[Path] = None
        self.current_file_handle: Optional[io.TextIOWrapper] = None

        logger.info(
            "FileHashStrategy initialized with output_dir=%s, async_enabled=%s",
            self.output_dir,
            self.async_enabled,
        )

    def _record_async(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics asynchronously."""
        self._queue_item((token_ids, lookup_id))

    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics synchronously."""
        chunk_hashes = self._compute_chunk_hashes(token_ids)
        self._write_hashes_to_file(chunk_hashes, lookup_id)

    def _compute_chunk_hashes(self, token_ids: list[int]) -> list[str]:
        """Compute SHA256 hashes for each chunk."""
        token_count = len(token_ids)
        num_chunks = (token_count + self.chunk_size - 1) // self.chunk_size
        chunk_hashes = []
        prefix_hash_bytes = b""

        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, token_count)
            chunk_slice = token_ids[start_idx:end_idx]

            # Compute hash for this chunk
            h = hashlib.sha256()
            h.update(prefix_hash_bytes)
            h.update(struct.pack(f">{len(chunk_slice)}i", *chunk_slice))

            digest = h.digest()
            prefix_hash_bytes = digest[:8]  # Chain for next chunk

            chunk_hash = digest.hex()
            chunk_hashes.append(chunk_hash)

        return chunk_hashes

    def _write_hashes_to_file(self, chunk_hashes: list[str], lookup_id: str) -> None:
        """Write chunk hashes to file."""
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

            if self.current_file_handle is not None:
                file_handle = cast(io.TextIOWrapper, self.current_file_handle)
                file_handle.write(json.dumps(data) + "\n")
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

        logger.info("Rotated to new file: %s", self.current_file)

    def _async_worker(self) -> None:
        """Background worker that processes statistics asynchronously."""
        logger.info("FileHashStrategy async worker started")

        if self.async_queue is None:
            return

        while not self.async_shutdown:
            try:
                # Get item from queue with timeout
                item = self.async_queue.get(timeout=0.1)

                # Check for sentinel value (shutdown signal)
                if item is None:
                    break

                token_ids, lookup_id = item
                chunk_hashes = self._compute_chunk_hashes(token_ids)
                self._write_hashes_to_file(chunk_hashes, lookup_id)

                self.async_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error in async file hash worker: %s", e, exc_info=True)

        # Process remaining items in queue before shutdown
        while not self.async_queue.empty():
            try:
                item = self.async_queue.get_nowait()
                if item is not None:
                    token_ids, lookup_id = item
                    chunk_hashes = self._compute_chunk_hashes(token_ids)
                    self._write_hashes_to_file(chunk_hashes, lookup_id)
                self.async_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error("Error processing remaining items: %s", e)

        logger.info("FileHashStrategy async worker stopped")

    def get_statistics(self) -> dict:
        """Get current statistics from this strategy."""
        with self.lock:
            return {
                "total_chunks": self.total_chunks,
                "unique_chunks": self.unique_chunks_count,
                "file_count": self.file_count,
                "current_file_size": self.current_file_size,
                "output_dir": str(self.output_dir),
                "async_enabled": self.async_enabled,
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
            self._clear_async_queue()

    def close(self) -> None:
        """Clean up resources."""
        super().close()

        if self.current_file_handle is not None:
            file_handle = cast(io.TextIOWrapper, self.current_file_handle)
            file_handle.close()
            self.current_file_handle = None
            self.current_file = None

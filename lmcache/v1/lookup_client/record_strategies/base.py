# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Optional, Union
import hashlib
import queue
import struct
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class RecordStrategy(ABC):
    """Abstract base class for chunk recording strategies."""

    def __init__(
        self,
        chunk_size: int,
        async_enabled: bool = False,
        async_queue_capacity: int = 10000,
        async_preprocess_chunks: bool = False,
    ):
        """
        Initialize base strategy with common parameters.

        Args:
            chunk_size: Size of each chunk
            async_enabled: Whether to enable async processing
            async_queue_capacity: Maximum size of async queue
            async_preprocess_chunks: Whether to preprocess chunks before queuing
        """
        self.chunk_size = chunk_size
        self.async_enabled = async_enabled
        self.async_queue_capacity = async_queue_capacity
        self.async_preprocess_chunks = async_preprocess_chunks

        # Async processing
        self.async_queue: Optional[queue.Queue] = None
        self.async_worker_thread: Optional[threading.Thread] = None
        self.async_shutdown = False
        self.queue_full_blocks = 0
        self.queue_max_size = 0

        # Common statistics
        self.total_chunks = 0
        self.unique_chunks_count = 0

        # Thread safety
        self.lock = threading.RLock()

        # Start async worker if enabled
        if self.async_enabled:
            self._start_async_worker()

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Return the name identifier for this strategy.

        Returns:
            String name that can be used in configuration
        """
        pass

    def _start_async_worker(self) -> None:
        """Start the async processing worker thread."""
        self.async_queue = queue.Queue(maxsize=self.async_queue_capacity)
        self.async_shutdown = False
        self.async_worker_thread = threading.Thread(
            target=self._async_worker,
            daemon=True,
            name=f"{self.__class__.__name__}Worker",
        )
        self.async_worker_thread.start()
        logger.info(
            "%s async worker started with queue capacity=%d",
            self.__class__.__name__,
            self.async_queue_capacity,
        )

    def _async_worker(self) -> None:
        """Background worker that processes items asynchronously."""
        logger.info("%s async worker started", self.__class__.__name__)

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
                logger.error(
                    "Error in %s async worker: %s",
                    self.__class__.__name__,
                    e,
                    exc_info=True,
                )

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

        logger.info("%s async worker stopped", self.__class__.__name__)

    def _queue_item(self, item, timeout: float = 10.0) -> None:
        """
        Queue an item for async processing.

        Args:
            item: Item to queue
            timeout: Timeout for blocking put
        """
        if self.async_queue is not None:
            try:
                self.async_queue.put(item, block=True, timeout=timeout)
            except queue.Full:
                with self.lock:
                    self.queue_full_blocks += 1
                logger.warning(
                    "Async queue full (capacity=%d), blocking until space available",
                    self.async_queue_capacity,
                )
                self.async_queue.put(item, block=True)

    def record(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
    ) -> None:
        """
        Record statistics for the given token_ids and lookup_id.

        Args:
            token_ids: Token IDs to process
            lookup_id: Unique identifier for the request
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if self.async_enabled:
            self._record_async(token_ids, lookup_id)
        else:
            self._record_sync(token_ids, lookup_id)

    def _compute_chunk_hash(
        self, prefix_hash_bytes: bytes, chunk_slice: list[int]
    ) -> bytes:
        """Compute hash for a single chunk."""
        h = hashlib.sha256()
        h.update(prefix_hash_bytes)
        h.update(struct.pack(f">{len(chunk_slice)}i", *chunk_slice))
        return h.digest()[:8]

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

            prefix_hash_bytes = self._compute_chunk_hash(prefix_hash_bytes, chunk_slice)
            chunk_hash = prefix_hash_bytes.hex()
            chunk_hashes.append(chunk_hash)

        return chunk_hashes

    @abstractmethod
    def _process_queue_item(self, item) -> None:
        """Process a single item from the queue."""
        pass

    @abstractmethod
    def _record_async(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics asynchronously."""
        pass

    @abstractmethod
    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        """Record statistics synchronously."""
        pass

    def get_statistics(self) -> dict:
        """
        Get current statistics from this strategy.

        Returns:
            Dictionary containing common and strategy-specific statistics
        """
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

            base_stats = {
                "total_chunks": self.total_chunks,
                "unique_chunks": self.unique_chunks_count,
                "duplicate_chunks": duplicate_chunks,
                "reuse_rate": reuse_rate,
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

            # Merge with strategy-specific statistics
            strategy_stats = self._get_strategy_specific_statistics()
            base_stats.update(strategy_stats)
            return base_stats

    def _get_strategy_specific_statistics(self) -> dict:
        """
        Get strategy-specific statistics.

        Subclasses should override this method to provide additional statistics.

        Returns:
            Dictionary containing strategy-specific statistics
        """
        return {}

    @abstractmethod
    def reset(self) -> None:
        """Reset all statistics and state."""
        pass

    def wait_for_async_processing(self, timeout: float = 5.0) -> bool:
        """
        Wait for async processing to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if processing is complete, False if timeout
        """
        if not self.async_enabled or self.async_queue is None:
            return True

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.async_queue.empty():
                time.sleep(0.01)
                if self.async_queue.empty():
                    return True
            time.sleep(0.01)

        return self.async_queue.empty()

    def _clear_async_queue(self) -> None:
        """Clear all items from async queue."""
        if self.async_queue is not None:
            while not self.async_queue.empty():
                try:
                    self.async_queue.get_nowait()
                except queue.Empty:
                    break

    def close(self) -> None:
        """Clean up resources."""
        if self.async_enabled and self.async_worker_thread is not None:
            self.async_shutdown = True
            if self.async_queue is not None:
                try:
                    self.async_queue.put(None, block=False)
                except queue.Full:
                    pass

            self.async_worker_thread.join(timeout=5.0)
            if self.async_worker_thread.is_alive():
                logger.warning("Async worker thread did not stop within timeout")
            else:
                logger.info("Async worker thread stopped")
            self.async_worker_thread = None
            self.async_queue = None

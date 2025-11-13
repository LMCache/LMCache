# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import queue
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.token_database import ChunkedTokenDatabase

logger = init_logger(__name__)


class RecordStrategy(ABC):
    """Base class for chunk recording strategies."""

    def __init__(
        self,
        chunk_size: int,
        async_enabled: bool = False,
        async_queue_capacity: int = 10000,
        async_preprocess_chunks: bool = False,
    ):
        self.chunk_size = chunk_size
        self.async_enabled = async_enabled
        self.async_queue_capacity = async_queue_capacity
        self.async_preprocess_chunks = async_preprocess_chunks

        self.async_queue: Optional[queue.Queue] = None
        self.async_worker_thread: Optional[threading.Thread] = None
        self.async_shutdown = False
        # Number of times the async queue was full and waited timeout
        self.queue_full_blocks = 0
        self.queue_max_size = 0
        self.total_chunks = 0
        self.unique_chunks_count = 0
        self.lock = threading.RLock()

        self._token_db = ChunkedTokenDatabase()
        self._token_db.chunk_size = chunk_size

        if self.async_enabled:
            self._start_async_worker()

    def _start_async_worker(self) -> None:
        self.async_queue = queue.Queue(maxsize=self.async_queue_capacity)
        self.async_shutdown = False
        self.async_worker_thread = threading.Thread(
            target=self._async_worker,
            daemon=True,
            name=f"{self.__class__.__name__}Worker",
        )
        self.async_worker_thread.start()

    def _async_worker(self) -> None:
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
                logger.error("Async worker error: %s", e, exc_info=True)
        while not self.async_queue.empty():
            try:
                item = self.async_queue.get_nowait()
                if item is not None:
                    self._process_queue_item(item)
                self.async_queue.task_done()
            except (queue.Empty, Exception):
                break

    def _queue_item(self, item, timeout: float = 10.0) -> None:
        if self.async_queue is not None:
            try:
                self.async_queue.put(item, block=True, timeout=timeout)
            except queue.Full:
                with self.lock:
                    self.queue_full_blocks += 1
                self.async_queue.put(item, block=True)

    def record(self, token_ids: Union[torch.Tensor, list[int]], lookup_id: str) -> None:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if self.async_enabled:
            self._record_async(token_ids, lookup_id)
        else:
            self._record_sync(token_ids, lookup_id)

    def _compute_chunk_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute prefix hashes for all chunks using ChunkedTokenDatabase.

        Returns:
            List of hash values (integers) for each chunk.
        """
        chunk_hashes = []
        for _, _, hash_val in self._token_db.process_tokens(
            tokens=token_ids, make_key=False
        ):
            chunk_hashes.append(hash_val)
        return chunk_hashes

    def _compute_chunk_hashes_hex(self, token_ids: list[int]) -> list[str]:
        """Compute prefix hashes for all chunks and return as hex strings.

        Returns:
            List of hash values (hex strings) for each chunk.
        """
        chunk_hashes = []
        for hash_val in self._compute_chunk_hashes(token_ids):
            if hash_val < 0:
                hash_val = hash_val & ((1 << 64) - 1)
            chunk_hashes.append(hex(hash_val))
        return chunk_hashes

    @abstractmethod
    def _preprocess_for_async(self, token_ids: list[int]) -> Any:
        pass

    def _record_async(self, token_ids: list[int], lookup_id: str) -> None:
        data = (
            self._preprocess_for_async(token_ids)
            if self.async_preprocess_chunks
            else token_ids
        )
        self._queue_item((data, lookup_id))

    @abstractmethod
    def _process_queue_item(self, item) -> None:
        pass

    @abstractmethod
    def _record_sync(self, token_ids: list[int], lookup_id: str) -> None:
        pass

    def get_statistics(self) -> dict:
        with self.lock:
            dup = self.total_chunks - self.unique_chunks_count
            queue_size = self.async_queue.qsize() if self.async_queue else 0
            self.queue_max_size = max(self.queue_max_size, queue_size)
            base_stats = {
                "total_chunks": self.total_chunks,
                "unique_chunks": self.unique_chunks_count,
                "duplicate_chunks": dup,
                "reuse_rate": dup / self.total_chunks if self.total_chunks > 0 else 0.0,
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
            return base_stats

    def setup_metrics(self, prometheus_logger) -> None:
        prometheus_logger.chunk_statistics_total_chunks.set_function(
            lambda: self.total_chunks
        )
        prometheus_logger.chunk_statistics_unique_chunks.set_function(
            lambda: self.unique_chunks_count
        )
        prometheus_logger.chunk_statistics_reuse_rate.set_function(
            lambda: (self.total_chunks - self.unique_chunks_count) / self.total_chunks
            if self.total_chunks > 0
            else 0.0
        )

    @abstractmethod
    def reset(self) -> None:
        pass

    def wait_for_async_processing(self, timeout: float = 5.0) -> bool:
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
        if self.async_queue is not None:
            while not self.async_queue.empty():
                try:
                    self.async_queue.get_nowait()
                except queue.Empty:
                    break

    def close(self) -> None:
        if self.async_enabled and self.async_worker_thread is not None:
            self.async_shutdown = True
            if self.async_queue is not None:
                try:
                    self.async_queue.put(None, block=False)
                except queue.Full:
                    pass
            self.async_worker_thread.join(timeout=5.0)
            if self.async_worker_thread.is_alive():
                logger.warning("Async worker did not stop")
            self.async_worker_thread = None
            self.async_queue = None

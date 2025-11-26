# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Optional
import asyncio
import threading

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    KVOpEvent,
    OpType,
)
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class BatchedMessageSender:
    """
    Batched message sender for KVOperation.

    This class accumulates KV admit/evict messages and sends them in batches
    to reduce communication overhead. Messages are flushed when either:
    1. The batch size threshold is reached (default: 50 messages)
    2. The timeout period expires (default: 0.01 seconds)

    Each message is assigned a unique, monotonically increasing sequence number
    to enable the receiver to detect missing or out-of-order messages.

    Design rationale:
    - Uses a SINGLE queue for both admit and evict messages to maintain strict
      order consistency. This is critical because operations like
      admit(key) -> evict(key) -> admit(key) must be processed in exact order
      to avoid race conditions and state inconsistencies on the receiver side.

    Thread-safe: Uses locks to protect internal queue and sequence counter.

    Args:
        metadata: Metadata for the worker
        config: Configuration for the worker
        location: Location of the worker
        lmcache_worker: The worker to send messages to. If None, batching is disabled.
    """

    def __init__(
        self,
        metadata: LMCacheEngineMetadata,
        config: LMCacheEngineConfig,
        location: str,
        lmcache_worker: "LMCacheWorker",
    ):
        self.batch_size = config.get_extra_config_value("kv_msg_batch_size", 50)
        self.batch_timeout = config.get_extra_config_value("kv_msg_batch_timeout", 0.01)
        self.lmcache_worker = lmcache_worker

        # Common fields shared by all operations in the batch
        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id
        self.location = location

        # Use a single queue to maintain order consistency between admit and evict
        # Store lightweight operations without redundant common fields
        self.message_queue: List[KVOpEvent] = []
        self.sequence_number = 0
        self.lock = threading.Lock()

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False

        self._start_background_thread()

        self._setup_metrics()

    def _setup_metrics(self):
        """Setup metrics for monitoring queue size."""
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.kv_msg_queue_size.set_function(
                lambda: len(self.message_queue)
            )

    def _start_background_thread(self):
        """Start background thread for periodic flushing."""
        self.running = True
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self.loop)
        self.flush_task = self.loop.create_task(self._periodic_flush())
        self.loop.run_forever()

    async def _periodic_flush(self):
        """Periodically flush messages based on timeout."""
        while self.running:
            await asyncio.sleep(self.batch_timeout)
            self._flush_if_needed(force=True)

    def _get_next_sequence_number(self) -> int:
        """Get next sequence number for message tracking."""
        seq = self.sequence_number
        self.sequence_number += 1
        return seq

    def add_kv_op(
        self,
        op_type: OpType,
        key: int,
    ):
        """Add a KV operation to the batch queue.

        Args:
            op_type: Operation type (ADMIT or EVICT)
            key: Chunk hash key
        """

        with self.lock:
            # Store only the lightweight operation
            seq_num = self._get_next_sequence_number()
            op = KVOpEvent(op_type=op_type, key=key, seq_num=seq_num)
            self.message_queue.append(op)
            self._flush_if_needed(force=False)

    def _flush_if_needed(self, force: bool = False):
        """Flush messages if batch size threshold is reached or force is True.

        NOTE: This method must be called with self.lock held.
        """

        should_flush = (force and len(self.message_queue) > 0) or (
            len(self.message_queue) >= self.batch_size
        )
        if should_flush:
            ops_to_send: List[KVOpEvent] = self.message_queue[:]
            self.message_queue.clear()

            # Ensure common fields are set (should be set by first message)
            assert self.instance_id is not None, "instance_id must be set"
            assert self.worker_id is not None, "worker_id must be set"
            assert self.location is not None, "location must be set"

            # Create batched message with common fields and lightweight operations
            # This reduces redundancy: common fields are sent once instead of N times
            batched_msg = BatchedKVOperationMsg(
                instance_id=self.instance_id,
                worker_id=self.worker_id,
                location=self.location,
                operations=ops_to_send,
            )
            self.lmcache_worker.put_msg(batched_msg)

    def flush(self):
        """Manually flush all pending messages."""
        with self.lock:
            self._flush_if_needed(force=True)

    def close(self):
        """Close the batched message sender and flush remaining messages."""
        self.running = False

        # Flush remaining messages
        self.flush()

        # Stop and close event loop
        if self.loop is not None:
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            # Wait for thread to finish
            if self.thread is not None and self.thread.is_alive():
                self.thread.join(timeout=1.0)
                if self.thread.is_alive():
                    logger.warning(
                        "Batched message sender thread did not terminate within timeout"
                    )
            # Close the loop after thread stops
            if not self.loop.is_closed():
                self.loop.close()

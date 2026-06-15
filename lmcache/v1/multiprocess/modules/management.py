# SPDX-License-Identifier: Apache-2.0
"""Management and utility operations for the MPCacheServer."""

# Standard
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocols.base import RequestType

logger = init_logger(__name__)


class ManagementModule:
    """Handles management and utility operations for the cache engine.

    Owns the lock used during cache clearing and provides handlers for
    ping, chunk-size queries, clear, debug, and block-allocation reporting.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._clear_lock = threading.Lock()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(RequestType.CLEAR, self.clear, ThreadPoolType.NORMAL),
            HandlerSpec(
                RequestType.GET_CHUNK_SIZE,
                self.get_chunk_size,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(RequestType.PING, self.ping, ThreadPoolType.NORMAL),
            HandlerSpec(RequestType.NOOP, self.debug, ThreadPoolType.SYNC),
            HandlerSpec(
                RequestType.REPORT_BLOCK_ALLOCATION,
                self.report_block_allocations,
                ThreadPoolType.NORMAL,
            ),
        ]

    def report_status(self) -> dict:
        """Return module-specific status information.

        Returns:
            An empty dict; management has no module-level metrics.
        """
        return {}

    def close(self) -> None:
        """Release resources owned by this module."""
        pass

    def ping(self) -> bool:
        """Respond to a ping request.

        Returns:
            Always True.
        """
        return True

    def get_chunk_size(self) -> int:
        """Return the chunk size used for KV cache operations.

        Returns:
            The chunk size.
        """
        return self._ctx.chunk_size

    def clear(self) -> None:
        """Clear all stored KV cache data from the storage manager."""
        with self._clear_lock:
            self._ctx.storage_manager.memcheck()
            self._ctx.storage_manager.clear(force=True)
            self._ctx.storage_manager.memcheck()

    def debug(self) -> str:
        """Return a simple health-check string.

        Returns:
            The literal string ``"OK"``.
        """
        return "OK"

    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus.

        Args:
            instance_id: The scheduler instance ID.
            model_name: The model name from the adapter.
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "records": records,
                },
            )
        )

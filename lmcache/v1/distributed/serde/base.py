# SPDX-License-Identifier: Apache-2.0
"""
Serde interfaces for the distributed (multiprocess) storage controllers.

Two layers:
1. **Sync interface** (Serializer / Deserializer): user implements these.
   Pure transform logic, no threading or eventfds.
2. **Async interface** (SerdeProcessor): controllers consume this.
   Provides non-blocking submit/query/eventfd pattern matching L2 adapters.

Use ``AsyncSerdeProcessor`` to wrap sync implementations into the async
interface automatically.
"""

# Standard
import abc

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_management import MemoryObj

SerdeTaskId = int


# ============================================================================
# Sync interface (user-facing)
# ============================================================================


class Serializer(abc.ABC):
    """Sync serializer — users implement this.

    Defines the pure transform from KV-shaped data to serialized bytes.
    """

    @abc.abstractmethod
    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Serialize src KV data into dst byte buffer (in-place).

        Args:
            src: Source MemoryObj containing KV-shaped data (read-locked).
            dst: Destination MemoryObj byte buffer (write-locked).
                 Must have capacity >= estimate_serialized_size().

        Returns:
            The actual number of bytes written to dst.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Estimate the byte size of serialized output for this layout.

        Called BEFORE serialization to allocate the temp buffer.
        Actual buffer = SERDE_BUFFER_FACTOR * this value.
        """
        raise NotImplementedError


class Deserializer(abc.ABC):
    """Sync deserializer — users implement this.

    Defines the pure transform from serialized bytes to KV-shaped data.
    """

    @abc.abstractmethod
    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        """Deserialize src byte buffer into dst KV-shaped MemoryObj (in-place).

        Args:
            src: Source MemoryObj containing serialized bytes.
            dst: Destination MemoryObj with KV-shaped layout (write-locked).
        """
        raise NotImplementedError


# ============================================================================
# Async interface (controller-facing)
# ============================================================================


class SerdeProcessor(abc.ABC):
    """Async serde processor with eventfd-based completion notification.

    Provides non-blocking serialize/deserialize with the same
    submit -> eventfd -> query pattern as L2 adapters.

    The serialize event fd is consumed by the store controller.
    The deserialize event fd is consumed by the prefetch controller.
    They must be distinct file descriptors.

    Users should NOT implement this directly. Instead, implement
    Serializer/Deserializer and wrap with AsyncSerdeProcessor.
    """

    # ----- Event fds -----

    @abc.abstractmethod
    def get_serialize_event_fd(self) -> int:
        """Event fd signaled when serialize tasks complete.

        Must be distinct from the deserialize event fd.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_deserialize_event_fd(self) -> int:
        """Event fd signaled when deserialize tasks complete.

        Must be distinct from the serialize event fd.
        """
        raise NotImplementedError

    # ----- Serialize (store controller: L1 KV -> temp bytes) -----

    @abc.abstractmethod
    def submit_serialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
    ) -> SerdeTaskId:
        """Submit a batch serialization task.

        Args:
            src_objs: Source KV-shaped MemoryObjs (read-locked).
            dst_objs: Destination byte-buffer MemoryObjs (write-locked).

        Returns:
            Task ID for querying completion.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query_serialize_result(self, task_id: SerdeTaskId) -> bool | None:
        """Query serialization task completion.

        Returns True on success, False on failure, None if still pending.
        Non-idempotent: only returns a non-None value once per task.
        """
        raise NotImplementedError

    # ----- Deserialize (prefetch controller: temp bytes -> L1 KV) -----

    @abc.abstractmethod
    def submit_deserialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
    ) -> SerdeTaskId:
        """Submit a batch deserialization task.

        Args:
            src_objs: Source byte-buffer MemoryObjs (filled by L2 load).
            dst_objs: Destination KV-shaped MemoryObjs (write-locked).

        Returns:
            Task ID for querying completion.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query_deserialize_result(self, task_id: SerdeTaskId) -> bool | None:
        """Query deserialization task completion.

        Returns True on success, False on failure, None if still pending.
        Non-idempotent: only returns a non-None value once per task.
        """
        raise NotImplementedError

    # ----- Size estimation -----

    @abc.abstractmethod
    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Estimate byte size of serialized output for this layout.

        Called BEFORE serialization to allocate temp buffers.
        """
        raise NotImplementedError

    # ----- Lifecycle -----

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources (event fds, threads, etc.)."""
        raise NotImplementedError

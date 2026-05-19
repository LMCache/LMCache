# SPDX-License-Identifier: Apache-2.0
"""
AsyncSerdeProcessor: wraps sync Serializer/Deserializer into the async
SerdeProcessor interface expected by the controllers.

Runs serialization/deserialization tasks in a thread pool and signals
event notifiers on completion, matching the L2 adapter async pattern.
The notifiers come from :mod:`lmcache.v1.platform` so the same code
runs on Linux (eventfd) and other POSIX platforms (pipe fallback).
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import enum
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.base import (
    Deserializer,
    SerdeProcessor,
    SerdeTaskId,
    Serializer,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    get_event_bus,
    is_observability_enabled,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


class _TaskType(enum.Enum):
    SERIALIZE = enum.auto()
    DESERIALIZE = enum.auto()


class AsyncSerdeProcessor(SerdeProcessor):
    """Wraps sync Serializer/Deserializer into async SerdeProcessor.

    Runs each submitted task in a thread pool. On completion, stores
    the result and signals the appropriate event notifier so the
    controller's poll loop wakes up.

    Args:
        serializer: Sync serializer implementation.
        deserializer: Sync deserializer implementation.
        max_workers: Thread pool size. Default 1 (serialization is
            typically CPU-bound, more threads may help if the transform
            releases the GIL).
    """

    def __init__(
        self,
        serializer: Serializer,
        deserializer: Deserializer,
        max_workers: int = 1,
        serde_type: str = "unknown",
    ) -> None:
        self._serializer = serializer
        self._deserializer = deserializer
        self._serde_type = serde_type

        self._serialize_efd = create_event_notifier()
        self._deserialize_efd = create_event_notifier()

        self._lock = threading.Lock()
        self._next_task_id: SerdeTaskId = 0
        # task_id -> (success: bool) for completed tasks, partitioned by type
        self._completed_serialize: dict[SerdeTaskId, bool] = {}
        self._completed_deserialize: dict[SerdeTaskId, bool] = {}

        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    # ----- Event fds -----

    def get_serialize_event_fd(self) -> int:
        """Return the fd signaled on serialize completion."""
        return self._serialize_efd.fileno()

    def get_deserialize_event_fd(self) -> int:
        """Return the fd signaled on deserialize completion."""
        return self._deserialize_efd.fileno()

    # ----- Serialize -----

    def submit_serialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
    ) -> SerdeTaskId:
        """Submit a batch serialize task to the thread pool."""
        task_id = self._alloc_task_id()
        logger.debug(
            "Serde: submitted serialize task %d (%d objects)",
            task_id,
            len(src_objs),
        )
        self._pool.submit(
            self._run_task,
            task_id,
            _TaskType.SERIALIZE,
            src_objs,
            dst_objs,
        )
        return task_id

    def query_serialize_result(self, task_id: SerdeTaskId) -> bool | None:
        """Pop and return the serialize task result, or None if pending."""
        with self._lock:
            return self._completed_serialize.pop(task_id, None)

    # ----- Deserialize -----

    def submit_deserialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
    ) -> SerdeTaskId:
        """Submit a batch deserialize task to the thread pool."""
        task_id = self._alloc_task_id()
        logger.debug(
            "Serde: submitted deserialize task %d (%d objects)",
            task_id,
            len(src_objs),
        )
        self._pool.submit(
            self._run_task,
            task_id,
            _TaskType.DESERIALIZE,
            src_objs,
            dst_objs,
        )
        return task_id

    def query_deserialize_result(self, task_id: SerdeTaskId) -> bool | None:
        """Pop and return the deserialize task result, or None if pending."""
        with self._lock:
            return self._completed_deserialize.pop(task_id, None)

    # ----- Size estimation (delegates to sync serializer) -----

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Delegate to the sync serializer's estimate (includes margin)."""
        return self._serializer.estimate_serialized_size(layout_desc)

    # ----- Lifecycle -----

    def close(self) -> None:
        """Shut down the thread pool and close event notifiers."""
        self._pool.shutdown(wait=True)
        self._serialize_efd.close()
        self._deserialize_efd.close()

    # ----- Internal -----

    def _alloc_task_id(self) -> SerdeTaskId:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        return task_id

    def _run_task(
        self,
        task_id: SerdeTaskId,
        task_type: _TaskType,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
    ) -> None:
        """Execute a serialize/deserialize task in the thread pool.

        On completion (success or failure), stores the result and
        signals the event notifier.
        """
        success = True
        bytes_in = self._sum_object_sizes(src_objs)
        bytes_out = 0
        event_session_id = self._event_session_id(task_type, task_id)
        self._publish_serde_event(
            task_type,
            is_start=True,
            session_id=event_session_id,
            num_objects=len(src_objs),
        )
        failure_reason = ""
        try:
            if task_type == _TaskType.SERIALIZE:
                for src, dst in zip(src_objs, dst_objs, strict=True):
                    bytes_out += int(self._serializer.serialize(src, dst) or 0)
            else:
                for src, dst in zip(src_objs, dst_objs, strict=True):
                    self._deserializer.deserialize(src, dst)
                bytes_out = self._sum_object_sizes(dst_objs)
        except Exception as exc:
            logger.exception(
                "Serde task %d (%s) failed",
                task_id,
                task_type.name,
            )
            success = False
            failure_reason = type(exc).__name__

        self._publish_serde_event(
            task_type,
            is_start=False,
            session_id=event_session_id,
            num_objects=len(src_objs),
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            success=success,
            failure_reason=failure_reason,
        )

        if success:
            logger.debug(
                "Serde: %s task %d completed successfully (%d objects)",
                task_type.name.lower(),
                task_id,
                len(src_objs),
            )
        else:
            logger.warning(
                "Serde: %s task %d failed (%d objects)",
                task_type.name.lower(),
                task_id,
                len(src_objs),
            )

        with self._lock:
            if task_type == _TaskType.SERIALIZE:
                self._completed_serialize[task_id] = success
            else:
                self._completed_deserialize[task_id] = success

        # Signal the appropriate notifier to wake the controller's poll loop
        notifier = (
            self._serialize_efd
            if task_type == _TaskType.SERIALIZE
            else self._deserialize_efd
        )
        try:
            notifier.notify()
        except OSError:
            logger.exception("Failed to signal notifier for serde task %d", task_id)

    @staticmethod
    def _sum_object_sizes(objects: list[MemoryObj]) -> int:
        """Return logical byte size for objects without failing the task."""
        total = 0
        for obj in objects:
            try:
                total += int(obj.get_size())
            except Exception:
                logger.debug("Serde: failed to read MemoryObj size", exc_info=True)
        return total

    def _event_session_id(
        self,
        task_type: _TaskType,
        task_id: SerdeTaskId,
    ) -> str:
        """Return a process-local unique session id for serde start/end events."""
        return f"serde-{id(self):x}-{task_type.name.lower()}-{task_id}"

    def _publish_serde_event(
        self,
        task_type: _TaskType,
        *,
        is_start: bool,
        session_id: str,
        num_objects: int,
        bytes_in: int = 0,
        bytes_out: int = 0,
        success: bool = True,
        failure_reason: str = "",
    ) -> None:
        """Publish serde start/end events when MP observability is enabled."""
        if not is_observability_enabled():
            return
        if task_type == _TaskType.SERIALIZE:
            event_type = (
                EventType.CB_SERDE_ENCODE_START
                if is_start
                else EventType.CB_SERDE_ENCODE_END
            )
        else:
            event_type = (
                EventType.CB_SERDE_DECODE_START
                if is_start
                else EventType.CB_SERDE_DECODE_END
            )
        metadata: dict[str, object] = {
            "serde_type": self._serde_type,
            "num_objects": num_objects,
        }
        if not is_start:
            metadata.update(
                {
                    "bytes_in": bytes_in,
                    "bytes_out": bytes_out,
                    "success": success,
                }
            )
            if not success and failure_reason:
                metadata["failure_reason"] = failure_reason
        get_event_bus().publish(
            Event(
                event_type=event_type,
                session_id=session_id,
                metadata=metadata,
            )
        )

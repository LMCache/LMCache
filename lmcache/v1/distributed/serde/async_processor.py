# SPDX-License-Identifier: Apache-2.0
"""
AsyncSerdeProcessor: wraps sync Serializer/Deserializer into the async
eventfd-based SerdeProcessor interface expected by the controllers.

Runs serialization/deserialization tasks in a thread pool and signals
eventfds on completion, matching the L2 adapter async pattern.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import enum
import os
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

logger = init_logger(__name__)


class _TaskType(enum.Enum):
    SERIALIZE = enum.auto()
    DESERIALIZE = enum.auto()


class AsyncSerdeProcessor(SerdeProcessor):
    """Wraps sync Serializer/Deserializer into async SerdeProcessor.

    Runs each submitted task in a thread pool. On completion, stores
    the result and signals the appropriate eventfd so the controller's
    poll loop wakes up.

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
    ) -> None:
        self._serializer = serializer
        self._deserializer = deserializer

        self._serialize_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._deserialize_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        self._lock = threading.Lock()
        self._next_task_id: SerdeTaskId = 0
        # task_id -> (success: bool) for completed tasks, partitioned by type
        self._completed_serialize: dict[SerdeTaskId, bool] = {}
        self._completed_deserialize: dict[SerdeTaskId, bool] = {}

        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    # ----- Event fds -----

    def get_serialize_event_fd(self) -> int:
        """Return the eventfd signaled on serialize completion."""
        return self._serialize_efd

    def get_deserialize_event_fd(self) -> int:
        """Return the eventfd signaled on deserialize completion."""
        return self._deserialize_efd

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
        """Shut down the thread pool and close event fds."""
        self._pool.shutdown(wait=True)
        os.close(self._serialize_efd)
        os.close(self._deserialize_efd)

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
        signals the eventfd.
        """
        success = True
        try:
            if task_type == _TaskType.SERIALIZE:
                for src, dst in zip(src_objs, dst_objs, strict=True):
                    self._serializer.serialize(src, dst)
            else:
                for src, dst in zip(src_objs, dst_objs, strict=True):
                    self._deserializer.deserialize(src, dst)
        except Exception:
            logger.exception(
                "Serde task %d (%s) failed",
                task_id,
                task_type.name,
            )
            success = False

        if success:
            logger.info(
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

        # Signal the appropriate eventfd to wake the controller's poll loop
        efd = (
            self._serialize_efd
            if task_type == _TaskType.SERIALIZE
            else self._deserialize_efd
        )
        try:
            os.eventfd_write(efd, 1)
        except OSError:
            logger.exception("Failed to signal eventfd for serde task %d", task_id)

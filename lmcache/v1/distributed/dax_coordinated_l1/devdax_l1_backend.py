# SPDX-License-Identifier: Apache-2.0
"""Adapt the DAX-Coordinated L1 lifecycle to the public L1 manager contract."""

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    DaxCoordinatedL1Config,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_client import (
    DaxCoordinatedL1Client,
    raw_result_to_l1_error,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_transfer_logging import (
    DevDaxTransferLoggingSubscriber,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_types import (
    DevDaxReservationResult,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1ObjectMeta,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus

L1OperationResult = tuple[L1Error, MemoryObj | None]


def _map_reservations(
    results: dict[ObjectKey, DevDaxReservationResult],
) -> dict[ObjectKey, L1OperationResult]:
    return {
        key: (raw_result_to_l1_error(item.result), item.memory_obj)
        for key, item in results.items()
    }


class DaxCoordinatedL1Backend:
    """Own DAX-Coordinated L1-specific lifecycle adaptation and event publication.

    The generic :class:`L1Manager` keeps the existing local lifecycle intact
    and delegates here only when the opt-in DAX-Coordinated L1 backend is active.
    Callers serialize mutating operations with the L1 manager lock.
    """

    def __init__(
        self,
        dax_coordinated_l1_config: DaxCoordinatedL1Config,
        memory_config: L1MemoryManagerConfig,
        listeners: list[L1ManagerListener],
        event_bus: EventBus,
    ) -> None:
        """Create the native DAX-Coordinated L1 client and bind integration callbacks.

        Args:
            dax_coordinated_l1_config: Opt-in DAX-Coordinated L1 lifecycle
                configuration.
            memory_config: Existing L1 Device-DAX path and alignment settings.
            listeners: Live listener list owned by the generic L1 manager.
            event_bus: Process event bus used for normal L1 lifecycle events.
        """
        self._client = DaxCoordinatedL1Client(dax_coordinated_l1_config, memory_config)
        self._listeners = listeners
        self._event_bus = event_bus
        self._writers: dict[ObjectKey, tuple[str, int]] = {}
        if dax_coordinated_l1_config.per_transfer_logging:
            event_bus.register_subscriber(DevDaxTransferLoggingSubscriber())

    @property
    def client(self) -> DaxCoordinatedL1Client:
        """Expose the lifecycle owner for registration, diagnostics and shutdown."""
        return self._client

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve shared reader activity and publish normal L1 events."""
        raw = self._client.reserve_read(keys, read_locks)
        output = _map_reservations(raw)
        successful_keys = self._successful_operation_keys(output)
        for listener in self._listeners:
            listener.on_l1_keys_reserved_read(successful_keys)
        self._publish(EventType.L1_READ_RESERVED, successful_keys)
        return output

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return shared payload views for existing local reservations."""
        return _map_reservations(self._client.unsafe_read(keys))

    def finish_read(
        self, keys: list[ObjectKey], read_locks: int
    ) -> dict[ObjectKey, L1Error]:
        """Release shared reader activity and publish normal L1 events."""
        raw = self._client.finish_read(keys, read_locks)
        output = {key: raw_result_to_l1_error(result) for key, result in raw.items()}
        successful_keys = self._successful_error_keys(output)
        for listener in self._listeners:
            listener.on_l1_keys_read_finished(successful_keys)
        self._publish(EventType.L1_READ_FINISHED, successful_keys)
        return output

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        *,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve shared allocations without using the local lifecycle map.

        Raises:
            ValueError: If the key and temporary-flag counts differ.
        """
        if len(keys) != len(is_temporary):
            raise ValueError("keys and is_temporary must have the same length")
        supported_keys = [
            key
            for key, temporary in zip(keys, is_temporary, strict=True)
            if not temporary
        ]
        raw = self._client.reserve_write(supported_keys, layout_desc)
        output: dict[ObjectKey, L1OperationResult] = {}
        for key, temporary in zip(keys, is_temporary, strict=True):
            if temporary:
                output[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue
            item = raw[key]
            output[key] = raw_result_to_l1_error(item.result), item.memory_obj
        successful_keys = self._successful_operation_keys(output)
        for key in successful_keys:
            memory_obj = output[key][1]
            assert memory_obj is not None
            self._writers[key] = (tag, memory_obj.get_size())
        for listener in self._listeners:
            listener.on_l1_keys_reserved_write(successful_keys)
        self._publish(EventType.L1_WRITE_RESERVED, successful_keys)
        return output

    def finish_write(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Commit completed payload DMA and publish normal L1 events."""
        owned = self._owned_write_keys(keys, tag)
        raw = self._client.finish_write(owned)
        output = {key: L1Error.KEY_NOT_EXIST for key in keys}
        output.update(
            {key: raw_result_to_l1_error(result) for key, (result, _) in raw.items()}
        )
        successful_keys = self._successful_error_keys(output)
        for key in successful_keys:
            del self._writers[key]
        for listener in self._listeners:
            listener.on_l1_keys_write_finished(successful_keys)
        self._publish(EventType.L1_WRITE_FINISHED, successful_keys, raw)
        return output

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], read_locks: int, tag: str = ""
    ) -> dict[ObjectKey, L1OperationResult]:
        """Commit shared writes and atomically publish reader activity."""
        owned = self._owned_write_keys(keys, tag)
        raw = self._client.finish_write_and_reserve_read(owned, read_locks)
        output: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.KEY_NOT_EXIST, None) for key in keys
        }
        output.update(_map_reservations(raw))
        successful_keys = self._successful_operation_keys(output)
        for key in successful_keys:
            del self._writers[key]
        for listener in self._listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._publish(
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
            successful_keys,
            raw,
        )
        return output

    def finish_write_and_delete(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Fail closed for unsupported write abort without publishing payload.

        This backend rejects L2 configuration and has no native write-abort
        operation. Keep a matching reservation locked rather than expose
        incomplete bytes through a publish-then-delete sequence.
        """
        owned = set(self._owned_write_keys(keys, tag))
        return {
            key: L1Error.KEY_NOT_WRITABLE if key in owned else L1Error.KEY_NOT_EXIST
            for key in keys
        }

    def get_staging_memory_usage(self) -> int:
        """Return payload bytes held by this MP's unfinished tagged writes."""
        return sum(size for _, size in self._writers.values())

    def delete(self, keys: list[ObjectKey], force: bool) -> dict[ObjectKey, L1Error]:
        """Invalidate owner-local objects while rejecting unsafe force delete."""
        if force:
            return {key: L1Error.KEY_NOT_WRITABLE for key in keys}
        raw = self._client.delete(keys)
        output = {
            key: raw_result_to_l1_error(result) for key, (result, _) in raw.items()
        }
        successful_keys = self._successful_error_keys(output)
        for listener in self._listeners:
            listener.on_l1_keys_deleted_by_manager(successful_keys)
        self._publish(EventType.L1_KEYS_EVICTED, successful_keys, raw)
        return output

    def clear(self, force: bool) -> None:
        """Reject global clear until an owner-routing control plane exists.

        Raises:
            RuntimeError: Always, because cross-owner clear is unsupported.
        """
        raise RuntimeError(
            "clear is unsupported for DAX-Coordinated L1; evict through owners"
        )

    def _publish(
        self,
        event_type: EventType,
        keys: list[ObjectKey],
        results: dict[ObjectKey, DevDaxReservationResult] | None = None,
    ) -> None:
        metadata: dict[str, object] = {"keys": keys}
        if results is not None:
            metadata["meta"] = [
                self._object_meta(results[key].memory_obj) for key in keys
            ]
        self._event_bus.publish(Event(event_type=event_type, metadata=metadata))

    def _owned_write_keys(self, keys: list[ObjectKey], tag: str) -> list[ObjectKey]:
        """Select writes belonging to this caller without changing ownership."""
        return [
            key for key in keys if key in self._writers and self._writers[key][0] == tag
        ]

    @staticmethod
    def _successful_operation_keys(
        output: dict[ObjectKey, L1OperationResult],
    ) -> list[ObjectKey]:
        return [key for key, (error, _) in output.items() if error == L1Error.SUCCESS]

    @staticmethod
    def _successful_error_keys(
        output: dict[ObjectKey, L1Error],
    ) -> list[ObjectKey]:
        return [key for key, error in output.items() if error == L1Error.SUCCESS]

    @staticmethod
    def _object_meta(memory_obj: MemoryObj | None) -> L1ObjectMeta:
        if memory_obj is None:
            raise ValueError(
                "successful DAX-Coordinated L1 operation has no memory object"
            )
        return L1ObjectMeta(
            size_bytes=memory_obj.get_size(),
            backend=L1BackendType.DEVDAX,
        )

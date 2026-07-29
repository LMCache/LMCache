# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
from _thread import LockType
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias
import abc
import pickle

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor
from lmcache.v1.multiprocess.transfer_plan import TransferPlan

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

PickleChunk: TypeAlias = torch.Tensor | list[torch.Tensor]
PicklePayload: TypeAlias = list[torch.Tensor] | list[list[PickleChunk]]


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Return a stable torch dtype name without module prefix."""
    return str(dtype).split(".")[-1]


def _make_shm_slot_descriptor(
    memory_obj: "MemoryObj", tensor: torch.Tensor
) -> ShmSlotDescriptor:
    """Describe one contiguous tensor within a shared-memory memory object.

    Args:
        memory_obj: The SHM-backed storage object containing ``tensor``.
        tensor: One contiguous tensor view owned by ``memory_obj``.

    Returns:
        A slot descriptor whose offset and length identify exactly ``tensor``.

    Raises:
        ValueError: If the tensor is not a contiguous in-bounds view of the
            memory object.
    """
    if not tensor.is_contiguous():
        raise ValueError("SHM slot tensor must be contiguous")
    tensor_offset = tensor.data_ptr() - memory_obj.data_ptr
    tensor_length = tensor.numel() * tensor.element_size()
    if tensor_offset < 0 or tensor_offset + tensor_length > memory_obj.shm_byte_length:
        raise ValueError("SHM slot tensor is outside its memory object")
    return ShmSlotDescriptor(
        offset=memory_obj.shm_offset + tensor_offset,
        length=tensor_length,
        shape=list(tensor.shape),
        dtype=_dtype_to_name(tensor.dtype),
    )


def _object_keys_bound_to_plan(
    context: EngineDrivenContextMetadata,
    transfer_plan: TransferPlan | None,
    resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
    key: IPCCacheServerKey,
) -> list[list[ObjectKey]]:
    """Bind resolved storage object keys to a shared logical transfer plan.

    Legacy dense contexts have no transfer metadata or plan, so their single
    object-key group is returned unchanged. Multi-group plans select the
    requested source chunks in the plan's object-group order.

    Args:
        context: Registered engine-driven context metadata.
        transfer_plan: Shared logical plan, or ``None`` for legacy mode.
        resolve_obj_keys: Resolver returning keys in original object-group and
            source-chunk order.
        key: Cache key whose storage objects are resolved.

    Returns:
        Object keys in plan object-group and plan chunk order.

    Raises:
        ValueError: If a multi-group transfer has no plan, plan ordering does
            not match registration, or a plan chunk cannot bind to an object.
    """
    object_keys_by_group = resolve_obj_keys(key)
    if not context.object_group_layout_descs:
        return object_keys_by_group
    if transfer_plan is None:
        raise ValueError("multi-group storage transfer requires a transfer plan")
    if len(object_keys_by_group) != len(transfer_plan.object_groups):
        raise ValueError("multi-group object-key count does not match transfer plan")

    planned_object_keys: list[list[ObjectKey]] = []
    for object_group_plan, object_keys in zip(
        transfer_plan.object_groups, object_keys_by_group, strict=True
    ):
        object_group_id = object_group_plan.object_group_id
        if object_group_id >= len(context.object_group_layout_descs):
            raise ValueError(
                f"transfer plan references unregistered object group {object_group_id}"
            )
        try:
            planned_object_keys.append(
                [
                    object_keys[chunk_index]
                    for chunk_index in object_group_plan.chunk_indices
                ]
            )
        except IndexError as exc:
            raise ValueError(
                f"object group {object_group_id} has fewer keys than its transfer plan"
            ) from exc
    return planned_object_keys


def create_transfer_strategy(
    storage_manager: "StorageManager",
    *,
    shm_name: str,
    pool_size: int,
    pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_lock: LockType,
    transfer_key_factory: Callable[
        [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
    ],
) -> "TransferStrategy":
    """Create the non-GPU transfer strategy for a registered context.

    Args:
        storage_manager: Storage manager used by the selected strategy.
        shm_name: Shared-memory pool name advertised to workers.
        pool_size: Shared-memory pool size in bytes.
        pending_writes: Map of pending SHM write reservations keyed by transfer key.
        pending_reads: Map of pending SHM read reservations keyed by transfer key.
        pending_lock: Lock guarding shared pending SHM reservation state.
        transfer_key_factory: Factory that builds the `(instance_id, key)` lookup key
            used in the pending SHM reservation maps.

    Returns:
        ``ShmTransferStrategy`` when SHM is configured with a non-empty pool name and
        positive pool size, otherwise ``PickleTransferStrategy``.
    """
    if shm_name and pool_size > 0:
        logger.info("Using shm non-GPU transfer strategy")
        return ShmTransferStrategy(
            storage_manager=storage_manager,
            pending_writes=pending_writes,
            pending_reads=pending_reads,
            pending_lock=pending_lock,
            transfer_key_factory=transfer_key_factory,
            fallback_strategy=PickleTransferStrategy(storage_manager),
        )

    logger.info("Using pickle non-GPU transfer strategy")
    return PickleTransferStrategy(storage_manager)


class TransferStrategy(abc.ABC):
    """Contract for non-GPU transport backends used by the server.

    Implementations encapsulate the transport-specific prepare/commit lifecycle for
    store and retrieve operations, allowing the server to use either pickle-based or
    shared-memory-based transfers behind a common interface.
    """

    @abc.abstractmethod
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.
            transfer_plan: Shared store plan that binds object-key order for
                multi-group registrations; ``None`` for legacy mode.

        Returns:
            Transport-specific store preparation response.
        """

    @abc.abstractmethod
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> bool:
        """Finalize a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            cpu_data: Serialized payload from the worker.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.
            transfer_plan: Shared store plan that binds object-key order for
                multi-group registrations; ``None`` for legacy mode.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def abort_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Abort an unfinished store and release transport-side resources."""

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.
            transfer_plan: Shared retrieve plan that binds object-key order
                for multi-group registrations; ``None`` for legacy mode.

        Returns:
            Transport-specific retrieve preparation response.
        """

    @abc.abstractmethod
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.

        Returns:
            ``True`` when retrieve finalization succeeds.
        """


class PickleTransferStrategy(TransferStrategy):
    """Pickle-based transport for non-GPU transfer requests.

    This is the default transport when SHM is unavailable, and it is also used as a
    fallback by the SHM strategy when the worker sends an inline serialized payload.
    ``prepare_store`` returns an empty context, while ``commit_store`` deserializes
    the pickle payload and writes the resulting tensors into reserved objects.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
    ) -> None:
        """Initialize pickle transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
        """
        self._storage_manager = storage_manager

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareStoreResponse:
        """Return empty store context for pickle mode.

        Pickle transport does not pre-allocate SHM slots during prepare.
        """
        return PrepareStoreResponse(context={})

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> bool:
        """Deserialize and write pickled chunks into reserved objects.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        obj_keys_by_group = _object_keys_bound_to_plan(
            context, transfer_plan, resolve_obj_keys, key
        )
        payload: PicklePayload = pickle.loads(cpu_data)
        if len(obj_keys_by_group) == 1 and not context.object_group_layout_descs:
            return self._commit_single_group_store(
                obj_keys_by_group[0], payload, context.layout_desc
            )

        chunks_by_group = self._validate_multi_group_store_payload(
            payload, obj_keys_by_group, context
        )
        reserved_objects: list[tuple[int, list[ObjectKey], dict[ObjectKey, Any]]] = []
        write_succeeded = False
        try:
            for object_group_id, obj_keys in enumerate(obj_keys_by_group):
                reserved_dict = self._storage_manager.reserve_write(
                    obj_keys,
                    context.object_group_layout_descs[object_group_id],
                    "new",
                )
                reserved_objects.append((object_group_id, obj_keys, reserved_dict))

            for object_group_id, obj_keys, reserved_dict in reserved_objects:
                group_chunks = chunks_by_group[object_group_id]
                for chunk_idx, obj_key in enumerate(obj_keys):
                    memory_obj = reserved_dict.get(obj_key)
                    if memory_obj is None:
                        continue
                    chunk_parts = group_chunks[chunk_idx]
                    for tensor_idx, chunk_part in enumerate(chunk_parts):
                        target = memory_obj.get_tensor(tensor_idx)
                        if (
                            target is None
                            or target.shape != chunk_part.shape
                            or target.dtype != chunk_part.dtype
                        ):
                            raise ValueError(
                                f"reserved object group {object_group_id}, "
                                f"chunk {chunk_idx}, tensor {tensor_idx} does not "
                                "match the registered layout"
                            )

            for object_group_id, obj_keys, reserved_dict in reserved_objects:
                group_chunks = chunks_by_group[object_group_id]
                for chunk_idx, obj_key in enumerate(obj_keys):
                    memory_obj = reserved_dict.get(obj_key)
                    if memory_obj is None:
                        continue
                    chunk_parts = group_chunks[chunk_idx]
                    for tensor_idx, chunk_part in enumerate(chunk_parts):
                        target = memory_obj.get_tensor(tensor_idx)
                        if target is None:
                            raise ValueError(
                                f"reserved object group {object_group_id}, "
                                f"chunk {chunk_idx}, tensor {tensor_idx} is unavailable"
                            )
                        target.copy_(chunk_part)
            write_succeeded = True
        finally:
            reserved_keys = [
                obj_key
                for _, _, reserved_dict in reserved_objects
                for obj_key in reserved_dict
            ]
            if reserved_keys:
                if write_succeeded:
                    self._storage_manager.finish_write(reserved_keys)
                else:
                    self._storage_manager.delete_l1_keys(reserved_keys, force=True)

        return True

    def abort_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Return success because pickle stores allocate only during commit."""
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        obj_keys_by_group = _object_keys_bound_to_plan(
            context, transfer_plan, resolve_obj_keys, key
        )
        if len(obj_keys_by_group) == 1 and not context.object_group_layout_descs:
            return self._prepare_single_group_retrieve(obj_keys_by_group[0])

        prefetched_keys: list[ObjectKey] = []
        try:
            chunks_by_group: list[list[list[torch.Tensor]]] = []
            for object_group_id, obj_keys in enumerate(obj_keys_by_group):
                read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
                with read_ctx as maybe_memory_objs:
                    if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                        return PrepareRetrieveResponse(
                            success=False, data=b"", context={}
                        )
                    prefetched_keys.extend(obj_keys)
                    group_chunks: list[list[torch.Tensor]] = []
                    for memory_obj in maybe_memory_objs:
                        chunk_parts: list[torch.Tensor] = []
                        for tensor_idx in range(
                            len(
                                context.object_group_layout_descs[
                                    object_group_id
                                ].shapes
                            )
                        ):
                            tensor = memory_obj.get_tensor(tensor_idx)
                            if tensor is None:
                                return PrepareRetrieveResponse(
                                    success=False, data=b"", context={}
                                )
                            chunk_parts.append(tensor.cpu().clone())
                        group_chunks.append(chunk_parts)
                    chunks_by_group.append(group_chunks)
            return PrepareRetrieveResponse(
                success=True, data=pickle.dumps(chunks_by_group), context={}
            )
        finally:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """No-op for pickle mode; data was already copied during prepare."""
        return True

    def _commit_single_group_store(
        self,
        obj_keys: list[ObjectKey],
        payload: PicklePayload,
        layout_desc: "MemoryLayoutDesc",
    ) -> bool:
        """Store the legacy flat pickle payload without changing its wire shape."""
        if not all(isinstance(chunk, torch.Tensor) for chunk in payload):
            return False
        chunks = payload
        if len(chunks) != len(obj_keys):
            return False
        reserved_dict = self._storage_manager.reserve_write(
            obj_keys, layout_desc, "new"
        )
        written_keys: list[ObjectKey] = []
        try:
            for obj_key, chunk_cpu in zip(obj_keys, chunks, strict=True):
                memory_obj = reserved_dict.get(obj_key)
                if memory_obj is None:
                    continue
                target = memory_obj.tensor
                if target is None or chunk_cpu.shape != target.shape:
                    return False
                target.copy_(chunk_cpu)
                written_keys.append(obj_key)
        finally:
            if written_keys:
                self._storage_manager.finish_write(written_keys)
        return len(written_keys) == len(reserved_dict)

    def _prepare_single_group_retrieve(
        self, obj_keys: list[ObjectKey]
    ) -> PrepareRetrieveResponse:
        """Read and serialize the unchanged legacy flat pickle payload."""
        prefetched_keys: list[ObjectKey] = []
        try:
            read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
            with read_ctx as maybe_memory_objs:
                if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
                prefetched_keys = obj_keys[: len(maybe_memory_objs)]
                chunks: list[torch.Tensor] = []
                for memory_obj in maybe_memory_objs:
                    if memory_obj.tensor is None:
                        return PrepareRetrieveResponse(
                            success=False, data=b"", context={}
                        )
                    chunks.append(memory_obj.tensor.cpu().clone())
                return PrepareRetrieveResponse(
                    success=True, data=pickle.dumps(chunks), context={}
                )
        finally:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)

    @staticmethod
    def _validate_multi_group_store_payload(
        payload: PicklePayload,
        obj_keys_by_group: list[list[ObjectKey]],
        context: EngineDrivenContextMetadata,
    ) -> list[list[list[torch.Tensor]]]:
        """Validate a complete multi-group payload before reserving any object."""
        if (
            not isinstance(payload, list)
            or len(payload) != len(obj_keys_by_group)
            or len(context.object_group_layout_descs) != len(obj_keys_by_group)
        ):
            raise ValueError(
                "multi-group pickle payload has unexpected object-group count"
            )
        chunks_by_group: list[list[list[torch.Tensor]]] = []
        for object_group_id, (group_payload, obj_keys, layout_desc) in enumerate(
            zip(
                payload,
                obj_keys_by_group,
                context.object_group_layout_descs,
                strict=True,
            )
        ):
            if not isinstance(group_payload, list) or len(group_payload) != len(
                obj_keys
            ):
                raise ValueError(
                    f"object group {object_group_id} has an incomplete pickle payload"
                )
            group_chunks: list[list[torch.Tensor]] = []
            for chunk_idx, chunk_parts in enumerate(group_payload):
                if not isinstance(chunk_parts, list) or len(chunk_parts) != len(
                    layout_desc.shapes
                ):
                    raise ValueError(
                        f"object group {object_group_id}, chunk {chunk_idx} "
                        "has an invalid kernel-group payload"
                    )
                for tensor_idx, (chunk_part, shape, dtype) in enumerate(
                    zip(
                        chunk_parts,
                        layout_desc.shapes,
                        layout_desc.dtypes,
                        strict=True,
                    )
                ):
                    if (
                        not isinstance(chunk_part, torch.Tensor)
                        or chunk_part.shape != shape
                        or chunk_part.dtype != dtype
                    ):
                        raise ValueError(
                            f"object group {object_group_id}, chunk {chunk_idx}, "
                            f"tensor {tensor_idx} does not match registered layout"
                        )
                group_chunks.append(chunk_parts)
            chunks_by_group.append(group_chunks)
        return chunks_by_group


class ShmTransferStrategy(TransferStrategy):
    """Shared-memory transport for non-GPU transfer requests.

    This strategy exposes SHM slot descriptors during ``prepare_store`` and
    ``prepare_retrieve`` so workers can access storage buffers directly. It tracks
    pending SHM reservations until the matching commit step releases them, and it
    falls back to pickle-based commit handling when ``cpu_data`` is non-empty.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_lock: LockType,
        transfer_key_factory: Callable[
            [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
        ],
        fallback_strategy: PickleTransferStrategy,
    ) -> None:
        """Initialize SHM transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
            pending_writes: Shared pending SHM write reservations map.
            pending_reads: Shared pending SHM read reservations map.
            pending_lock: Lock guarding shared pending SHM maps.
            transfer_key_factory: Factory to build `(instance_id, key)` transfer keys.
            fallback_strategy: Pickle fallback for non-empty ``cpu_data`` payloads.
        """
        self._storage_manager = storage_manager
        self._pending_writes = pending_writes
        self._pending_reads = pending_reads
        self._pending_lock = pending_lock
        self._transfer_key_factory = transfer_key_factory
        self._fallback_strategy = fallback_strategy

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareStoreResponse:
        """Reserve SHM-backed objects and return slot descriptors.

        Returns:
            Legacy registrations receive a flat ``slots`` / ``chunk_indices``
            context. Multi-object-group registrations receive an
            ``object_groups`` context ordered by object group, chunk, and
            kernel-group tensor.
        """
        if context.object_group_layout_descs:
            return self._prepare_multi_group_store(
                key, instance_id, context, resolve_obj_keys, transfer_plan
            )

        self.abort_store(key, instance_id)
        obj_keys = resolve_obj_keys(key)[0]
        reserved = self._storage_manager.reserve_write(
            obj_keys, context.layout_desc, "new"
        )
        slots: list[dict[str, Any]] = []
        chunk_indices: list[int] = []
        reserved_keys: list[ObjectKey] = []
        try:
            for idx, obj_key in enumerate(obj_keys):
                memory_obj = reserved.get(obj_key)
                if memory_obj is None or memory_obj.tensor is None:
                    continue
                slots.append(
                    ShmSlotDescriptor(
                        offset=memory_obj.shm_offset,
                        length=memory_obj.shm_byte_length,
                        shape=list(memory_obj.tensor.shape),
                        dtype=_dtype_to_name(memory_obj.tensor.dtype),
                    ).to_dict()
                )
                chunk_indices.append(idx)
                reserved_keys.append(obj_key)
        finally:
            reserved_keys_set = set(reserved_keys)
            unused_keys = [
                obj_key for obj_key in reserved if obj_key not in reserved_keys_set
            ]
            if unused_keys:
                self._abort_write_reservations(unused_keys)
        transfer_key = self._transfer_key_factory(key, instance_id)
        if not reserved_keys:
            self._replace_pending_write(transfer_key, [])
            return PrepareStoreResponse(context={"slots": [], "chunk_indices": []})
        self._replace_pending_write(transfer_key, reserved_keys)
        return PrepareStoreResponse(
            context={"slots": slots, "chunk_indices": chunk_indices}
        )

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> bool:
        """Finalize SHM store write locks or fallback to pickle commit.

        Returns:
            ``True`` when pending SHM reservation is committed successfully.
        """
        transfer_key = self._transfer_key_factory(key, instance_id)
        if cpu_data != b"":
            # A worker that prepared direct SHM slots must never commit an
            # inline payload over those reservations. Drop the uncommitted
            # objects before attempting the compatibility fallback so malformed
            # payloads cannot leave a lock or a partially initialized cache key.
            pending_keys = self._take_pending_write(transfer_key)
            if pending_keys:
                self._abort_write_reservations(pending_keys)
            try:
                return self._fallback_strategy.commit_store(
                    key=key,
                    instance_id=instance_id,
                    cpu_data=cpu_data,
                    context=context,
                    resolve_obj_keys=resolve_obj_keys,
                    transfer_plan=transfer_plan,
                )
            except Exception:
                logger.exception("Malformed inline SHM store commit")
                return False

        reserved_keys = self._take_pending_write(transfer_key)
        if reserved_keys is None:
            return False
        if reserved_keys:
            self._storage_manager.finish_write(reserved_keys)
        return True

    def abort_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Discard pending SHM writes without publishing partial contents."""
        transfer_key = self._transfer_key_factory(key, instance_id)
        pending_keys = self._take_pending_write(transfer_key)
        if pending_keys:
            self._abort_write_reservations(pending_keys)
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareRetrieveResponse:
        """Read SHM objects and return slot descriptors for worker access."""
        if context.object_group_layout_descs:
            return self._prepare_multi_group_retrieve(
                key, instance_id, context, resolve_obj_keys, transfer_plan
            )

        obj_keys = resolve_obj_keys(key)[0]
        shm_prefetched_keys, shm_memory_objs = self._storage_manager.unsafe_read(
            obj_keys
        )
        if (
            not shm_memory_objs
            or len(shm_prefetched_keys) != len(obj_keys)
            or len(shm_memory_objs) != len(obj_keys)
        ):
            if shm_prefetched_keys:
                self._storage_manager.finish_read_prefetched(shm_prefetched_keys)
            return PrepareRetrieveResponse(success=False, data=b"", context={})
        slots: list[dict[str, Any]] = []
        for memory_obj in shm_memory_objs:
            if memory_obj.tensor is None:
                self._storage_manager.finish_read_prefetched(shm_prefetched_keys)
                return PrepareRetrieveResponse(success=False, data=b"", context={})
            slots.append(
                ShmSlotDescriptor(
                    offset=memory_obj.shm_offset,
                    length=memory_obj.shm_byte_length,
                    shape=list(memory_obj.tensor.shape),
                    dtype=_dtype_to_name(memory_obj.tensor.dtype),
                ).to_dict()
            )
        transfer_key = self._transfer_key_factory(key, instance_id)
        self._replace_pending_read(transfer_key, shm_prefetched_keys)
        return PrepareRetrieveResponse(success=True, data=b"", context={"slots": slots})

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Release pending SHM read locks for the completed retrieve request."""
        transfer_key = self._transfer_key_factory(key, instance_id)
        prefetched_keys = self._take_pending_read(transfer_key)
        if prefetched_keys:
            self._storage_manager.finish_read_prefetched(prefetched_keys)
        return True

    def _prepare_multi_group_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareStoreResponse:
        """Reserve all multi-group store slots before publishing any of them."""
        self.abort_store(key, instance_id)
        if transfer_plan is None:
            raise ValueError("multi-group SHM store has no transfer plan")
        obj_keys_by_group = _object_keys_bound_to_plan(
            context, transfer_plan, resolve_obj_keys, key
        )
        layouts = context.object_group_layout_descs
        if len(obj_keys_by_group) != len(layouts):
            raise ValueError(
                "multi-group SHM object-key count does not match registration"
            )

        reserved_keys: list[ObjectKey] = []
        object_groups: list[dict[str, Any]] = []
        try:
            for object_group_plan, obj_keys, layout in zip(
                transfer_plan.object_groups,
                obj_keys_by_group,
                layouts,
                strict=True,
            ):
                object_group_id = object_group_plan.object_group_id
                reserved = self._storage_manager.reserve_write(obj_keys, layout, "new")
                unexpected_keys = [
                    obj_key for obj_key in reserved if obj_key not in obj_keys
                ]
                if unexpected_keys:
                    reserved_keys.extend(reserved)
                    raise ValueError(
                        f"SHM reserve returned unknown object keys for group "
                        f"{object_group_id}"
                    )

                group_slots: list[list[dict[str, Any]]] = []
                chunk_indices: list[int] = []
                for chunk_idx, obj_key in zip(
                    object_group_plan.chunk_indices, obj_keys, strict=True
                ):
                    memory_obj = reserved.get(obj_key)
                    if memory_obj is None:
                        continue
                    # Add the key before validating the object so every
                    # reservation is aborted if one descriptor is malformed.
                    reserved_keys.append(obj_key)
                    slot_chunk: list[dict[str, Any]] = []
                    for tensor_idx, (shape, dtype) in enumerate(
                        zip(layout.shapes, layout.dtypes, strict=True)
                    ):
                        tensor = memory_obj.get_tensor(tensor_idx)
                        if (
                            tensor is None
                            or tensor.shape != shape
                            or tensor.dtype != dtype
                        ):
                            raise ValueError(
                                f"SHM object group {object_group_id}, chunk "
                                f"{chunk_idx}, tensor {tensor_idx} does not match "
                                "the registered layout"
                            )
                        slot_chunk.append(
                            _make_shm_slot_descriptor(memory_obj, tensor).to_dict()
                        )
                    chunk_indices.append(chunk_idx)
                    group_slots.append(slot_chunk)
                object_groups.append(
                    {
                        "object_group_id": object_group_id,
                        "chunk_indices": chunk_indices,
                        "slots": group_slots,
                    }
                )
        except Exception:
            if reserved_keys:
                self._abort_write_reservations(reserved_keys)
            raise

        transfer_key = self._transfer_key_factory(key, instance_id)
        self._replace_pending_write(transfer_key, reserved_keys)
        return PrepareStoreResponse(context={"object_groups": object_groups})

    def _prepare_multi_group_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[list[ObjectKey]]],
        transfer_plan: TransferPlan | None,
    ) -> PrepareRetrieveResponse:
        """Build an all-or-nothing ordered multi-group SHM read response."""
        obj_keys_by_group = _object_keys_bound_to_plan(
            context, transfer_plan, resolve_obj_keys, key
        )
        layouts = context.object_group_layout_descs
        if len(obj_keys_by_group) != len(layouts):
            raise ValueError(
                "multi-group SHM object-key count does not match registration"
            )
        if transfer_plan is None:
            raise ValueError("multi-group SHM retrieve has no transfer plan")

        prefetched_keys: list[ObjectKey] = []
        object_groups: list[dict[str, Any]] = []
        try:
            for object_group_plan, obj_keys, layout in zip(
                transfer_plan.object_groups,
                obj_keys_by_group,
                layouts,
                strict=True,
            ):
                object_group_id = object_group_plan.object_group_id
                group_prefetched_keys, memory_objs = self._storage_manager.unsafe_read(
                    obj_keys
                )
                if (
                    not memory_objs
                    or len(group_prefetched_keys) != len(obj_keys)
                    or len(memory_objs) != len(obj_keys)
                ):
                    if group_prefetched_keys:
                        self._storage_manager.finish_read_prefetched(
                            group_prefetched_keys
                        )
                    raise ValueError(
                        f"SHM object group {object_group_id} is not fully readable"
                    )
                # Record locks before validating descriptors so a malformed
                # memory object follows the same all-or-nothing cleanup path.
                prefetched_keys.extend(group_prefetched_keys)

                group_slots: list[list[dict[str, Any]]] = []
                for chunk_idx, memory_obj in zip(
                    object_group_plan.chunk_indices, memory_objs, strict=True
                ):
                    slot_chunk: list[dict[str, Any]] = []
                    for tensor_idx, (shape, dtype) in enumerate(
                        zip(layout.shapes, layout.dtypes, strict=True)
                    ):
                        tensor = memory_obj.get_tensor(tensor_idx)
                        if (
                            tensor is None
                            or tensor.shape != shape
                            or tensor.dtype != dtype
                        ):
                            raise ValueError(
                                f"SHM object group {object_group_id}, chunk "
                                f"{chunk_idx}, tensor {tensor_idx} "
                                "does not match the registered layout"
                            )
                        slot_chunk.append(
                            _make_shm_slot_descriptor(memory_obj, tensor).to_dict()
                        )
                    group_slots.append(slot_chunk)
                object_groups.append(
                    {
                        "object_group_id": object_group_id,
                        "chunk_indices": list(object_group_plan.chunk_indices),
                        "slots": group_slots,
                    }
                )
        except ValueError as exc:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)
            logger.debug("Multi-group SHM retrieve unavailable: %s", exc)
            return PrepareRetrieveResponse(success=False, data=b"", context={})
        except Exception:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)
            logger.exception("Failed to prepare multi-group SHM retrieve")
            return PrepareRetrieveResponse(success=False, data=b"", context={})

        transfer_key = self._transfer_key_factory(key, instance_id)
        self._replace_pending_read(transfer_key, prefetched_keys)
        return PrepareRetrieveResponse(
            success=True, data=b"", context={"object_groups": object_groups}
        )

    def _abort_write_reservations(self, object_keys: list[ObjectKey]) -> None:
        """Discard uncommitted SHM objects without publishing their contents."""
        if object_keys:
            self._storage_manager.delete_l1_keys(object_keys, force=True)

    def _replace_pending_write(
        self,
        transfer_key: tuple[int, IPCCacheServerKey],
        object_keys: list[ObjectKey],
    ) -> None:
        """Atomically replace a pending write and discard its stale reservation."""
        with self._pending_lock:
            stale_keys = self._pending_writes.pop(transfer_key, [])
            if object_keys:
                self._pending_writes[transfer_key] = object_keys
        if stale_keys:
            self._abort_write_reservations(stale_keys)

    def _replace_pending_read(
        self,
        transfer_key: tuple[int, IPCCacheServerKey],
        object_keys: list[ObjectKey],
    ) -> None:
        """Atomically replace a pending read and release its stale read locks."""
        with self._pending_lock:
            stale_keys = self._pending_reads.pop(transfer_key, [])
            if object_keys:
                self._pending_reads[transfer_key] = object_keys
        if stale_keys:
            self._storage_manager.finish_read_prefetched(stale_keys)

    def _take_pending_write(
        self, transfer_key: tuple[int, IPCCacheServerKey]
    ) -> list[ObjectKey] | None:
        """Remove and return one pending write reservation."""
        with self._pending_lock:
            return self._pending_writes.pop(transfer_key, None)

    def _take_pending_read(
        self, transfer_key: tuple[int, IPCCacheServerKey]
    ) -> list[ObjectKey]:
        """Remove and return one pending read reservation."""
        with self._pending_lock:
            return self._pending_reads.pop(transfer_key, [])

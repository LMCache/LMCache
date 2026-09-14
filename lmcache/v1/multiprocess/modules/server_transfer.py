# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
from _thread import LockType
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
import abc
import pickle

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Return a stable torch dtype name without module prefix."""
    return str(dtype).split(".")[-1]


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
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

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
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Finalize a store request from its serialized wire payload.

        Callers that already hold decoded chunks should use
        :meth:`commit_store_chunks` instead.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            cpu_data: Serialized payload from the worker.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def commit_store_chunks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        chunks: list[torch.Tensor],
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Finalize a store request from already-decoded chunks.

        Implementations must treat an empty ``chunks`` list the same way
        ``commit_store`` treats ``cpu_data=b""``, so the two entry points
        stay interchangeable.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            chunks: Decoded CPU chunk tensors for this call, positionally
                aligned with ``resolve_obj_keys(key)``.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            Transport-specific retrieve preparation response.
        """

    @abc.abstractmethod
    def prepare_retrieve_chunks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> tuple[PrepareRetrieveResponse, list[torch.Tensor]]:
        """Prepare a retrieve, returning chunks alongside the response.

        Lets a multi-group caller concatenate every group's chunks and
        serialize the result once instead of unpickling each group's payload
        only to re-pickle the concatenation.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            ``(response, chunks)``. ``response.data`` is always ``b""`` --
            the payload is carried by ``chunks`` instead, which is empty for
            transports that hand back SHM slots rather than tensors.
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
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Deserialize and write pickled chunks into reserved objects.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        return self.commit_store_chunks(
            key=key,
            instance_id=instance_id,
            chunks=pickle.loads(cpu_data),
            context=context,
            resolve_obj_keys=resolve_obj_keys,
        )

    def commit_store_chunks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        chunks: list[torch.Tensor],
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Write already-decoded chunks into reserved objects.

        Also serves as ``ShmTransferStrategy``'s fallback when ``chunks`` is
        non-empty, meaning that group's store went through pickle mode even
        though SHM is the registered strategy.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        obj_keys = resolve_obj_keys(key)
        reserved_dict = self._storage_manager.reserve_write(
            obj_keys, context.layout_desc, "new"
        )
        written_keys: list[ObjectKey] = []
        try:
            for idx, obj_key in enumerate(obj_keys):
                if obj_key not in reserved_dict:
                    continue
                if idx >= len(chunks):
                    logger.error(
                        "Engine-driven pickle store is missing chunk %d "
                        "(instance_id=%d, object_keys=%d, chunks=%d)",
                        idx,
                        instance_id,
                        len(obj_keys),
                        len(chunks),
                    )
                    continue
                memory_obj = reserved_dict[obj_key]
                if memory_obj.tensor is None:
                    logger.error(
                        "Engine-driven pickle store reserved an object without "
                        "a tensor (instance_id=%d, chunk_index=%d)",
                        instance_id,
                        idx,
                    )
                    continue
                chunk_cpu = chunks[idx]
                if chunk_cpu.shape != memory_obj.tensor.shape:
                    logger.error(
                        "Engine-driven pickle store chunk shape mismatch "
                        "(instance_id=%d, chunk_index=%d, chunk_shape=%s, "
                        "object_shape=%s)",
                        instance_id,
                        idx,
                        tuple(chunk_cpu.shape),
                        tuple(memory_obj.tensor.shape),
                    )
                    continue
                memory_obj.tensor.copy_(chunk_cpu)
                written_keys.append(obj_key)
        finally:
            if written_keys:
                self._storage_manager.finish_write(written_keys)

        success = len(written_keys) == len(reserved_dict)
        if not success:
            logger.error(
                "Engine-driven pickle store incomplete (instance_id=%d, "
                "object_keys=%d, reserved=%d, chunks=%d, written=%d)",
                instance_id,
                len(obj_keys),
                len(reserved_dict),
                len(chunks),
                len(written_keys),
            )
        return success

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        response, chunks = self.prepare_retrieve_chunks(
            key=key, instance_id=instance_id, resolve_obj_keys=resolve_obj_keys
        )
        if not response.success:
            return response
        return PrepareRetrieveResponse(
            success=True, data=pickle.dumps(chunks), context=response.context
        )

    def prepare_retrieve_chunks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> tuple[PrepareRetrieveResponse, list[torch.Tensor]]:
        """Read prefetched objects and return their chunk tensors."""
        obj_keys = resolve_obj_keys(key)
        prefetched_keys: list[ObjectKey] = []
        try:
            read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
            with read_ctx as maybe_memory_objs:
                if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                    return (
                        PrepareRetrieveResponse(success=False, data=b"", context={}),
                        [],
                    )
                prefetched_keys = obj_keys[: len(maybe_memory_objs)]
                chunks = []
                for memory_obj in maybe_memory_objs:
                    if memory_obj.tensor is None:
                        return (
                            PrepareRetrieveResponse(
                                success=False, data=b"", context={}
                            ),
                            [],
                        )
                    chunks.append(memory_obj.tensor.cpu().clone())
                return (
                    PrepareRetrieveResponse(success=True, data=b"", context={}),
                    chunks,
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
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Reserve SHM-backed objects and return slot descriptors.

        Returns:
            Context with ``slots`` and ``chunk_indices``.
        """
        obj_keys = resolve_obj_keys(key)
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
                self._storage_manager.finish_write(unused_keys)
        if not reserved_keys:
            return PrepareStoreResponse(context={"slots": [], "chunk_indices": []})
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            # Multi-group callers reuse the same transfer key per group, so
            # accumulate rather than overwrite -- one commit_store releases
            # all groups' reservations.
            existing = self._pending_writes.get(transfer_key)
            if existing is None:
                self._pending_writes[transfer_key] = list(reserved_keys)
            else:
                existing.extend(reserved_keys)
        return PrepareStoreResponse(
            context={"slots": slots, "chunk_indices": chunk_indices}
        )

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Finalize SHM store write locks or fallback to pickle commit.

        Multi-group prepare_store accumulates every group's reservations
        under the same transfer key (see :meth:`prepare_store`), so one
        commit_store call releases all of them together.

        Returns:
            ``False`` if no ``prepare_store`` reservation is pending for this
            key (including one that reserved nothing but was still called),
            otherwise ``True`` once any pending write locks are released.
        """
        if cpu_data != b"":
            # Non-empty payload: caller used the pickle path for this call.
            return self._fallback_strategy.commit_store(
                key=key,
                instance_id=instance_id,
                cpu_data=cpu_data,
                context=context,
                resolve_obj_keys=resolve_obj_keys,
            )
        return self._release_write_locks(key, instance_id)

    def commit_store_chunks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        chunks: list[torch.Tensor],
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> bool:
        """Finalize SHM store write locks or fallback to a chunk commit.

        Empty ``chunks`` (like ``cpu_data=b""`` in :meth:`commit_store`) means
        the worker wrote straight into the SHM slots, so this only releases
        the pending write locks.

        Returns:
            ``False`` if no ``prepare_store`` reservation is pending for this
            key, otherwise ``True`` once any pending write locks are released.
        """
        if chunks:
            return self._fallback_strategy.commit_store_chunks(
                key=key,
                instance_id=instance_id,
                chunks=chunks,
                context=context,
                resolve_obj_keys=resolve_obj_keys,
            )
        return self._release_write_locks(key, instance_id)

    def _release_write_locks(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Release the write locks a ``prepare_store`` reserved for this key.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.

        Returns:
            ``False`` if no ``prepare_store`` reservation is pending for this
            key (including one that reserved nothing but was still called),
            otherwise ``True``.
        """
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            # A key missing from the map means no prepare_store reservation
            # is pending for it, which is a caller error. Distinguished from
            # a reservation that legitimately reserved zero objects.
            if transfer_key not in self._pending_writes:
                return False
            reserved_keys = self._pending_writes.pop(transfer_key)
        if reserved_keys:
            self._storage_manager.finish_write(reserved_keys)
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Read SHM objects and return slot descriptors for worker access."""
        obj_keys = resolve_obj_keys(key)
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
        with self._pending_lock:
            # Multi-group engine-driven prepare_retrieve calls this method
            # once per LMCache group with the same transfer key. Accumulate
            # the prefetched key sets so commit_retrieve releases all of them.
            existing = self._pending_reads.get(transfer_key)
            if existing is None:
                self._pending_reads[transfer_key] = list(shm_prefetched_keys)
            else:
                existing.extend(shm_prefetched_keys)
        return PrepareRetrieveResponse(success=True, data=b"", context={"slots": slots})

    def prepare_retrieve_chunks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
    ) -> tuple[PrepareRetrieveResponse, list[torch.Tensor]]:
        """Read SHM objects, returning slot descriptors and no chunks.

        SHM hands the worker slot descriptors to read in place rather than
        tensors, so :meth:`prepare_retrieve` already emits ``data=b""`` and
        the chunk list is always empty here.
        """
        return (
            self.prepare_retrieve(
                key=key, instance_id=instance_id, resolve_obj_keys=resolve_obj_keys
            ),
            [],
        )

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Release pending SHM read locks for the completed retrieve request."""
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            prefetched_keys = self._pending_reads.pop(transfer_key, [])
        if prefetched_keys:
            self._storage_manager.finish_read_prefetched(prefetched_keys)
        return True

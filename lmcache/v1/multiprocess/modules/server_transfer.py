# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
from _thread import LockType
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any
import abc

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor
from lmcache.v1.multiprocess.transfer_context.tensor_wire import (
    decode_grouped_tensors,
    decode_tensors,
    encode_grouped_tensors,
    encode_tensors,
)

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
    read_owner_resolver: Callable[[str], dict[ObjectKey, int] | None] | None = None,
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
        positive pool size, otherwise the safe inline strategy.
    """
    if shm_name and pool_size > 0:
        logger.info("Using shm engine-driven transfer strategy")
        return ShmTransferStrategy(
            storage_manager=storage_manager,
            pending_writes=pending_writes,
            pending_reads=pending_reads,
            pending_lock=pending_lock,
            transfer_key_factory=transfer_key_factory,
            fallback_strategy=InlineTransferStrategy(
                storage_manager, read_owner_resolver
            ),
        )

    logger.info("Using safe inline engine-driven transfer strategy")
    return InlineTransferStrategy(storage_manager, read_owner_resolver)


class TransferStrategy(abc.ABC):
    """Contract for non-GPU transport backends used by the server.

    Implementations encapsulate the transport-specific prepare/commit lifecycle for
    store and retrieve operations, allowing the server to use either inline or
    shared-memory-based transfers behind a common interface.
    """

    @abc.abstractmethod
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        group_keys: "list[list[ObjectKey]] | None" = None,
        skipped_chunks: "Sequence[Sequence[int]]" = (),
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.
            group_keys: Group-major object keys for grouped-protocol workers,
                including a single compressed group; ``None`` for legacy workers.
            skipped_chunks: Per group, chunk positions the worker will not
                store (all-null engine blocks); empty means none.

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
        group_keys: "list[list[ObjectKey]] | None" = None,
    ) -> bool:
        """Finalize a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            cpu_data: Serialized payload from the worker.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.
            group_keys: Group-major object keys for grouped-protocol workers,
                including a single compressed group; ``None`` for legacy workers.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        group_keys: "list[list[ObjectKey]] | None" = None,
        group_first_chunk: "Sequence[int]" = (),
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            resolve_obj_keys: Callable that resolves object keys from ``key``.
            group_keys: Group-major object keys for grouped-protocol workers,
                already narrowed to each group's in-window suffix; ``None``
                for legacy workers.
            group_first_chunk: Per group, the chunk position of
                ``group_keys[g][0]`` within the request (non-zero when a
                window drops leading chunks); empty means all zero.

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


class InlineTransferStrategy(TransferStrategy):
    """Safe inline transport for non-GPU transfer requests.

    This is the default transport when SHM is unavailable, and it is also used as a
    fallback by the SHM strategy when the worker sends an inline serialized payload.
    ``prepare_store`` returns an empty context, while ``commit_store`` validates a
    typed tensor envelope and writes the resulting tensors into reserved objects.
    Explicit group layouts, including a single compressed group, use a
    group-major envelope validated against each group's registered layout.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        read_owner_resolver: Callable[[str], dict[ObjectKey, int] | None] | None = None,
    ) -> None:
        """Initialize the safe inline transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
        """
        self._storage_manager = storage_manager
        self._read_owner_resolver = read_owner_resolver

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        group_keys: "list[list[ObjectKey]] | None" = None,
        skipped_chunks: "Sequence[Sequence[int]]" = (),
    ) -> PrepareStoreResponse:
        """Return empty store context for inline mode.

        Inline transport does not pre-allocate SHM slots during prepare; the
        whole (legacy or grouped) payload arrives with ``commit_store``.
        ``skipped_chunks`` is not needed here: a grouped payload names
        the chunks it carries.
        """
        return PrepareStoreResponse(context={})

    def _write_chunks(
        self,
        instance_id: int,
        obj_keys: list[ObjectKey],
        chunks: list[torch.Tensor],
        layout_desc: Any,
    ) -> bool:
        """Reserve ``obj_keys``, copy the matching validated ``chunks`` in.

        Returns:
            ``True`` when every reserved object was written.
        """
        reserved_dict = self._storage_manager.reserve_write(obj_keys, layout_desc)
        written_keys: list[ObjectKey] = []
        try:
            for idx, obj_key in enumerate(obj_keys):
                if obj_key not in reserved_dict:
                    continue
                if idx >= len(chunks):
                    logger.error(
                        "Engine-driven inline store is missing chunk %d "
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
                        "Engine-driven inline store reserved an object without "
                        "a tensor (instance_id=%d, chunk_index=%d)",
                        instance_id,
                        idx,
                    )
                    continue
                chunk_cpu = chunks[idx]
                if chunk_cpu.shape != memory_obj.tensor.shape:
                    logger.error(
                        "Engine-driven inline store chunk shape mismatch "
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
            # Complete by owner: with several L1 managers the reservations can
            # live in different ones, and finish_write(keys) refuses that case.
            if written_keys:
                self._storage_manager.finish_write_by_owner(
                    self._storage_manager.prepare_write_completion(
                        {k: reserved_dict[k] for k in written_keys}
                    )
                )
            # Release reservations that were never written so their write
            # locks don't wedge later operations on the same keys, then drop
            # the unwritten objects so they can never surface as cache hits.
            written = set(written_keys)
            leftover = [k for k in reserved_dict if k not in written]
            if leftover:
                self._storage_manager.finish_write_by_owner(
                    self._storage_manager.prepare_write_completion(
                        {k: reserved_dict[k] for k in leftover}
                    )
                )
                self._storage_manager.delete_l1_keys(leftover)

        success = len(written_keys) == len(reserved_dict)
        if not success:
            logger.error(
                "Engine-driven inline store incomplete (instance_id=%d, "
                "object_keys=%d, reserved=%d, chunks=%d, written=%d)",
                instance_id,
                len(obj_keys),
                len(reserved_dict),
                len(chunks),
                len(written_keys),
            )
        return success

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        group_keys: "list[list[ObjectKey]] | None" = None,
    ) -> bool:
        """Deserialize and write validated inline chunks into reserved objects.

        Legacy payloads are a flat tensor envelope; explicit group layouts use
        a group-major envelope even for one group. Each group is validated and
        written against its own layout from ``context.group_layouts``.
        Validation runs before any storage access.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        if group_keys is not None:
            group_layouts = context.group_layouts
            if group_layouts is None or len(group_layouts) != len(group_keys):
                logger.error(
                    "multi-group inline store has %d key groups but %s "
                    "layouts are registered (instance_id=%d)",
                    len(group_keys),
                    None if group_layouts is None else len(group_layouts),
                    instance_id,
                )
                return False
            decoded = decode_grouped_tensors(
                cpu_data,
                expected=[
                    (layout.shapes[0], layout.dtypes[0], len(g_keys))
                    for layout, g_keys in zip(group_layouts, group_keys, strict=True)
                ],
            )
            # Each group carries only the chunks the worker stores; all-null
            # chunks are absent and their keys are never reserved.
            ok = True
            for g_keys, (g_indices, g_chunks), g_layout in zip(
                group_keys, decoded, group_layouts, strict=True
            ):
                ok = (
                    self._write_chunks(
                        instance_id, [g_keys[i] for i in g_indices], g_chunks, g_layout
                    )
                    and ok
                )
            return ok

        obj_keys = resolve_obj_keys(key)
        chunks = decode_tensors(
            cpu_data,
            expected_shape=context.layout_desc.shapes[0],
            expected_dtype=context.layout_desc.dtypes[0],
            expected_count=len(obj_keys),
        )
        return self._write_chunks(instance_id, obj_keys, chunks, context.layout_desc)

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        group_keys: "list[list[ObjectKey]] | None" = None,
        group_first_chunk: "Sequence[int]" = (),
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return a typed inline tensor payload.

        Legacy responses carry a flat tensor envelope. Grouped-protocol
        responses, including one group, carry a group-major envelope with each
        group's absolute chunk positions, and are all-or-nothing: a retrieve
        is a miss unless EVERY group has every chunk of its in-window suffix.
        """
        grouped = group_keys is not None
        if group_keys is None:
            group_keys = [resolve_obj_keys(key)]
        owners = (
            self._read_owner_resolver(key.request_id)
            if self._read_owner_resolver is not None
            else None
        )

        group_chunks: list[list[torch.Tensor]] = []
        for g_keys in group_keys:
            prefetched_keys: list[ObjectKey] = []
            try:
                read_ctx = self._storage_manager.read_prefetched_results(
                    g_keys, l1_owners=owners
                )
                with read_ctx as maybe_memory_objs:
                    if not maybe_memory_objs or len(maybe_memory_objs) != len(g_keys):
                        return PrepareRetrieveResponse(
                            success=False, data=b"", context={}
                        )
                    prefetched_keys = g_keys[: len(maybe_memory_objs)]
                    chunks: list[torch.Tensor] = []
                    for memory_obj in maybe_memory_objs:
                        if memory_obj.tensor is None:
                            return PrepareRetrieveResponse(
                                success=False, data=b"", context={}
                            )
                        chunks.append(memory_obj.tensor.cpu().clone())
                    group_chunks.append(chunks)
            finally:
                if prefetched_keys:
                    self._storage_manager.finish_read_prefetched(
                        prefetched_keys, l1_owners=owners
                    )

        if grouped:
            firsts = list(group_first_chunk) or [0] * len(group_keys)
            data = encode_grouped_tensors(
                [
                    (list(range(first, first + len(chunks))), chunks)
                    for first, chunks in zip(firsts, group_chunks, strict=True)
                ]
            )
        else:
            data = encode_tensors(group_chunks[0])
        return PrepareRetrieveResponse(success=True, data=data, context={})

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """No-op for inline mode; data was already copied during prepare."""
        return True


class ShmTransferStrategy(TransferStrategy):
    """Shared-memory transport for non-GPU transfer requests.

    This strategy exposes SHM slot descriptors during ``prepare_store`` and
    ``prepare_retrieve`` so workers can access storage buffers directly. It tracks
    pending SHM reservations until the matching commit step releases them, and it
    falls back to typed inline commit handling when ``cpu_data`` is non-empty.
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
        fallback_strategy: InlineTransferStrategy,
    ) -> None:
        """Initialize SHM transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
            pending_writes: Shared pending SHM write reservations map.
            pending_reads: Shared pending SHM read reservations map.
            pending_lock: Lock guarding shared pending SHM maps.
            transfer_key_factory: Factory to build `(instance_id, key)` transfer keys.
            fallback_strategy: Safe inline fallback for non-empty ``cpu_data``
                payloads.
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
        group_keys: "list[list[ObjectKey]] | None" = None,
        skipped_chunks: "Sequence[Sequence[int]]" = (),
    ) -> PrepareStoreResponse:
        """Reserve SHM-backed objects and return slot descriptors.

        Args:
            group_keys: Group-major object keys for grouped-protocol workers
                (``group_keys[g][c]`` is chunk ``c``'s key in group ``g``).
                ``None`` means legacy protocol via ``resolve_obj_keys``.
            skipped_chunks: Per group, chunk positions the worker will not
                store; they are never reserved and get no slot.

        Returns:
            Context with ``slots`` and ``chunk_indices``. Grouped-protocol
            responses also carry ``group_ids``, even for one group.
        """
        grouped = group_keys is not None
        if group_keys is None:
            group_layouts = [context.layout_desc]
            group_keys = [resolve_obj_keys(key)]
        else:
            assert context.group_layouts is not None
            group_layouts = context.group_layouts

        slots: list[dict[str, Any]] = []
        chunk_indices: list[int] = []
        group_ids: list[int] = []
        reserved_keys: list[ObjectKey] = []
        for gid, (g_keys, g_layout) in enumerate(
            zip(group_keys, group_layouts, strict=True)
        ):
            # Never reserve chunks the worker will not store (all-null engine
            # blocks): an unwritten slot would be committed as garbage.
            skip = set(skipped_chunks[gid]) if skipped_chunks else set()
            reserved = self._storage_manager.reserve_write(
                [k for idx, k in enumerate(g_keys) if idx not in skip], g_layout
            )
            g_reserved: list[ObjectKey] = []
            try:
                for idx, obj_key in enumerate(g_keys):
                    if idx in skip:
                        continue
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
                    group_ids.append(gid)
                    g_reserved.append(obj_key)
            finally:
                g_reserved_set = set(g_reserved)
                unused_keys = [
                    obj_key for obj_key in reserved if obj_key not in g_reserved_set
                ]
                if unused_keys:
                    self._storage_manager.finish_write(unused_keys)
            reserved_keys.extend(g_reserved)

        if not reserved_keys:
            empty: dict[str, Any] = {"slots": [], "chunk_indices": []}
            if grouped:
                empty["group_ids"] = []
            return PrepareStoreResponse(context=empty)
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            self._pending_writes[transfer_key] = reserved_keys
        ctx: dict[str, Any] = {"slots": slots, "chunk_indices": chunk_indices}
        if grouped:
            ctx["group_ids"] = group_ids
        return PrepareStoreResponse(context=ctx)

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        group_keys: "list[list[ObjectKey]] | None" = None,
    ) -> bool:
        """Finalize SHM store write locks or use the safe inline fallback.

        Returns:
            ``True`` when pending SHM reservation is committed successfully.
        """
        if cpu_data != b"":
            return self._fallback_strategy.commit_store(
                key=key,
                instance_id=instance_id,
                cpu_data=cpu_data,
                context=context,
                resolve_obj_keys=resolve_obj_keys,
                group_keys=group_keys,
            )
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            reserved_keys = self._pending_writes.pop(transfer_key, None)
        if reserved_keys is None:
            return False
        if reserved_keys:
            self._storage_manager.finish_write(reserved_keys)
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheServerKey], list[ObjectKey]],
        group_keys: "list[list[ObjectKey]] | None" = None,
        group_first_chunk: "Sequence[int]" = (),
    ) -> PrepareRetrieveResponse:
        """Read SHM objects and return slot descriptors for worker access.

        Args:
            group_keys: Group-major object keys for grouped-protocol workers,
                narrowed to each group's in-window suffix. A retrieve is a
                miss unless EVERY group has every one of them. ``None`` means
                legacy protocol.
            group_first_chunk: Per group, the chunk position of its first
                key; grouped-protocol responses report ``group_ids`` and
                absolute ``chunk_indices`` parallel to ``slots``, even for
                one group.
        """
        grouped = group_keys is not None
        if group_keys is None:
            group_keys = [resolve_obj_keys(key)]

        all_prefetched: list[ObjectKey] = []
        slots: list[dict[str, Any]] = []
        group_ids: list[int] = []
        chunk_indices: list[int] = []
        firsts = list(group_first_chunk) or [0] * len(group_keys)

        def _miss() -> PrepareRetrieveResponse:
            if all_prefetched:
                self._storage_manager.finish_read_prefetched(all_prefetched)
            return PrepareRetrieveResponse(success=False, data=b"", context={})

        for gid, g_keys in enumerate(group_keys):
            prefetched, memory_objs = self._storage_manager.unsafe_read(g_keys)
            all_prefetched.extend(prefetched)
            if (
                not memory_objs
                or len(prefetched) != len(g_keys)
                or len(memory_objs) != len(g_keys)
            ):
                return _miss()
            for pos, memory_obj in enumerate(memory_objs):
                if memory_obj.tensor is None:
                    return _miss()
                slots.append(
                    ShmSlotDescriptor(
                        offset=memory_obj.shm_offset,
                        length=memory_obj.shm_byte_length,
                        shape=list(memory_obj.tensor.shape),
                        dtype=_dtype_to_name(memory_obj.tensor.dtype),
                    ).to_dict()
                )
                group_ids.append(gid)
                chunk_indices.append(firsts[gid] + pos)

        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            self._pending_reads[transfer_key] = all_prefetched
        ctx: dict[str, Any] = {"slots": slots}
        if grouped:
            ctx["group_ids"] = group_ids
            ctx["chunk_indices"] = chunk_indices
        return PrepareRetrieveResponse(success=True, data=b"", context=ctx)

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

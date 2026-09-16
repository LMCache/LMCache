# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import pickle
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    GroupKind,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata

# Local
from .server_transfer import (
    TransferStrategy,
    create_transfer_strategy,
)

logger = init_logger(__name__)


def _group_null_chunk_mask(
    null_chunk_mask: tuple[tuple[bool, ...], ...] | None,
    group_id: int,
) -> tuple[bool, ...] | None:
    """Return one group's null-chunk mask from the key's per-group masks.

    Args:
        null_chunk_mask: ``key.null_chunk_mask`` -- per-group, per-chunk
            mask, or ``None`` when the sender predates the field (no chunk
            masked) or the request has no recurrent-state groups.
        group_id: The LMCache group to select.

    Returns:
        The group's mask, or ``None`` when ``null_chunk_mask`` is ``None`` or
        does not cover this group (treated as "nothing masked").
    """
    if null_chunk_mask is None or group_id >= len(null_chunk_mask):
        return None
    return null_chunk_mask[group_id]


def _is_chunk_masked(group_mask: tuple[bool, ...] | None, idx: int) -> bool:
    """Return whether chunk ``idx`` is null-masked.

    Args:
        group_mask: The group's null-chunk mask, or ``None`` (nothing
            masked).
        idx: Chunk position within the group.

    Returns:
        ``True`` if ``idx`` is within ``group_mask`` and marked null.
    """
    return group_mask is not None and idx < len(group_mask) and group_mask[idx]


def _unmasked_chunk_positions(
    group_mask: tuple[bool, ...] | None,
    num_group_chunks: int,
) -> list[int]:
    """Return one group's chunk positions that are not null-masked.

    Args:
        group_mask: The group's null-chunk mask (``True`` = masked), or
            ``None`` to treat nothing as masked.
        num_group_chunks: This group's full (unfiltered) chunk count.

    Returns:
        Chunk positions in ``range(num_group_chunks)`` that are not masked,
        in order.
    """
    return [
        idx for idx in range(num_group_chunks) if not _is_chunk_masked(group_mask, idx)
    ]


def _apply_null_chunk_mask(
    obj_keys: list[ObjectKey],
    group_mask: tuple[bool, ...] | None,
) -> list[ObjectKey]:
    """Drop the object keys of chunks marked null for one group.

    Args:
        obj_keys: One group's object keys, in chunk order.
        group_mask: That group's null-chunk mask (``True`` = drop), or
            ``None`` to keep every key.

    Returns:
        ``obj_keys`` with masked positions removed, preserving order.
    """
    if group_mask is None:
        return obj_keys
    return [
        obj_keys[idx] for idx in _unmasked_chunk_positions(group_mask, len(obj_keys))
    ]


def _rebase_masked_chunk_indices(
    reserved_chunk_indices: list[int],
    group_mask: tuple[bool, ...] | None,
    num_group_chunks: int,
    chunk_offset: int,
) -> list[int]:
    """Rebase a strategy's chunk indices from filtered to full group space.

    ``strategy.prepare_store`` returns indices into the filtered key list,
    but the worker's block IDs are never filtered, so each index is mapped
    back to its unmasked position before ``chunk_offset`` is added.

    Args:
        reserved_chunk_indices: Indices into the filtered key list that
            ``strategy.prepare_store`` reserved.
        group_mask: The mask that was applied before resolving those indices,
            or ``None`` if none was (indices already match the full range).
        num_group_chunks: This group's full (unfiltered) chunk count.
        chunk_offset: Flat offset of this group's first chunk in the
            multi-group sequence.

    Returns:
        ``reserved_chunk_indices``, rebased to the full multi-group flat
        chunk sequence.
    """
    if group_mask is None:
        return [idx + chunk_offset for idx in reserved_chunk_indices]
    full_positions = _unmasked_chunk_positions(group_mask, num_group_chunks)
    return [full_positions[idx] + chunk_offset for idx in reserved_chunk_indices]


def _masked_per_group_keys(
    per_group_keys: list[list[ObjectKey]],
    null_chunk_mask: tuple[tuple[bool, ...], ...] | None,
) -> list[list[ObjectKey]]:
    """Apply each group's null-chunk mask to its resolved object keys.

    Args:
        per_group_keys: Every group's resolved object keys, in protocol
            order, as returned by ``_resolve_per_group_obj_keys``.
        null_chunk_mask: ``key.null_chunk_mask`` -- per-group, per-chunk
            mask, or ``None`` when nothing is masked.

    Returns:
        ``per_group_keys`` with each group's masked (null-chunk) positions
        removed.
    """
    return [
        _apply_null_chunk_mask(
            group_keys, _group_null_chunk_mask(null_chunk_mask, group_id)
        )
        for group_id, group_keys in enumerate(per_group_keys)
    ]


@dataclass
class EngineDrivenContextEntry:
    """Registered non-GPU context metadata for a single worker instance.

    Attributes:
        metadata: Layout metadata describing the non-CUDA chunk format.
        model_name: The name of the model associated with this context.
        world_size: The world size associated with this context.
        last_seen: ``time.monotonic()`` of the most recent activity from this
            instance (register, PING, prepare/commit). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window. Latched only by PING.
        metadata_by_group: Per-LMCache-group layout metadata, in protocol
            order. Empty for a single non-hybrid group, in which case
            ``metadata`` above is used as-is.
    """

    metadata: EngineDrivenContextMetadata
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    metadata_by_group: list[EngineDrivenContextMetadata] = field(default_factory=list)


class EngineDrivenTransferModule(InstanceLivenessTarget):
    """Handles Engine-driven KV cache transfer operations.

    Owns non-GPU context registrations and provides handlers for
    register, unregister, prepare/commit store, and prepare/commit retrieve
    of CPU-serialized KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._engine_driven_contexts: dict[int, EngineDrivenContextEntry] = {}
        self._strategies: dict[int, TransferStrategy] = {}
        # Guards _engine_driven_contexts and _strategies together (the reaper
        # mutates them off the MQ main loop). Leaf lock, never held with
        # _pending_shm_lock.
        self._lock = threading.Lock()
        self._pending_shm_writes: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_reads: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_lock = threading.Lock()
        self._shm_pool_info: ShmPoolInfo = self._ctx.shm_pool_info

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict:
        """Return non-GPU transfer module status information.

        Returns:
            A dict containing registered non-CUDA instance IDs and
            per-instance context metadata.
        """
        registered_non_cuda_ids: list[int] = []
        non_cuda_context_meta: dict[str, dict] = {}

        with self._lock:
            entries = dict(self._engine_driven_contexts)
        for instance_id, entry in entries.items():
            registered_non_cuda_ids.append(instance_id)
            non_cuda_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "block_size": entry.metadata.block_size,
                "use_mla": entry.metadata.use_mla,
            }

        return {
            "registered_non_cuda_instance_ids": registered_non_cuda_ids,
            "non_cuda_context_meta": non_cuda_context_meta,
        }

    def close(self) -> None:
        """Release resources owned by this module."""
        with self._lock:
            self._engine_driven_contexts.clear()
            self._strategies.clear()

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered non-GPU instances."""
        with self._lock:
            return len(self._engine_driven_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap non-GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one that
        has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, EngineDrivenContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._engine_driven_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                entry = self._engine_driven_contexts.pop(iid)
                self._strategies.pop(iid, None)
                reaped.append((iid, entry))
        for iid, entry in reaped:
            self._release_entry(iid, entry)
            logger.warning(
                "Reaped non-GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - entry.last_seen,
                entry.has_liveness_signal,
            )
        return [iid for iid, _ in reaped]

    def _resolve_for_transfer(
        self, instance_id: int
    ) -> tuple[EngineDrivenContextEntry, TransferStrategy]:
        """Return (entry, strategy) for a transfer, refreshing last_seen.

        Pair-atomicity guarantees the entry exists whenever the strategy
        does. Refreshes last_seen (no latch) so an active worker is not
        reaped mid-transfer.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry and its transfer strategy.

        Raises:
            ValueError: If the instance is not registered (or was reaped).
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            strategy = self._strategies.get(instance_id)
            if entry is None or strategy is None:
                raise ValueError(
                    "non-GPU context not registered (or reaped) for "
                    f"instance ID {instance_id}"
                )
            entry.last_seen = now
            return entry, strategy

    def _release_entry(self, instance_id: int, entry: EngineDrivenContextEntry) -> None:
        """Release resources for a popped entry (run outside the lock).

        Sweeps the instance's pending SHM transfers and unregisters its
        layout descriptor.

        Args:
            instance_id: The popped instance ID.
            entry: The popped entry.
        """
        with self._pending_shm_lock:
            stale_writes = [k for k in self._pending_shm_writes if k[0] == instance_id]
            stale_reads = [k for k in self._pending_shm_reads if k[0] == instance_id]
            write_obj_keys = [self._pending_shm_writes.pop(k) for k in stale_writes]
            read_obj_keys = [self._pending_shm_reads.pop(k) for k in stale_reads]

        for obj_keys in write_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_write(obj_keys)
        for obj_keys in read_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_read_prefetched(obj_keys)

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)

    @staticmethod
    def _make_transfer_key(
        key: IPCCacheServerKey, instance_id: int
    ) -> tuple[int, IPCCacheServerKey]:
        return (instance_id, key)

    def _resolve_per_group_obj_keys(
        self, key: IPCCacheServerKey, num_groups: int
    ) -> list[list[ObjectKey]]:
        """Resolve every LMCache group's object keys in one pass.

        One resolve per transfer instead of one per group keeps a
        multi-group RPC linear in the group count.

        Args:
            key: Cache key for the token range.
            num_groups: Total number of registered LMCache groups (``1`` for
                the single-group fallback).

        Returns:
            Element ``i`` holds the object keys for group ``i``. Each key
            retains its own ``object_group_id``, so downstream storage
            lookups still resolve per group.
        """
        return self._ctx.resolve_obj_keys(key, list(range(num_groups)))

    @staticmethod
    def _flatten_obj_keys(per_group_keys: list[list[ObjectKey]]) -> list[ObjectKey]:
        """Flatten per-group object keys group-major.

        Matches the worker's concatenation order, so a positional zip
        against its flat chunk list lines up.

        Args:
            per_group_keys: Object keys per LMCache group, in protocol order,
                as returned by :meth:`_resolve_per_group_obj_keys`.

        Returns:
            Object keys for all groups, flattened group-major.
        """
        return [obj_key for group_keys in per_group_keys for obj_key in group_keys]

    @staticmethod
    def _const_obj_keys(
        obj_keys: list[ObjectKey],
    ) -> Callable[[IPCCacheServerKey], list[ObjectKey]]:
        """Return a ``resolve_obj_keys`` callback that ignores its key and
        always returns ``obj_keys``.

        The per-group keys are already resolved by the time a strategy call
        needs this callback, so the callback itself is a constant lookup.
        """
        return lambda _key: obj_keys

    def _make_attn_window_desc(
        self, engine_group_infos: Sequence[EngineGroupInfo]
    ) -> AttnWindowDesc:
        """Build the per-object-group attention windows for a registration.

        Read straight off ``engine_group_infos`` rather than bucketed like
        the CUDA path, since each group already maps 1:1 to an object group.

        Args:
            engine_group_infos: The worker's registered LMCache groups, in
                protocol order. Must be non-empty.

        Returns:
            One window and one kind label per group, in protocol order.
            ``-1`` marks a full-attention group (the whole prefix must be
            present); ``>= 1`` is a sliding window in chunks.
        """
        chunk_size = self._ctx.chunk_size
        windows: list[int] = []
        kinds: list[GroupKind] = []
        for group_info in engine_group_infos:
            sw_tokens = group_info.sw_size_tokens
            if sw_tokens < 1:
                # -1 (not sliding-window) and any non-positive report both
                # mean "needs the whole prefix".
                windows.append(-1)
            else:
                # Round up: a partial trailing chunk is still required.
                window = (sw_tokens + chunk_size - 1) // chunk_size
                windows.append(window if window >= 1 else -1)
            # Extra (connector-private) pools are standalone; otherwise the
            # group is recurrent state or ordinary attention KV.
            if group_info.extra_object_group_tag != 0:
                kinds.append("aux")
            elif group_info.recurrent_state:
                kinds.append("recurrent")
            else:
                kinds.append("attention")
        return AttnWindowDesc(num_chunks_in_sw=windows, group_kinds=tuple(kinds))

    @staticmethod
    def _make_group_layout_desc(
        num_layers: int,
        num_physical_slots: int,
        hidden_dim_size: int,
        dtype: torch.dtype,
        use_mla: bool,
    ) -> MemoryLayoutDesc:
        """Build one LMCache group's chunk layout descriptor.

        Args:
            num_layers: Number of layers carried by this group.
            num_physical_slots: Physical KV slots gathered into one chunk.
            hidden_dim_size: Flattened hidden dimension per token.
            dtype: Torch dtype of the chunk tensor.
            use_mla: Whether the worker KV format is single-plane (MLA or
                fused-K/V).

        Returns:
            A :class:`MemoryLayoutDesc` describing one chunk for this group.
        """
        shape = (
            torch.Size([num_layers, num_physical_slots, hidden_dim_size])
            if use_mla
            else torch.Size([2, num_layers, num_physical_slots, hidden_dim_size])
        )
        return MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])

    @request_handler(RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT)
    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla,
                num_physical_slots, engine_group_infos).

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]

        now = time.monotonic()
        with self._lock:
            existing = self._engine_driven_contexts.get(payload.instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered (non-GPU); refreshing liveness",
                    payload.instance_id,
                )
                return RegisterEngineDrivenContextResponse(
                    shm_name=shm_name, pool_size=pool_size
                )

        dtype = getattr(torch, payload.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{payload.dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )

        num_physical_slots = payload.num_physical_slots
        if num_physical_slots is None:
            # Compatibility with clients from before the physical-slot field
            # was added. Those clients require one slot per logical token.
            num_physical_slots = self._ctx.chunk_size
        elif num_physical_slots <= 0:
            raise ValueError(
                f"num_physical_slots must be positive, got {num_physical_slots}"
            )

        layout_desc = self._make_group_layout_desc(
            payload.num_layers,
            num_physical_slots,
            payload.hidden_dim_size,
            dtype,
            payload.use_mla,
        )
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
        )
        # Hybrid groups can pack K/V differently (e.g. fused vs split), so a
        # group's per-token width can differ from the shared hidden_dim_size.
        # Missing entries fall back to it.
        group_hidden_dim_sizes = payload.group_hidden_dim_sizes or []
        metadata_by_group = [
            EngineDrivenContextMetadata(
                layout_desc=self._make_group_layout_desc(
                    len(group_info.layer_indices),
                    num_physical_slots,
                    (
                        group_hidden_dim_sizes[i]
                        if i < len(group_hidden_dim_sizes)
                        else payload.hidden_dim_size
                    ),
                    dtype,
                    payload.use_mla,
                ),
                block_size=group_info.tokens_per_block or payload.block_size,
                use_mla=payload.use_mla,
            )
            for i, group_info in enumerate(payload.engine_group_infos)
        ]
        # Build the entry and strategy outside the lock, then insert the pair
        # atomically so a concurrent reap can never strand one without the
        # other. REGISTER is SYNC-serialized, so it is the sole inserter.
        entry = EngineDrivenContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
            last_seen=now,
            has_liveness_signal=False,
            metadata_by_group=metadata_by_group,
        )
        strategy: TransferStrategy = create_transfer_strategy(
            self._ctx.storage_manager,
            shm_name=shm_name,
            pool_size=pool_size,
            pending_writes=self._pending_shm_writes,
            pending_reads=self._pending_shm_reads,
            pending_lock=self._pending_shm_lock,
            transfer_key_factory=self._make_transfer_key,
        )
        with self._lock:
            self._engine_driven_contexts[payload.instance_id] = entry
            self._strategies[payload.instance_id] = strategy

        logger.info(
            "Registered non-GPU context for instance %d (model=%s, world_size=%d, "
            "num_groups=%d)",
            payload.instance_id,
            payload.model_name,
            payload.world_size,
            max(1, len(metadata_by_group)),
        )

        if metadata_by_group:
            # Hybrid: each group keeps its own layout_desc (distinct chunk
            # shapes for attention vs. mamba groups).
            self._ctx.layout_desc_registry.register(
                payload.model_name,
                payload.world_size,
                layout_desc,
                attn_desc=self._make_attn_window_desc(payload.engine_group_infos),
                group_layout_descs={
                    idx: m.layout_desc for idx, m in enumerate(metadata_by_group)
                },
            )
        else:
            # Non-hybrid: one shared layout_desc, matching pre-hybrid behavior.
            self._ctx.layout_desc_registry.register(
                payload.model_name, payload.world_size, layout_desc
            )
        return RegisterEngineDrivenContextResponse(
            shm_name=shm_name, pool_size=pool_size
        )

    @request_handler(RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT)
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister a non-GPU KV cache context for the given instance ID.

        Args:
            instance_id: The worker instance identifier.
        """
        with self._lock:
            entry = self._engine_driven_contexts.pop(instance_id, None)
            if entry is not None:
                self._strategies.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered non-GPU context found for instance ID %d",
                instance_id,
            )
            return

        self._release_entry(instance_id, entry)
        logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    @request_handler(
        RequestType.PREPARE_STORE,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare a store operation.

        ``slots`` / ``chunk_indices`` are concatenated group-major, offset
        by earlier groups' chunk counts, to match the worker's flat gather
        order. ``key.null_chunk_mask`` excludes null-block chunks from
        reservation like already-cached ones -- the offset still counts
        them, only ``chunk_indices`` omits them.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.

        Returns:
            PrepareStoreResponse with empty slots for pickle mode.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        contexts = entry.metadata_by_group or [entry.metadata]
        per_group_keys = self._resolve_per_group_obj_keys(key, len(contexts))
        all_slots: list[dict] = []
        all_chunk_indices: list[int] = []
        # No "slots" key (pickle mode) means "no pre-allocated buffers" to
        # the worker, while an empty slots *list* means "all chunks cached,
        # skip the store" -- so only emit the key if a group reported it.
        saw_slots_key = False
        chunk_offset = 0
        for group_id, context in enumerate(contexts):
            group_keys = per_group_keys[group_id]
            group_mask = _group_null_chunk_mask(key.null_chunk_mask, group_id)
            keys_to_reserve = _apply_null_chunk_mask(group_keys, group_mask)
            group_response = strategy.prepare_store(
                key=key,
                instance_id=instance_id,
                context=context,
                resolve_obj_keys=self._const_obj_keys(keys_to_reserve),
            )
            group_context = group_response.context
            if "slots" in group_context:
                saw_slots_key = True
                all_slots.extend(group_context["slots"])
                all_chunk_indices.extend(
                    _rebase_masked_chunk_indices(
                        group_context.get("chunk_indices", []),
                        group_mask,
                        len(group_keys),
                        chunk_offset,
                    )
                )
            chunk_offset += len(group_keys)

        session = self._ctx.session_manager.get_or_create(key.request_id)
        # Keyed by token range, not a fixed name: chunks of one request
        # commit interleaved, so a shared key would let one chunk's
        # commit_store pop another's timestamp.
        session.extras[f"store_start_time:{key.start}:{key.end}"] = time.perf_counter()
        if not saw_slots_key:
            return PrepareStoreResponse(context={})
        return PrepareStoreResponse(
            context={"slots": all_slots, "chunk_indices": all_chunk_indices}
        )

    @request_handler(
        RequestType.COMMIT_STORE,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Commit serialized CPU chunks to storage.

        SHM mode commits all groups in one call, since committing per group
        would release the shared write-lock entry after the first. Pickle
        mode loops per group over a single decode of ``cpu_data``. Both
        apply ``key.null_chunk_mask`` first -- the worker only sent chunks
        that survived it.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            cpu_data: Pickled list of CPU tensors produced by the worker, or
                ``b""`` for SHM mode.

        Returns:
            ``True`` when all reserved objects are written, otherwise ``False``.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        contexts = entry.metadata_by_group or [entry.metadata]
        per_group_keys = self._resolve_per_group_obj_keys(key, len(contexts))
        per_group_keys = _masked_per_group_keys(per_group_keys, key.null_chunk_mask)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop(f"store_start_time:{key.start}:{key.end}", None)

        if not cpu_data:
            # SHM mode: the worker wrote chunks straight into the reserved
            # slots, so this call only releases the write locks.
            flat_obj_keys = self._flatten_obj_keys(per_group_keys)
            result = strategy.commit_store_chunks(
                key=key,
                instance_id=instance_id,
                chunks=[],
                context=contexts[0],
                resolve_obj_keys=self._const_obj_keys(flat_obj_keys),
            )
        else:
            # Pickle mode: cpu_data is the worker's flat, group-major list
            # of CPU chunk tensors, decoded once and sliced per group below.
            all_chunks: list[torch.Tensor] = pickle.loads(cpu_data)
            result = True
            chunk_offset = 0
            for group_id, context in enumerate(contexts):
                group_obj_keys = per_group_keys[group_id]
                group_chunks = all_chunks[
                    chunk_offset : chunk_offset + len(group_obj_keys)
                ]
                chunk_offset += len(group_obj_keys)
                group_ok = strategy.commit_store_chunks(
                    key=key,
                    instance_id=instance_id,
                    chunks=group_chunks,
                    context=context,
                    resolve_obj_keys=self._const_obj_keys(group_obj_keys),
                )
                result = result and group_ok

        if st is not None and result:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                key.end - key.start,
                time.perf_counter() - st,
            )
        return result

    @request_handler(
        RequestType.PREPARE_RETRIEVE,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Retrieve prefetched chunks and return serialized CPU tensors.

        Results are concatenated group-major to match the worker's flat
        gather order, then serialized once (pickle mode). A miss in any
        group fails the whole retrieve.

        ``key.null_chunk_mask`` excludes never-stored null-block chunks
        from the lookup -- the worker already skips them when scattering.

        Args:
            key: Cache key for the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            PrepareRetrieveResponse with serialized data on hit.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        num_groups = max(1, len(entry.metadata_by_group))
        per_group_keys = self._resolve_per_group_obj_keys(key, num_groups)
        per_group_keys = _masked_per_group_keys(per_group_keys, key.null_chunk_mask)
        all_slots: list[dict] = []
        all_chunks: list[torch.Tensor] = []
        success = True
        for group_id in range(num_groups):
            group_obj_keys = per_group_keys[group_id]
            group_response, group_chunks = strategy.prepare_retrieve_chunks(
                key=key,
                instance_id=instance_id,
                resolve_obj_keys=self._const_obj_keys(group_obj_keys),
            )
            if not group_response.success:
                # A miss in any one group fails the whole retrieve.
                success = False
                break
            all_slots.extend(group_response.context.get("slots", []))
            all_chunks.extend(group_chunks)

        session = self._ctx.session_manager.get_or_create(key.request_id)
        # Keyed by this call's token range for the same reason as
        # prepare_store's store_start_time: concurrent chunks of the same
        # request can interleave their prepare/commit calls.
        retrieve_time_key = f"retrieve_start_time:{key.start}:{key.end}"
        session.extras[retrieve_time_key] = time.perf_counter()
        if not success:
            # Groups before the miss may have accumulated read locks under
            # the shared transfer key. Release them here rather than relying
            # on the worker to still call commit_retrieve after a miss.
            strategy.commit_retrieve(key=key, instance_id=instance_id)
            session.extras.pop(retrieve_time_key, None)
            return PrepareRetrieveResponse(success=False, data=b"", context={})
        return PrepareRetrieveResponse(
            success=True,
            data=pickle.dumps(all_chunks) if all_chunks else b"",
            context={"slots": all_slots},
        )

    @request_handler(
        RequestType.COMMIT_RETRIEVE,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation.

        Releases every group's SHM read locks in one call, since they were
        all accumulated under the same transfer key.

        Args:
            key: Cache key for the token range retrieved.
            instance_id: Worker instance identifier (unused for pickle).

        Returns:
            ``True`` when the retrieve finalizes successfully.
        """
        _entry, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop(f"retrieve_start_time:{key.start}:{key.end}", None)
        result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        if st is not None:
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                key.end - key.start,
                time.perf_counter() - st,
            )
        return result

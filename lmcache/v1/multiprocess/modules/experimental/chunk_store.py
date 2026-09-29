# SPDX-License-Identifier: Apache-2.0
"""Optional chunk store RPCs with explicitly leased completion events."""

# Standard
from dataclasses import dataclass, field
import threading
import uuid

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.multiprocess.chunk_event_future import ChunkStoreResponse
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
    all_null_chunk_masks,
    get_layout_desc,
)
from lmcache.v1.multiprocess.native_completion import submit_callback_to_stream
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    kept_blocks_per_chunk,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.platform.base.event_ipc import EventIPCBackend
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Backpressure rejects new requests; it must never evict an unacknowledged lease.
MAX_LEASES_PER_INSTANCE = 1024


@dataclass
class _EventLease:
    backend: EventIPCBackend
    device: object
    terminal: object
    events: list[object] = field(default_factory=list)


class ChunkStoreModule(InstanceLivenessTarget):
    """Store one logical range with source-safe completion events per chunk.

    Enable with ``--enable chunk_store``. Registrations and native completion
    callbacks are shared with the ordinary transfer module. Lease ownership is
    local to this module, serialized with enqueue and worker cleanup by a lock.
    No host/device synchronization occurs on the successful store submission path.

    Args:
        ctx: Shared storage, geometry and observability context.
        transfer_module: Owner of registered KV contexts and host callbacks.
    """

    def __init__(
        self, ctx: MPCacheServerContext, transfer_module: LMCacheDrivenTransferModule
    ) -> None:
        self._ctx = ctx
        self._transfer = transfer_module
        self._lock = threading.RLock()
        self._leases: dict[int, dict[str, _EventLease]] = {}
        transfer_module.add_unregister_listener(self.drop_instance_state)
        transfer_module.register_host_func(
            "abort_chunk_store", ctx.storage_manager.abort_write, list[ObjectKey]
        )

    @property
    def context(self) -> MPCacheServerContext:
        """Return this module's shared engine context."""
        return self._ctx

    def report_status(self) -> dict[str, int]:
        """Return the number of outstanding event leases."""
        with self._lock:
            return {"chunk_store_event_leases": sum(map(len, self._leases.values()))}

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def store_with_chunk_events(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> ChunkStoreResponse:
        """Submit all copies, retaining events until the client acknowledges them.

        Args:
            key: Worker cache key describing the absolute token range.
            instance_id: Registered worker ID.
            block_ids: Raw block IDs in kernel-group order.
            event_ipc_handle: Producer event ordering reads of the source KV.

        Returns:
            Terminal handle, ordered (handle, start, end) chunks, success, lease ID.
            An unregistered worker, malformed block range, or exhausted lease
            budget is rejected with empty handles and False, before device work.
            A transfer failure returns False and a terminal event covering all
            submitted work; successfully enqueued chunk events remain source-safe.

        Raises:
            RuntimeError: The registered context has no event backend.
        """
        with self._lock:
            entry = self._transfer.get_and_touch_context_entry(instance_id)
            if entry is None:
                return b"", [], False, ""
            if len(self._leases.get(instance_id, {})) >= MAX_LEASES_PER_INSTANCE:
                logger.warning(
                    "Chunk store event lease budget exhausted for %s", instance_id
                )
                return b"", [], False, ""
            return self._store(key, instance_id, block_ids, event_ipc_handle, entry)

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def release_chunk_store_events(self, instance_id: int, lease_id: str) -> None:
        """Release one lease after the client has stopped using imported events.

        Args:
            instance_id: Worker owning the lease.
            lease_id: ID returned by store_with_chunk_events.

        Notes:
            Idempotent, including after worker cleanup. The client must destroy
            all imported events first. Device completion alone is not permission
            to destroy an exporter; this RPC is the consumer's acknowledgement.
        """
        with self._lock:
            leases = self._leases.get(instance_id)
            if leases is None:
                return
            lease = leases.get(lease_id)
            if lease is not None:
                lease.backend.synchronize_event(lease.terminal, lease.device)
                del leases[lease_id]
            if not leases:
                del self._leases[instance_id]

    def drop_instance_state(self, instance_id: int) -> None:
        """Retire a worker's leases after unregister or liveness eviction.

        Args:
            instance_id: Worker whose future handles are no longer valid.

        Notes:
            Unregister requires clients to drain their futures first. A reaped
            worker must discard old futures and register again before reuse.
        """
        with self._lock:
            for lease in self._leases.pop(instance_id, {}).values():
                lease.backend.synchronize_event(lease.terminal, lease.device)

    def close(self) -> None:
        """Drain device work and release leases after request transport shutdown."""
        self._transfer.remove_unregister_listener(self.drop_instance_state)
        with self._lock:
            for instance_id in list(self._leases):
                self.drop_instance_state(instance_id)

    def _store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        block_ids: list[list[int]],
        producer_handle: bytes,
        entry: ContextEntry,
    ) -> ChunkStoreResponse:
        cache = entry.cache_context
        backend = entry.event_backend
        if backend is None:
            raise RuntimeError("Registered cache context has no event backend")
        groups = cache.kv_layer_groups_manager
        keys_by_group = self._ctx.resolve_obj_keys(
            key, list(range(groups.num_object_groups))
        )
        num_chunks = len(keys_by_group[0])
        blocks_per_chunk = [
            cache.calculate_num_blocks(self._ctx.chunk_size, group)
            for group in range(groups.num_kernel_groups)
        ]
        if len(block_ids) != len(blocks_per_chunk) or any(
            len(ids) != num_chunks * stride
            for ids, stride in zip(block_ids, blocks_per_chunk, strict=True)
        ):
            return b"", [], False, ""
        skipped = all_null_chunk_masks(
            block_ids,
            groups.object_groups,
            blocks_per_chunk,
            num_chunks,
            self._ctx.null_block_id,
        )
        reserved: dict[ObjectKey, MemoryObj] = {}
        objects_by_group: list[list[MemoryObj | None]] = []
        chunks: list[tuple[bytes, int, int]] = []
        lease_id = uuid.uuid4().hex
        transfer_key = next_transfer_key(key.request_id)
        succeeded = False
        with torch_dev.device(cache.device), torch_dev.stream(cache.stream):
            terminal = backend.create_event(cache.device)
            lease = _EventLease(backend, cache.device, terminal)
            self._leases.setdefault(instance_id, {})[lease_id] = lease
            try:
                staged = downsample_and_stage_block_ids(cache, block_ids)
                producer = backend.import_event(producer_handle, cache.device)
                lease.events.append(producer)
                backend.wait_event(producer, cache.stream)
                self._ctx.event_bus.publish(
                    Event(
                        event_type=EventType.MP_STORE_SUBMITTED,
                        session_id=key.request_id,
                        metadata={"device": str(cache.device)},
                    )
                )
                if key.worker_id == 0 and self._ctx.event_bus.has_subscribers(
                    EventType.MP_TOKENS
                ):
                    self._transfer.publish_token_bindings(key, keys_by_group[0])
                self._ctx.event_bus.publish_on_stream(
                    cache.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_START,
                        session_id=key.request_id,
                        metadata={
                            "device": str(cache.device),
                            "engine_id": instance_id,
                            "model_name": entry.model_name,
                            "transfer_key": transfer_key,
                        },
                    ),
                )
                for group_id, keys in enumerate(keys_by_group):
                    available = self._ctx.storage_manager.reserve_write(
                        [k for i, k in enumerate(keys) if not skipped[group_id][i]],
                        get_layout_desc(cache, self._ctx.chunk_size, group_id),
                    )
                    reserved.update(available)
                    objects_by_group.append([available.get(k) for k in keys])
                strides = [
                    kept_blocks_per_chunk(cache, g)
                    for g in range(groups.num_kernel_groups)
                ]
                for chunk in range(num_chunks):
                    device_ids = [
                        ids[chunk * n : (chunk + 1) * n]
                        for ids, n in zip(staged, strides, strict=True)
                    ]
                    host_ids = [
                        ids[chunk * n : (chunk + 1) * n]
                        for ids, n in zip(block_ids, strides, strict=True)
                    ]
                    for group_id, objects in enumerate(objects_by_group):
                        transfer_kv_per_object_group(
                            cache,
                            device_ids,
                            objects[chunk : chunk + 1],
                            object_group_id=group_id,
                            batch_size=1,
                            skip_first_n_tokens=0,
                            direction=lmcache_native.TransferDirection.D2H,
                            transfer_key=transfer_key,
                            block_ids_host=host_ids,
                        )
                    event = backend.create_event(cache.device)
                    lease.events.append(event)
                    backend.record_event(event, cache.stream)
                    start = key.start + chunk * self._ctx.chunk_size
                    chunks.append(
                        (
                            backend.export_event(event, cache.device),
                            start,
                            min(start + self._ctx.chunk_size, key.end),
                        )
                    )
                succeeded = True
            except Exception:
                # RPC boundary: preserve the completion handle even after a
                # partially submitted transfer so source buffers stay protected.
                logger.exception("Chunk store failed for request %s", key.request_id)
            finally:
                if reserved:
                    submit_callback_to_stream(
                        cache.cupy_stream,
                        "finish_write" if succeeded else "abort_chunk_store",
                        list(reserved),
                    )
                backend.record_event(terminal, cache.stream)
                self._ctx.event_bus.publish_on_stream(
                    cache.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "device": str(cache.device),
                            "engine_id": instance_id,
                            "model_name": entry.model_name,
                            "transfer_key": transfer_key,
                            "stored_count": len(reserved) if succeeded else 0,
                            "total_bytes": sum(m.get_size() for m in reserved.values())
                            if succeeded
                            else 0,
                            "num_tokens": num_chunks * self._ctx.chunk_size
                            if succeeded and reserved
                            else 0,
                        },
                    ),
                )
            return (
                backend.export_event(terminal, cache.device),
                chunks,
                succeeded,
                lease_id,
            )

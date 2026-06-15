# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata

# Local
from .server_transfer import (
    TransferStrategy,
    create_transfer_strategy,
)

logger = init_logger(__name__)


@dataclass
class EngineDrivenContextEntry:
    """Registered non-GPU context metadata for a single worker instance.

    Attributes:
        metadata: Layout metadata describing the non-CUDA chunk format.
        model_name: The name of the model associated with this context.
        world_size: The world size associated with this context.
    """

    metadata: EngineDrivenContextMetadata
    model_name: str
    world_size: int


class EngineDrivenTransferModule:
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

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                self.register_kv_cache_engine_driven_context,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                self.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.PREPARE_STORE,
                self.prepare_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_STORE,
                self.commit_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.PREPARE_RETRIEVE,
                self.prepare_retrieve,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_RETRIEVE,
                self.commit_retrieve,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        """Return non-GPU transfer module status information.

        Returns:
            A dict containing registered non-CUDA instance IDs and
            per-instance context metadata.
        """
        registered_non_cuda_ids: list[int] = []
        non_cuda_context_meta: dict[str, dict] = {}

        for instance_id, entry in self._engine_driven_contexts.items():
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
        self._engine_driven_contexts.clear()
        self._strategies.clear()

    @staticmethod
    def _make_transfer_key(
        key: IPCCacheServerKey, instance_id: int
    ) -> tuple[int, IPCCacheServerKey]:
        return (instance_id, key)

    def _resolve_single_group_obj_keys(self, key: IPCCacheServerKey) -> list[ObjectKey]:
        """Resolve object keys for the single object group used by
        non-GPU transfers."""
        return self._ctx.resolve_obj_keys(key, [0])[0]

    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla).

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]

        if payload.instance_id in self._engine_driven_contexts:
            logger.warning(
                "Instance %s's KV cache is already registered, "
                "skipping the new registration",
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

        shape = (
            torch.Size(
                [payload.num_layers, self._ctx.chunk_size, payload.hidden_dim_size]
            )
            if payload.use_mla
            else torch.Size(
                [2, payload.num_layers, self._ctx.chunk_size, payload.hidden_dim_size]
            )
        )
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
        )
        self._engine_driven_contexts[payload.instance_id] = EngineDrivenContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
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
        self._strategies[payload.instance_id] = strategy

        logger.info(
            "Registered non-GPU context for instance %d (model=%s, world_size=%d)",
            payload.instance_id,
            payload.model_name,
            payload.world_size,
        )

        self._ctx.layout_desc_registry.register(
            payload.model_name, payload.world_size, layout_desc
        )
        return RegisterEngineDrivenContextResponse(
            shm_name=shm_name, pool_size=pool_size
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister a non-GPU KV cache context for the given instance ID.

        Args:
            instance_id: The worker instance identifier.
        """
        entry = self._engine_driven_contexts.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered non-GPU context found for instance ID %d",
                instance_id,
            )
            return

        self._strategies.pop(instance_id, None)

        with self._pending_shm_lock:
            stale_writes = []
            stale_reads = []
            for transfer_key in self._pending_shm_writes:
                if transfer_key[0] == instance_id:
                    stale_writes.append(transfer_key)
            for transfer_key in self._pending_shm_reads:
                if transfer_key[0] == instance_id:
                    stale_reads.append(transfer_key)

            write_obj_keys = []
            for transfer_key in stale_writes:
                write_obj_keys.append(self._pending_shm_writes.pop(transfer_key))

            read_obj_keys = []
            for transfer_key in stale_reads:
                read_obj_keys.append(self._pending_shm_reads.pop(transfer_key))

        for obj_keys in write_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_write(obj_keys)
        for obj_keys in read_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_read_prefetched(obj_keys)

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)
        logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare a store operation.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.

        Returns:
            PrepareStoreResponse with empty slots for pickle mode.
        """
        entry = self._engine_driven_contexts.get(instance_id)
        if entry is None:
            raise ValueError(
                f"non-CUDA context not registered for instance ID {instance_id}"
            )
        strategy = self._strategies.get(instance_id)
        if strategy is None:
            raise ValueError(
                f"transfer strategy not registered for instance ID {instance_id}"
            )
        response = strategy.prepare_store(
            key=key,
            instance_id=instance_id,
            context=entry.metadata,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["store_start_time"] = time.perf_counter()
        return response

    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Commit serialized CPU chunks to storage.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            cpu_data: Pickled list of CPU tensors produced by the worker.

        Returns:
            ``True`` when all reserved objects are written, otherwise ``False``.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry = self._engine_driven_contexts.get(instance_id)
        if entry is None:
            raise ValueError(
                f"non-CUDA context not registered for instance ID {instance_id}"
            )
        strategy = self._strategies.get(instance_id)
        if strategy is None:
            raise ValueError(
                f"transfer strategy not registered for instance ID {instance_id}"
            )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("store_start_time", None)
        result = strategy.commit_store(
            key=key,
            instance_id=instance_id,
            cpu_data=cpu_data,
            context=entry.metadata,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
        )
        if st is not None and result:
            num_tokens = (
                len(self._resolve_single_group_obj_keys(key)) * self._ctx.chunk_size
            )
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result

    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Retrieve prefetched chunks and return serialized CPU tensors.

        Args:
            key: Cache key for the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            PrepareRetrieveResponse with serialized data on hit.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        strategy = self._strategies.get(instance_id)
        if strategy is None:
            raise ValueError(
                f"transfer strategy not registered for instance ID {instance_id}"
            )
        response = strategy.prepare_retrieve(
            key=key,
            instance_id=instance_id,
            resolve_obj_keys=self._resolve_single_group_obj_keys,
        )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["retrieve_start_time"] = time.perf_counter()
        return response

    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation.

        Args:
            key: Cache key (unused for pickle).
            instance_id: Worker instance identifier (unused for pickle).

        Returns:
            Always ``True``.
        """
        strategy = self._strategies.get(instance_id)
        if strategy is None:
            raise ValueError(
                f"transfer strategy not registered for instance ID {instance_id}"
            )
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("retrieve_start_time", None)
        result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        if st is not None:
            num_tokens = (
                len(self._resolve_single_group_obj_keys(key)) * self._ctx.chunk_size
            )
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result

# SPDX-License-Identifier: Apache-2.0
"""Non-GPU (pickle-based) KV cache transfer operations for the MPCacheEngine."""

# Standard
from dataclasses import dataclass
import pickle

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
    IPCCacheEngineKey,
    RegisterNonGpuContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.non_gpu_context import NonGpuContextMetadata
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)

logger = init_logger(__name__)


@dataclass
class NonGPUContextEntry:
    """Registered non-GPU context metadata for a single worker instance.

    Attributes:
        metadata: Layout metadata describing the non-CUDA chunk format.
        model_name: The name of the model associated with this context.
        world_size: The world size associated with this context.
    """

    metadata: NonGpuContextMetadata
    model_name: str
    world_size: int


class NonGPUTransferModule:
    """Handles non-GPU (pickle-based) KV cache transfer operations.

    Owns non-GPU context registrations and provides handlers for
    register, unregister, prepare/commit store, and prepare/commit retrieve
    of CPU-serialized KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheEngineContext) -> None:
        self._ctx = ctx
        self._non_gpu_contexts: dict[int, NonGPUContextEntry] = {}

    @property
    def context(self) -> MPCacheEngineContext:
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
                RequestType.REGISTER_KV_CACHE_NON_GPU_CONTEXT,
                self.register_kv_cache_non_gpu_context,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE,
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

        for instance_id, entry in self._non_gpu_contexts.items():
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
        self._non_gpu_contexts.clear()

    def register_kv_cache_non_gpu_context(
        self,
        payload: RegisterNonGpuContextPayload,
    ) -> None:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla).

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        if payload.instance_id in self._non_gpu_contexts:
            logger.warning(
                "Instance %s's KV cache is already registered, "
                "skipping the new registration",
                payload.instance_id,
            )
            return

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
        metadata = NonGpuContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
        )
        self._non_gpu_contexts[payload.instance_id] = NonGPUContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
        )

        self._ctx.layout_desc_registry.register(
            payload.model_name, payload.world_size, layout_desc
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister a non-GPU KV cache context for the given instance ID.

        Args:
            instance_id: The worker instance identifier.
        """
        entry = self._non_gpu_contexts.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered non-GPU context found for instance ID %d",
                instance_id,
            )
            return

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)
        logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare a store operation. For pickle mode, returns empty slots.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.

        Returns:
            PrepareStoreResponse with empty slots for pickle mode.
        """
        return PrepareStoreResponse(context={})

    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheEngineKey,
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
        obj_keys = self._ctx.resolve_obj_keys(key)

        entry = self._non_gpu_contexts.get(instance_id)
        if entry is None:
            raise ValueError(
                f"non-CUDA context not registered for instance ID {instance_id}"
            )
        ctx = entry.metadata
        chunks: list[torch.Tensor] = pickle.loads(cpu_data)
        reserved_dict = self._ctx.storage_manager.reserve_write(
            obj_keys, ctx.layout_desc, "new"
        )
        written_keys: list[ObjectKey] = []
        try:
            for idx, obj_key in enumerate(obj_keys):
                if obj_key not in reserved_dict:
                    continue
                if idx >= len(chunks):
                    continue
                memory_obj = reserved_dict[obj_key]
                if memory_obj.tensor is None:
                    continue
                chunk_cpu = chunks[idx]
                if chunk_cpu.shape != memory_obj.tensor.shape:
                    continue
                memory_obj.tensor.copy_(chunk_cpu)
                written_keys.append(obj_key)
        finally:
            if written_keys:
                self._ctx.storage_manager.finish_write(written_keys)

        return len(written_keys) == len(reserved_dict)

    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheEngineKey,
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
        obj_keys = self._ctx.resolve_obj_keys(key)

        entry = self._non_gpu_contexts.get(instance_id)
        if entry is None:
            raise ValueError(
                f"non-CUDA context not registered for instance ID {instance_id}"
            )

        prefetched_keys: list[ObjectKey] = []
        try:
            with self._ctx.storage_manager.read_prefetched_results(
                obj_keys
            ) as memory_objs:
                if not memory_objs or len(memory_objs) != len(obj_keys):
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
                prefetched_keys = obj_keys[: len(memory_objs)]
                chunks = []
                for memory_obj in memory_objs:
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
                self._ctx.storage_manager.finish_read_prefetched(prefetched_keys)

    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation. No-op for pickle mode.

        Args:
            key: Cache key (unused for pickle).
            instance_id: Worker instance identifier (unused for pickle).

        Returns:
            Always ``True``.
        """
        return True

# SPDX-License-Identifier: Apache-2.0
"""XPU-based KV cache IPC transfer operations for the MPCacheServer."""

# Standard
import time

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
    XpuIPCWrapper,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.gpu_transfer import (
    ContextEntry,
    downsample_and_stage_block_ids,
    get_layout_desc,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class XpuTransferModule:
    """Handles XPU-based KV cache transfer operations.

    XPU transfer uses Level Zero memory IPC handles for registered KV tensors.
    PyTorch XPU does not currently expose CUDA-like interprocess events, so this
    module uses device synchronization before returning the MQ response.

    Args:
        ctx: The shared server context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._cache_contexts: dict[int, ContextEntry] = {}

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared server context. Exposed for testing only."""
        return self._ctx

    @property
    def cache_contexts(self) -> dict[int, ContextEntry]:
        """Per-instance XPU context registry."""
        return self._cache_contexts

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves."""
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE,
                self.register_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE,
                self.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(RequestType.STORE, self.store, ThreadPoolType.AFFINITY),
            HandlerSpec(RequestType.RETRIEVE, self.retrieve, ThreadPoolType.AFFINITY),
        ]

    def report_status(self) -> dict:
        """Return XPU transfer module status information."""
        registered_xpu_ids: list[int] = []
        cache_context_meta: dict[str, dict] = {}
        for instance_id, entry in self._cache_contexts.items():
            registered_xpu_ids.append(instance_id)
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": entry.cache_context.report_status(),
            }
        return {
            "registered_xpu_ids": registered_xpu_ids,
            "cache_context_meta": cache_context_meta,
        }

    def close(self) -> None:
        """Release XPU resources owned by this module."""
        had_contexts = len(self._cache_contexts) > 0
        for entry in self._cache_contexts.values():
            entry.cache_context.close()
        self._cache_contexts.clear()
        if had_contexts:
            XpuIPCWrapper.clear_opened_ipc_tensors()
            torch_dev.empty_cache()

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register XPU KV cache tensors for a worker instance.

        Args:
            instance_id: XPU worker instance ID.
            kv_caches: KV cache tensor wrappers from the serving engine.
            model_name: Model name associated with the KV cache.
            world_size: KV world size.
            engine_type: Serving engine that produced the KV cache.
            layout_hints: Optional KV layout hints.
            engine_group_infos: Engine-neutral KV cache group metadata.
        """
        if instance_id in self._cache_contexts:
            logger.warning(
                "Instance %s's XPU KV cache is already registered, skipping",
                instance_id,
            )
            return

        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
        )
        self._cache_contexts[instance_id] = ContextEntry(
            cache_context=cache_context,
            model_name=model_name,
            world_size=world_size,
        )
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        self._ctx.layout_desc_registry.register(model_name, world_size, layout_desc)
        logger.info(
            "Registered KV cache for XPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister XPU KV cache tensors for a worker instance."""
        entry = self._cache_contexts.pop(instance_id, None)
        if entry is None:
            logger.warning("No registered XPU context found for instance ID %d", instance_id)
            return
        entry.cache_context.close()
        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)
        XpuIPCWrapper.clear_opened_ipc_tensors()
        torch_dev.empty_cache()
        logger.info("Unregistered KV cache for XPU ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store XPU KV cache blocks into LMCache storage."""
        del event_ipc_handle
        st = time.perf_counter()
        entry = self._cache_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No XPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "XPU STORE block ID underflow for request_id=%s; skipping",
                    key.request_id,
                )
                return b"", False

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(cache_context.device)},
                )
            )
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                )
            )

            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes = 0
            store_succeeded = False
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        obj_keys, layout_desc, "new"
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=1,
                        skip_first_n_tokens=0,
                        direction=lmc_ops.TransferDirection.D2H,
                    )
                store_succeeded = True
            except Exception:
                logger.exception("Cannot store XPU keys due to exception")
                return b"", False
            finally:
                torch_dev.synchronize()
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    self._ctx.storage_manager.finish_write(list(all_dict.keys()))
                else:
                    total_bytes = 0
                self._ctx.event_bus.publish(
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                        },
                    )
                )

        if len(all_dict):
            logger.info(
                "Stored %d XPU tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                time.perf_counter() - st,
            )
        return b"", True

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve LMCache storage chunks into XPU KV cache blocks."""
        del event_ipc_handle
        st = time.perf_counter()
        entry = self._cache_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No XPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                },
            )
        )

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "XPU RETRIEVE block ID underflow for request_id=%s; skipping",
                    key.request_id,
                )
                return b"", False

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            prefetched_keys: list[ObjectKey] = []
            total_bytes = 0
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    with self._ctx.storage_manager.read_prefetched_results(
                        obj_keys
                    ) as memory_objs:
                        if not memory_objs or len(memory_objs) != len(obj_keys):
                            logger.error("Some XPU keys not found during retrieve")
                            return b"", False
                        total_bytes += sum(mo.get_size() for mo in memory_objs)
                        transfer_kv_per_object_group(
                            cache_context,
                            block_ids_per_group_gpu,
                            memory_objs,
                            object_group_id=obj_group_id,
                            batch_size=cache_context.max_batch_size,
                            skip_first_n_tokens=skip_first_n_tokens,
                            direction=lmc_ops.TransferDirection.H2D,
                        )
                        prefetched_keys.extend(obj_keys)
            except Exception:
                logger.exception("Cannot retrieve XPU keys due to exception")
                return b"", False
            finally:
                torch_dev.synchronize()
                if prefetched_keys:
                    self._ctx.storage_manager.finish_read_prefetched(prefetched_keys)
                self._ctx.event_bus.publish(
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                        },
                    )
                )

        logger.info(
            "Retrieved %d XPU tokens in %.3f seconds",
            num_chunks * self._ctx.chunk_size,
            time.perf_counter() - st,
        )
        return b"", True

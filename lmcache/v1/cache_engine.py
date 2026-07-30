# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import gc
import logging
import multiprocessing
import os
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventType
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        if self.enable_controller:
            self.lmcache_worker = LMCacheWorker(config, metadata, self)

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.storage_manager = StorageManager(
            config,
            metadata,
            # self.memory_allocator,
            event_manager=self.event_manager,
            lmcache_worker=self.lmcache_worker,
        )

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        self.use_layerwise = config.use_layerwise
        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> [pinned keys]
        self.lookup_pins: dict[str, list] = defaultdict(list)

        InitializeUsageContext(config.to_original_config(), metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.post_inited = False

        # Whether to force store to wait if no CPU buffer is available
        self.force_store_wait = config.extra_config and config.extra_config.get(
            "force_store_wait", False
        )

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

    def post_init(self, **kwargs) -> None:
        if "async_lookup_server" in kwargs:
            self.async_lookup_server = kwargs["async_lookup_server"]
        if not self.post_inited:
            self.storage_manager.post_init(**kwargs)
            logger.info("Post-initializing LMCacheEngine")
            if self.gpu_connector is not None:
                self.gpu_connector.initialize_kvcaches_ptr(**kwargs)
            self.post_inited = True

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return
        # Drop stale caches when both storage and GPU memory are tight.
        self.storage_manager.maybe_discard_stale_cache(reason="store")

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts = []
        ends = []
        keys = []
        memory_objs = []

        offload_time = 0.0
        put_time = 0.0
        tot_kv_size = 0
        tot_token_num = 0
        t = time.perf_counter()

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        for start, end, key in self.token_database.process_tokens(
            tokens,
            hashes,
            offsets,
            mask,
            request_configs=request_configs,
            skip_last_segment=True,
        ):
            assert isinstance(key, CacheEngineKey)
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype

            # TODO (Jiayi): should be batched in the future
            memory_obj = self.storage_manager.allocate(
                kv_shape,
                kv_dtype,
                busy_loop=self.force_store_wait,
                fmt=self.fmt,
            )
            if memory_obj is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(key)
            memory_objs.append(memory_obj)
            tot_kv_size += memory_obj.get_size()
            tot_token_num += num_tokens

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return
        self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
        offload_time += time.perf_counter() - t

        t = time.perf_counter()

        transfer_spec = kwargs.get("transfer_spec", None)
        self.storage_manager.batched_put(keys, memory_objs, transfer_spec=transfer_spec)
        put_time += time.perf_counter() - t

        tot_time = offload_time + put_time

        logger.info(
            "Stored %d out of total %d tokens. size: %.4f gb, cost %.4f ms, "
            "throughput: %.4f GB/s; offload_time: %.4f ms, put_time: %.4f ms",
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3,
            offload_time * 1000,
            put_time * 1000,
        )

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        # Drop stale caches when both storage and GPU memory are tight.
        self.storage_manager.maybe_discard_stale_cache(reason="store_layer")

        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs,
            skip_last_segment=True,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            if self.storage_manager.contains(keys_multi_layer[0]):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.force_store_wait,
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

        mem_obj_consumer = None
        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )

            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                self.storage_manager.batched_put(keys[layer_id], memory_objs[layer_id])
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        logger.debug(f"Stored {tot_token_num} out of total {len(tokens)} tokens")
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        tot_kv_size = 0
        t = time.perf_counter()

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            if self.async_loading:
                reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
            else:
                reordered_chunks, tot_kv_size = self._process_tokens_internal(
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
        if self.save_only_first_rank:
            with torch.cuda.stream(self.broadcast_stream):
                self._broadcast_or_receive_memory_objs(
                    reordered_chunks,
                    ret_mask,
                )

            # if self.gpu_connector has load_stream, self.broadcast_stream is equals
            # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
            # will execute sequentially within the stream.
            # if self.gpu_connector does not have load_stream, self.broadcast_stream
            # is created by torch.cuda.Stream(), we need to synchronize broadcast
            # operation, and then process to_cpu operation.
            if not hasattr(self.gpu_connector, "load_stream"):
                self.broadcast_stream.synchronize()

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
            self.gpu_connector.batched_to_gpu(
                list(memory_objs), list(starts), list(ends), **kwargs
            )

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if memory_obj is None:
                continue
            if self.remove_after_retrieve and not self._is_passive():
                self.storage_manager.remove(key)
            memory_obj.ref_count_down()

        onload_time = time.perf_counter() - t

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.info(
            "Retrieved %d out of %d required tokens (from %d total tokens)."
            " size: %.4f gb,"
            " cost %.4f ms, throughput: %.4f GB/s;",
            retrieved_tokens,
            num_required_tokens,
            len(tokens),
            tot_kv_size / 1024**3,
            onload_time * 1000,
            tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
        )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        gpu_start_event = None
        gpu_end_event = None
        timing_stream = None
        if torch.cuda.is_available():
            gpu_start_event = torch.cuda.Event(enable_timing=True)
            gpu_end_event = torch.cuda.Event(enable_timing=True)
            timing_stream = (
                self.gpu_connector.load_stream
                if self.gpu_connector is not None
                and hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.current_stream()
            )
            gpu_start_event.record(timing_stream)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            # Require all layers to exist in the same location to avoid
            # lookup/retrieve mismatch and partial-layer retrieval.
            current_location = None
            all_layers_found = True
            for key_single_layer in keys_multi_layer:
                found_location = self.storage_manager.contains(key_single_layer)
                if not found_location:
                    all_layers_found = False
                    break
                if current_location is None:
                    current_location = found_location
                elif current_location != found_location:
                    all_layers_found = False
                    break
            if not all_layers_found or current_location is None:
                break
            if location is None:
                location = current_location
            else:
                # TODO(Jiayi): Support multi-location retrieval in the future
                assert location == current_location, (
                    "All retrieved keys should be from the same location "
                    "when use layerwise retrieval."
                    "Please support multi-location retrieval in the future."
                )

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        total_chunk_tokens = 0
        if starts:
            total_chunk_tokens = sum(
                end - start for start, end in zip(starts, ends, strict=False)
            )
        mem_obj_consumer = None
        if keys:
            # Transpose the keys into layer major format
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

            # Fix A (LMCACHE_ASYNC_LAYER_FETCH=1): use the NON-BLOCKING layerwise
            # loader so layer L+1's fetch is issued to the background async event
            # loop while layer L is still being copied to the GPU. Because the N
            # per-request retrieve_layer generators are stepped in lockstep by the
            # deferred blend driver, their loads are all submitted to that shared
            # loop and overlap across requests too (this is the practical Fix B2 —
            # cross-request concurrency — without a coalesced multi-request call).
            # Falls back to the blocking loader if the async serializer isn't wired
            # (post_init only builds it when use_layerwise/async), so a misconfig
            # degrades silently to today's behaviour rather than crashing.
            _use_async_fetch = (
                os.environ.get("LMCACHE_ASYNC_LAYER_FETCH", "0") == "1"
                and getattr(self.storage_manager, "async_serializer", None) is not None
            )
            if _use_async_fetch:
                get_generator = self.storage_manager.layerwise_batched_get(keys_layer_major)
            else:
                get_generator = self.storage_manager.layerwise_batched_get_sync(keys_layer_major)

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )
            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            to_count_down = []
            # Per-layer load timing costs a BLOCKING layer_end_event.synchronize()
            # on every one of the num_layers sends (see _send_layer below), which
            # stops layer L+1's copy from being enqueued while L is still in
            # flight -- i.e. it serialises the whole fetch phase. Its only consumer
            # is a logger.debug() line that is not emitted at INFO, so keep it OFF
            # by default. Set LMCACHE_LAYER_LOAD_TIMING=1 (or enable DEBUG logging)
            # to restore the old always-sync behaviour, which is what the A/B
            # control run needs.
            per_layer_timing = (
                torch.cuda.is_available()
                and self.gpu_connector is not None
                and hasattr(self.gpu_connector, "load_stream")
                and (
                    logger.isEnabledFor(logging.DEBUG)
                    or os.environ.get("LMCACHE_LAYER_LOAD_TIMING", "0") == "1"
                )
            )
            load_stream = (
                self.gpu_connector.load_stream
                if per_layer_timing
                else torch.cuda.current_stream()
            )

            # Send one layer's KV to the GPU (with optional per-layer timing).
            # Shared by the sync and async-prefetch loops below so the send/timing
            # logic stays identical between them.
            def _send_layer(layer_id, mem_objs_layer):
                if per_layer_timing:
                    layer_start_event = torch.cuda.Event(enable_timing=True)
                    layer_end_event = torch.cuda.Event(enable_timing=True)
                    layer_start_event.record(load_stream)
                    mem_obj_consumer.send(mem_objs_layer)
                    layer_end_event.record(load_stream)
                    layer_end_event.synchronize()
                    layer_elapsed_ms = layer_start_event.elapsed_time(layer_end_event)
                    layer_bytes = 0
                    for mem_obj in mem_objs_layer:
                        if mem_obj is not None and mem_obj.tensor is not None:
                            layer_bytes += (
                                mem_obj.tensor.numel()
                                * mem_obj.tensor.element_size()
                            )
                    layer_gb = layer_bytes / 1024**3
                    layer_gbps = (
                        layer_gb / (layer_elapsed_ms / 1000)
                        if layer_elapsed_ms > 0
                        else 0.0
                    )
                    logger.debug(
                        "Layer %d load-to-gpu cost %.3f ms for %d tokens, "
                        "%.4f GB, %.2f GB/s",
                        layer_id,
                        layer_elapsed_ms,
                        total_chunk_tokens,
                        layer_gb,
                        layer_gbps,
                    )
                else:
                    mem_obj_consumer.send(mem_objs_layer)

            if _use_async_fetch:
                # Fix A: one-ahead prefetch. `layerwise_batched_get` yields a
                # concurrent.futures.Future per layer (the load runs on the shared
                # background event loop). Hold the NEXT layer's in-flight future
                # while resolving+sending the CURRENT layer, so the fetch of L+1
                # overlaps the GPU copy of L. next()-count and yield-count are
                # IDENTICAL to the sync loop (num_layers each) so wait_for_layer_load
                # cadence is unchanged.
                pending = next(get_generator)
                for layer_id in range(self.num_layers):
                    cur = pending
                    assert cur is not None
                    # issue the next layer's fetch before we block on the current
                    pending = (
                        next(get_generator)
                        if layer_id < self.num_layers - 1
                        else None
                    )
                    if layer_id == 0:
                        yield torch.sum(ret_mask)
                    else:
                        yield None
                    mem_objs_layer = cur.result()  # Future -> resolved mem_objs
                    _send_layer(layer_id, mem_objs_layer)
                    to_count_down.extend(mem_objs_layer)
            else:
                for layer_id in range(self.num_layers):
                    task = next(get_generator)

                    assert task is not None

                    if layer_id == 0:
                        # NOTE(Yuwei): For sglang integration we need to provide retrieved
                        # tokens number in the first layer loading since there is no lookup
                        yield torch.sum(ret_mask)
                    else:
                        yield None

                    mem_objs_layer = task
                    _send_layer(layer_id, mem_objs_layer)
                    to_count_down.extend(mem_objs_layer)

            for mem_obj in to_count_down:
                if mem_obj is not None:
                    mem_obj.ref_count_down()
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

        # synchronize the last layer
        if mem_obj_consumer is not None:
            next(mem_obj_consumer)

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        elapsed_ms = None
        if gpu_start_event is not None and gpu_end_event is not None:
            assert timing_stream is not None
            gpu_end_event.record(timing_stream)
            gpu_end_event.synchronize()
            elapsed_ms = gpu_start_event.elapsed_time(gpu_end_event)
        logger.info(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens. "
            f"retrieve_layer gpu cost {elapsed_ms:.3f} ms"
        )

        yield ret_mask

    # ------------------------------------------------------------------
    # B1: coalesced multi-request layerwise fetch (LMCACHE_COALESCED_FETCH=1,
    # default OFF). Correctness gated (jobs 15005761 + 15005982); measured
    # no latency benefit at N=8 (see BLEND_OPT_IMPLEMENTATION.md).
    # ------------------------------------------------------------------
    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer_multi(
        self,
        tokens_list: List[Union[torch.Tensor, list]],
        masks_list: List[Optional[torch.Tensor]],
        slot_mappings: List[torch.Tensor],
        **kwargs,
    ) -> Generator[Optional[List[torch.Tensor]], None, None]:
        """
        Multi-request counterpart of `retrieve_layer`. Gathers every request's
        keys and issues ONE `layerwise_batched_get_sync` call, driving ONE
        `batched_to_gpu_multi` connector generator instead of N per-request
        `batched_to_gpu` generators -- collapsing `fetch_syncs = num_layers *
        N` to `num_layers` (see `gpu_connector.py` `_coalesced_layout` /
        `batched_to_gpu_multi`).

        `tokens_list[r]` / `masks_list[r]` / `slot_mappings[r]` are request
        r's own tokens / mask / slot mapping -- exactly what a lone
        `retrieve_layer` call would receive as `tokens` / `mask` /
        `kwargs["slot_mapping"]`. `kwargs` (e.g. `kvcaches`) is shared across
        all requests. Every request must contribute at least one retrievable
        chunk -- callers (the adapter) must apply the existing per-request
        gates (e.g. `_bail` in `_batched_blend_load_kv`) BEFORE calling this,
        exactly as they do today before the per-request `retrieve_layer` loop;
        this generator has no per-request fallback path once it starts.

        Yields `num_layers + 2` times total (NOT once per request) -- the
        whole point is replacing N generators that each yield `num_layers + 2`
        times with ONE generator yielding `num_layers + 2` times overall. The
        first layer's yield carries a list of per-request retrieved-token
        counts (mirroring `retrieve_layer`'s single count); the final yield
        carries a list of per-request `ret_mask` tensors, one per request in
        `tokens_list` order -- what N separate `retrieve_layer` calls would
        each have produced.

        Only wraps `layerwise_batched_get_sync` (Fix A's async loader was
        found to be a dead end -- see BLEND_OPT_IMPLEMENTATION.md -- so B1
        does not carry that complexity forward).
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer_multi operation"
        )
        assert isinstance(self.gpu_connector, VLLMBufferLayerwiseGPUConnector), (
            "retrieve_layer_multi requires VLLMBufferLayerwiseGPUConnector "
            "(the coalesced buffer layout is specific to it)."
        )
        n_reqs = len(tokens_list)
        assert n_reqs > 0, "retrieve_layer_multi requires at least one request."
        assert n_reqs == len(masks_list) == len(slot_mappings), (
            "retrieve_layer_multi: tokens_list/masks_list/slot_mappings must "
            "be parallel lists, one entry per request."
        )

        gpu_start_event = None
        gpu_end_event = None
        timing_stream = None
        if torch.cuda.is_available():
            gpu_start_event = torch.cuda.Event(enable_timing=True)
            gpu_end_event = torch.cuda.Event(enable_timing=True)
            timing_stream = self.gpu_connector.load_stream
            gpu_start_event.record(timing_stream)

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        ret_masks = [
            torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
            for tokens in tokens_list
        ]

        req_starts: List[List[int]] = []
        req_ends: List[List[int]] = []
        keys_layer_major_per_request: List[List[List[CacheEngineKey]]] = []
        num_required_tokens_total = 0

        for r in range(n_reqs):
            tokens = tokens_list[r]
            mask = masks_list[r]
            if mask is not None:
                num_required_tokens_total += torch.sum(mask).item()
            else:
                num_required_tokens_total += len(tokens)

            starts: List[int] = []
            ends: List[int] = []
            keys: List[List[CacheEngineKey]] = []
            location = None
            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)

                keys_multi_layer = key.split_layers(self.num_layers)

                # Same "all layers same location" invariant as retrieve_layer.
                current_location = None
                all_layers_found = True
                for key_single_layer in keys_multi_layer:
                    found_location = self.storage_manager.contains(key_single_layer)
                    if not found_location:
                        all_layers_found = False
                        break
                    if current_location is None:
                        current_location = found_location
                    elif current_location != found_location:
                        all_layers_found = False
                        break
                if not all_layers_found or current_location is None:
                    break
                if location is None:
                    location = current_location
                else:
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )

                starts.append(start)
                ends.append(end)
                keys.append(keys_multi_layer)

                ret_masks[r][start:end] = True

            assert starts, (
                f"retrieve_layer_multi: request {r} contributed zero "
                "retrievable chunks -- caller must filter these out (via the "
                "existing per-request gates) before coalescing."
            )

            req_starts.append(starts)
            req_ends.append(ends)
            keys_layer_major_per_request.append(
                [list(row) for row in zip(*keys, strict=False)]
            )

        monitor_req_id = self.stats_monitor.on_retrieve_request(
            num_required_tokens_total
        )

        # Flatten per-request per-layer keys into one list per layer, in
        # EXACTLY the order `_coalesced_layout` uses for `chunk_slices`
        # (request-major, then chunk order within a request) --
        # `layerwise_batched_get_sync` returns one memory object per key, in
        # the same order as the keys given for that layer.
        keys_layer_major_all: List[List[CacheEngineKey]] = [
            [
                key
                for per_request in keys_layer_major_per_request
                for key in per_request[layer_id]
            ]
            for layer_id in range(self.num_layers)
        ]

        get_generator = self.storage_manager.layerwise_batched_get_sync(
            keys_layer_major_all
        )

        mem_obj_consumer = self.gpu_connector.batched_to_gpu_multi(
            req_starts, req_ends, slot_mappings, **kwargs
        )
        next(mem_obj_consumer)

        # Same D-controlled per-layer timing sync as retrieve_layer, kept
        # symmetric so a coalesced-vs-baseline comparison isn't confounded by
        # a NEW timing asymmetry (this project has been bitten by exactly that
        # shape of bug three times already -- see BLEND_OPT_IMPLEMENTATION.md).
        per_layer_timing = (
            torch.cuda.is_available()
            and hasattr(self.gpu_connector, "load_stream")
            and (
                logger.isEnabledFor(logging.DEBUG)
                or os.environ.get("LMCACHE_LAYER_LOAD_TIMING", "0") == "1"
            )
        )
        load_stream = (
            self.gpu_connector.load_stream
            if per_layer_timing
            else torch.cuda.current_stream()
        )

        to_count_down = []
        for layer_id in range(self.num_layers):
            task = next(get_generator)
            assert task is not None

            if layer_id == 0:
                yield [torch.sum(m) for m in ret_masks]
            else:
                yield None

            mem_objs_layer = task
            if per_layer_timing:
                layer_start_event = torch.cuda.Event(enable_timing=True)
                layer_end_event = torch.cuda.Event(enable_timing=True)
                layer_start_event.record(load_stream)
                mem_obj_consumer.send(mem_objs_layer)
                layer_end_event.record(load_stream)
                layer_end_event.synchronize()
                layer_elapsed_ms = layer_start_event.elapsed_time(layer_end_event)
                logger.debug(
                    "Layer %d load-to-gpu (coalesced, %d reqs) cost %.3f ms",
                    layer_id, n_reqs, layer_elapsed_ms,
                )
            else:
                mem_obj_consumer.send(mem_objs_layer)
            to_count_down.extend(mem_objs_layer)

        for mem_obj in to_count_down:
            if mem_obj is not None:
                mem_obj.ref_count_down()

        # Per-request gap positions, sliced back out of the coalesced tensor
        # into each request's own local coordinates -- mirrors what a lone
        # `retrieve_layer` call would leave on `current_gap_positions` for
        # ITS request. `_compute_hit_indices` (blender.py) reads that
        # attribute per request, so the adapter swaps this list in, one
        # entry at a time, before selecting each request. Recomputing
        # `_coalesced_layout` here is cheap (pure index arithmetic, no CUDA)
        # and avoids plumbing `segments` out of `batched_to_gpu_multi` itself.
        _, segments, _, _ = VLLMBufferLayerwiseGPUConnector._coalesced_layout(
            req_starts, req_ends
        )
        coalesced_gaps = self.gpu_connector.current_gap_positions
        per_request_gaps = []
        for buf_off, _src_start, n_r in segments:
            local = coalesced_gaps[
                (coalesced_gaps >= buf_off) & (coalesced_gaps < buf_off + n_r)
            ] - buf_off
            per_request_gaps.append(local)
        self.gpu_connector.current_gap_positions_per_request = per_request_gaps

        yield None

        # synchronize the last layer
        next(mem_obj_consumer)

        total_retrieved = sum(int(torch.sum(m)) for m in ret_masks)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, total_retrieved)
        elapsed_ms = None
        if gpu_start_event is not None and gpu_end_event is not None:
            assert timing_stream is not None
            gpu_end_event.record(timing_stream)
            gpu_end_event.synchronize()
            elapsed_ms = gpu_start_event.elapsed_time(gpu_end_event)
        logger.info(
            f"retrieve_layer_multi: retrieved {total_retrieved} "
            f"out of {num_required_tokens_total} tokens across {n_reqs} "
            f"requests. gpu cost {elapsed_ms:.3f} ms"
        )

        yield ret_masks

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens are cached.
        """

        if tokens is not None:
            lookup_request_id = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_request_id = self.stats_monitor.on_lookup_request(sum(offsets))

        res = 0
        total_hit_tokens = 0
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                for start, end, key in chunk_info_iterator:
                    if start == 0: continue
                    # logger.info(f"Looking up start={start}, end={end}, key={key}")
                    assert isinstance(key, CacheEngineKey)

                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = key.split_layers(self.num_layers)

                    all_layers_found = True
                    for key_single_layer in key_all_layers:
                        # logger.info(f"Looking up single layer key={key_single_layer}, search_range={search_range}, pin={pin}")
                        if not self.storage_manager.contains(
                            key_single_layer, search_range, pin
                        ):
                            all_layers_found = False
                            break
                    if all_layers_found:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            self.lookup_pins[lookup_id].extend(  # type: ignore
                                key_all_layers
                            )
                        res = end
                        total_hit_tokens += end - start
                        logger.debug(f"Hit: start={start}, end={end}, key={key}")
                        continue
                    logger.debug(f"total_hit_tokens to {total_hit_tokens}, res is {res}")
                    return res
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    chunk_info_list.append(chunk_info)
                    keys.append(chunk_info[2])

                batched_contains_res = self.storage_manager.batched_contains(
                    keys, search_range, pin, True
                )
                assert len(batched_contains_res) == len(chunk_info_list)
                for (start, end, key), exists in zip(
                    chunk_info_list, batched_contains_res, strict=False
                ):
                    if exists:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            self.lookup_pins[lookup_id].append(key)
                        res = end
                        total_hit_tokens += end - start
                        logger.debug(f"Hit: start={start}, end={end}, key={key}")
                        continue
                    logger.debug(f"Res: {res}, updating total_hit_tokens to {total_hit_tokens}")
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_request_id, res)
            # vllm lookup sets pin to True
            if pin:
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """

        num_tokens = self.lookup(
            tokens,
            search_range=old_position,
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        keys = self.lookup_pins[event_id]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert memory_objs is not None, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        keys = self.lookup_pins[event_id]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        keys = self.lookup_pins[event_id]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert compressed_memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            self.storage_manager.batched_unpin(self.lookup_pins[lookup_id])
            del self.lookup_pins[lookup_id]

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""

        if self.lmcache_worker is not None:
            self.lmcache_worker.close()

        self.storage_manager.close()

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.pop_event(EventType.LOADING, kwargs["req_id"])

        memory_objs = future.result()
        memory_objs = [mm for m in memory_objs for mm in m]

        # NOTE(Jiayi): here we assume the retrieved memory_objs have
        # the same order as the lookup order.
        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_indices = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            idx = start // self.config.chunk_size
            memory_obj = memory_objs[idx]
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_indices.add(idx)

        for idx, unused_mem_obj in enumerate(memory_objs):
            if idx not in used_indices and unused_mem_obj is not None:
                unused_mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """

        tot_kv_size = 0
        # location -> [(CacheEngineKey, start, end)]
        block_mapping: dict[str, list[tuple[CacheEngineKey, int, int]]] = defaultdict(
            list
        )

        reordered_chunks: List[ProcessedChunk] = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        # In some scenarios, lookup is called first, and then the original tokens
        # is sliced based on the lookup result. In these scenarios, the tokens
        # passed in must exist in LMCache, and we can set skip_contains_check to True.
        # When skip_contains_check is True and there is only one backend, the `contains`
        # call can be skipped.
        skip_contains_check = (
            kwargs["skip_contains_check"] if "skip_contains_check" in kwargs else False
        )
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            location = None
            if key in self.lookup_cache:
                # TODO(Jiayi): we can reduce the number of `contains` calls
                # by checking the lookup cache first (should be updated in `lookup`)
                pass
            else:
                # NOTE: key should always be in the lookup cache once we support it.
                # TODO: use lookup_cache to skip the contains
                if (
                    skip_contains_check
                    and len(self.storage_manager.non_allocator_backends) == 1
                ):
                    location = self.storage_manager.non_allocator_backends[0]
                else:
                    location = self.storage_manager.contains(key)
                if location is None:
                    break

                # NOTE: Here we make the assumption that the underlying
                # storage backend support pin operation, and the memory
                # object is already pinned in the storage backend.
                ret_mask[start:end] = True

            assert location is not None

            block_mapping[location].append((key, start, end))

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )
            assert memory_objs is not None, (
                "Failed to get memory objects from storage backend"
            )

            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                tensor_to_broadcast = memory_obj.tensor.to(
                    f"cuda:{self.metadata.worker_id}"
                )
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                tensor = torch.empty(
                    metadata.shape,
                    dtype=metadata.dtype,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                torch.Size(metadata.kv_shape),
                metadata.kv_dtype,
                MemoryFormat.KV_2LTD,
            )

        if config.weka_path is not None or config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            stat_logger.shutdown()
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
            cls._stat_loggers.pop(instance_id, None)
            LMCStatsMonitor.DestroyInstance()

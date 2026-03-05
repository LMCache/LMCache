# SPDX-License-Identifier: Apache-2.0
# Standard
import itertools
import os
import time

# Third Party
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import RangePatternMatcher
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_keys_to_object_keys,
)
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.config import (
    PrometheusConfig,
    parse_args_to_prometheus_config,
)
from lmcache.v1.mp_observability.prometheus_controller import (
    get_prometheus_controller,
    init_prometheus_controller,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.gpu_context import (
    PlainGPUCacheContext,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.server import MPCacheEngine, parse_args

logger = init_logger(__name__)


# Helper functions
def create_ipc_key_with_hashes(
    key: IPCCacheEngineKey, hashes: list[bytes]
) -> list[IPCCacheEngineKey]:
    """
    Create the list of IPCCacheEngineKey with specific hash.
    """
    ipc_keys: list[IPCCacheEngineKey] = []
    for hash_bytes in hashes:
        ipc_keys.append(
            IPCCacheEngineKey(
                model_name=key.model_name,
                world_size=key.world_size,
                worker_id=key.worker_id,
                token_ids=key.token_ids,
                start=key.start,
                end=key.end,
                request_id=key.request_id,
                chunk_hash=hash_bytes,
            )
        )

    return ipc_keys


def create_temp_ipc_key_by_range(
    key: IPCCacheEngineKey,
    start: int,
    end: int,
) -> IPCCacheEngineKey:
    """
    Create a temporary IPCCacheEngineKey for the specific token range. This is used
    for the lookup of each paragraph when doing the separate lookup for blend engine.

    Note that this function is only used for lookup, and the hash value in the key is
    not used and can be set to empty bytes.
    """
    return IPCCacheEngineKey(
        model_name=key.model_name,
        world_size=key.world_size,
        worker_id=key.worker_id,
        token_ids=key.token_ids[start:end],
        start=start,
        end=end,
        request_id=key.request_id,
        chunk_hash=bytes(),
    )


# Main class and main functions
class BlendEngine(MPCacheEngine):
    BLEND_HASH_PREFIX = 0xB1ED

    def __init__(
        self,
        sep_tokens: tuple[list[int], list[int]],
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
    ):
        super().__init__(storage_manager_config, chunk_size, hash_algorithm="blake3")

        self._cb_gpu_contexts: dict[int, PlainGPUCacheContext] = {}

        # CB GPU ID -> (model name, world size) as metadata
        # NOTE: This is mainly for determining the layout desc during prefetch
        self._cb_gpu_context_meta: dict[int, tuple[str, int]] = {}

        # self._sep_token_len = len(sep_tokens)
        # self._token_matcher = ParallelPatternMatcher(sep_tokens)
        self._token_matcher = RangePatternMatcher(sep_tokens[0], sep_tokens[1])

    def cb_register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
    ) -> None:
        """
        Register the KV cache buffer from the blend engine

        Args:
            instance_id: Unique identifier for the blend engine instance
            kv_caches: KVCache object containing the GPU buffer pointers
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
        """
        gpu_context = PlainGPUCacheContext(kv_caches, self.chunk_size)
        self._cb_gpu_contexts[instance_id] = gpu_context
        self._cb_gpu_context_meta[instance_id] = (model_name, world_size)
        logger.info(
            "Registered CB KV cache for instance_id %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def cb_unregister_kv_cache(self, instance_id: int) -> None:
        """
        Unregister the KV cache buffer for the given instance_id

        Args:
            instance_id: Unique identifier for the blend engine instance to unregister
        """
        if instance_id in self._cb_gpu_contexts:
            del self._cb_gpu_contexts[instance_id]
            del self._cb_gpu_context_meta[instance_id]
            logger.info("Unregistered CB KV cache for instance_id %d", instance_id)
        else:
            logger.warning(
                "Attempted to unregister non-existent CB KV cache for instance_id %d",
                instance_id,
            )

    def cb_lookup_pre_computed(self, key: IPCCacheEngineKey) -> list[tuple[int, int]]:
        """
        Lookup the pre-computed chunks in the underly storage that was stored as
        pre-computed.

        The tokens will be split to paragraphs by the sep_tokens. Then, we do
        a lookup for each paragraph in the storage, and return the match ranges for
        the pre-computed chunks.

        Args:
            key: IPCCacheEngineKey containing the token ids to lookup

        Returns:
            List of tuples (start, end) indicating the match ranges for the
            pre-computed token ranges
        """
        # Match and split the token ids into paragraphs
        prefetch_handles: list[PrefetchHandle] = []
        expected_found_count: list[int] = []
        found_ranges: list[tuple[int, int]] = []
        ranges = self._separate_tokens_by_pattern(key.token_ids)
        model_name, world_size = key.model_name, key.world_size

        # Find the cb gpu context and calculate the layout desc
        layout_desc: MemoryLayoutDesc | None = None
        for gpu_id, (m_name, w_size) in self._cb_gpu_context_meta.items():
            if m_name == model_name and w_size == world_size:
                cb_ctx = self._cb_gpu_contexts[gpu_id]
                layout_desc = MemoryLayoutDesc(
                    shapes=[cb_ctx.get_kv_buffer_shape(self.chunk_size)],
                    dtypes=[cb_ctx.dtype],
                )
                break

        if layout_desc is None:
            logger.error(
                "No CB GPU context found for model %s with world size %d "
                "during cb_lookup_pre_computed!",
                model_name,
                world_size,
            )
            return []

        # Submit Lookup for each paragraph
        for start, end in ranges:
            temp_ipc_key = create_temp_ipc_key_by_range(key, start, end)
            ipc_keys = temp_ipc_key.to_hash_keys(
                hasher=self.token_hasher,
                full_chunk_only=True,
                prefix_hash=self.BLEND_HASH_PREFIX,
            )

            obj_keys = ipc_keys_to_object_keys(ipc_keys)
            handle = self.storage_manager.submit_prefetch_task(obj_keys, layout_desc)

            prefetch_handles.append(handle)
            expected_found_count.append(len(ipc_keys))

            logger.debug(
                "DEBUG: Submitted prefetch for obj keys %s for range (%d, %d), ",
                obj_keys,
                start,
                end,
            )

        # Query the prefetch handle
        for handle, exp_count, (start, end) in zip(
            prefetch_handles, expected_found_count, ranges, strict=False
        ):
            found_count = None
            while True:
                found_count = self.storage_manager.query_prefetch_status(handle)
                if found_count is not None:
                    break

            # Real found count after dedup the TP
            found_count = found_count // world_size

            # All found or not
            # if found_count == exp_count:
            #    found_ranges.append((start, end))
            #    logger.debug(
            #        "Found all pre-computed chunks for paragraph with range (%d, %d)",
            #        start,
            #        end,
            #    )
            # elif found_count > 0:
            if found_count > 0:
                found_ranges.append((start, start + found_count * self.chunk_size))
                logger.debug(
                    "Partially found pre-computed chunks for paragraph with range "
                    "(%d, %d), found chunk count: %d, real range (%d, %d)",
                    start,
                    end,
                    found_count,
                    start,
                    start + found_count * self.chunk_size,
                )
            else:
                logger.debug(
                    "No pre-computed chunks found for paragraph with range (%d, %d)",
                    start,
                    end,
                )

        return found_ranges

    def _cb_store_gpu_copy(
        self,
        obj_keys: list[ObjectKey],
        gpu_context: PlainGPUCacheContext,
        offset: int,
        event_ipc_handle: bytes,
    ) -> tuple[torch.cuda.Event, dict]:
        """
        Helper function to perform GPU-to-CPU copy operations for storing chunks.

        Args:
            obj_keys: List of object keys to store.
            gpu_context: GPU context for the blend engine instance.
            offset: The starting offset in the CB KV cache buffer.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            A tuple of (event, reserved_dict) where event is the CUDA event and
            reserved_dict is the dictionary of reserved memory objects.
        """
        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)

            # Wait for vLLM event to finish
            vllm_event = torch.cuda.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            # Prepare for the copy
            num_tokens = self.chunk_size
            cpu_shape = gpu_context.get_kv_buffer_shape(num_tokens)
            layout_desc = MemoryLayoutDesc(
                shapes=[cpu_shape], dtypes=[gpu_context.dtype]
            )

            reserved_dict = self.storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )

            for idx, obj_key in enumerate(obj_keys):
                if obj_key in reserved_dict:
                    memory_obj = reserved_dict[obj_key]
                else:
                    continue

                offset_start = idx * self.chunk_size + offset
                offset_end = offset_start + self.chunk_size

                # Copy from GPU to CPU
                tmp_buffer = gpu_context.get_tmp_gpu_buffer(offset_end - offset_start)
                gpu_kv_slice = gpu_context.slice_kv_cache_on_tokens(
                    offset_start, offset_end
                )
                with self.lock:
                    tmp_buffer.copy_(gpu_kv_slice, non_blocking=True)
                    lmcache_memcpy_async_d2h(tmp_buffer, memory_obj)

                event.record()

        # Call finish_write after the copy is done
        gpu_context.cupy_stream.launch_host_func(
            self.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )

        return event, reserved_dict

    def cb_store_pre_computed(
        self,
        key: IPCCacheEngineKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Store the pre-computed chunks in the underlying storage for later retrieval.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the pre-computed
                chunks are stored.
            offset: The starting offset in the CB KV cache buffer where the
                pre-computed
            instance_id: The instance_id of the blend engine instance to store the
                pre-computed chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of storing the
            pre-computed chunks, and a boolean flag indicating if the store is
            successful.

        Note:
            The input tokens should not have any separator in it. It should just be
            one "paragraph".
            This function will discard the last partial chunk and only store the full
            chunks
        """
        # Compute blend-only hash for the keys
        hashed_ipc_keys = key.to_hash_keys(
            hasher=self.token_hasher,
            full_chunk_only=True,
            prefix_hash=self.BLEND_HASH_PREFIX,
        )
        # convert to object key
        obj_keys = ipc_keys_to_object_keys(hashed_ipc_keys)

        assert instance_id in self._cb_gpu_contexts, (
            f"Instance ID {instance_id} not registered for CB KV cache"
        )
        gpu_context = self._cb_gpu_contexts[instance_id]

        event, reserved_dict = self._cb_store_gpu_copy(
            obj_keys, gpu_context, offset, event_ipc_handle
        )

        logger.info(
            "Stored pre-computed doc with %d tokens, num stored chunks: %d",
            key.end - key.start,
            len(reserved_dict),
        )
        return event.ipc_handle(), True

    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheEngineKey,
        ranges: list[tuple[int, int]],
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Retrieve the pre-computed chunks from the underlying storage and copy them to
        the CB KV cache buffer.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the pre-computed
                chunks are retrieved.
            ranges: List of tuples (start, end) indicating the match ranges for the
                pre-computed chunks to retrieve.
            offset: The starting offset in the CB KV cache buffer to copy the retrieved
                chunks to.
            instance_id: The instance_id of the blend engine instance to retrieve the
                pre-computed chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of retrieving the
            pre-computed chunks, and a boolean flag indicating if the retrieval is
            successful.

        Note:
            We must call `cb_lookup_pre_computed` first before calling this function
        """
        obj_keys_for_paragraphs: list[list[ObjectKey]] = []

        assert instance_id in self._cb_gpu_contexts, (
            f"Instance ID {instance_id} not registered for CB KV cache"
        )
        gpu_context = self._cb_gpu_contexts[instance_id]

        # We already have the token range, so can directly start from the obj keys
        for start, end in ranges:
            temp_ipc_key = create_temp_ipc_key_by_range(key, start, end)
            hash_ipc_keys = temp_ipc_key.to_hash_keys(
                hasher=self.token_hasher,
                full_chunk_only=True,
                prefix_hash=self.BLEND_HASH_PREFIX,
            )
            obj_keys = ipc_keys_to_object_keys(hash_ipc_keys)
            obj_keys_for_paragraphs.append(obj_keys)

        logger.debug("DEBUG object keys to retrieve: %s", obj_keys_for_paragraphs)

        # Now, do the real retrieval job
        def _retrieve_one_paragraph(
            obj_keys: list[ObjectKey],
            memory_objs: list[MemoryObj],
            gpu_offset: int,
        ):
            for idx, (key, memory_obj) in enumerate(
                zip(obj_keys, memory_objs, strict=False)
            ):
                offset_start = gpu_offset + idx * self.chunk_size
                offset_end = offset_start + self.chunk_size

                # Copy from CPU to GPU
                tmp_buffer = gpu_context.get_tmp_gpu_buffer(offset_end - offset_start)
                target_buffer = gpu_context.slice_kv_cache_on_tokens(
                    offset_start, offset_end
                )

                with self.lock:
                    lmcache_memcpy_async_h2d(memory_obj, tmp_buffer)
                    target_buffer.copy_(
                        tmp_buffer,
                        non_blocking=True,
                    )

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)
            retrieved_paragraph_objects: list[list[MemoryObj]] = []

            try:
                # Populate the objects first
                for obj_keys in obj_keys_for_paragraphs:
                    with self.storage_manager.read_prefetched_results(
                        obj_keys
                    ) as memory_objs:
                        if memory_objs is None:
                            logger.error("Some keys not found during CB retrieve!")
                            return event.ipc_handle(), False
                        retrieved_paragraph_objects.append(memory_objs)

                # Then do to-gpu
                for obj_keys, memory_objs, (start, end) in zip(
                    obj_keys_for_paragraphs,
                    retrieved_paragraph_objects,
                    ranges,
                    strict=False,
                ):
                    gpu_offset = start + offset
                    _retrieve_one_paragraph(obj_keys, memory_objs, gpu_offset)

            except Exception as e:
                logger.error("Error during retrieving prefetched results: %s", e)
                return event.ipc_handle(), False

            finally:
                event.record()
                # TODO: here we simply "unlock" all the keys, which may cause
                # double-unlock if error happens during read_prefetched_results.
                # We should consider not unlocking objects in read_prefetched_results
                # if error happens.
                all_keys = list(itertools.chain.from_iterable(obj_keys_for_paragraphs))
                gpu_context.cupy_stream.launch_host_func(
                    self.storage_manager.finish_read_prefetched, all_keys
                )

        logger.info(
            "Retrieved pre-computed with ranges %s to GPU offset starting at %d",
            ranges,
            offset,
        )
        return event.ipc_handle(), True

    def cb_store_final(
        self,
        key: IPCCacheEngineKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Store the final chunks in the underlying storage after processing. The stored
        chunk should be accessible for normal mode LLMs.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the final chunks
                are stored.
            offset: The starting offset in the CB KV cache buffer where the final
                chunks are stored.
            instance_id: The instance_id of the blend engine instance to store the final
                chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of storing the
            final chunks, and a boolean flag indicating if the store is successful.
        """
        # Compute normal hash for the keys
        hashed_ipc_keys = key.to_hash_keys(
            hasher=self.token_hasher,
            full_chunk_only=True,
            prefix_hash=None,
        )

        # convert to object key
        obj_keys = ipc_keys_to_object_keys(hashed_ipc_keys)

        # Get GPU context
        assert instance_id in self._cb_gpu_contexts, (
            f"Instance ID {instance_id} not registered for CB KV cache"
        )
        gpu_context = self._cb_gpu_contexts[instance_id]

        event, reserved_dict = self._cb_store_gpu_copy(
            obj_keys, gpu_context, offset, event_ipc_handle
        )

        logger.info(
            "Stored final doc with %d tokens, num stored chunks: %d",
            key.end - key.start,
            len(reserved_dict),
        )
        return event.ipc_handle(), True

    # Helper functions
    def _separate_tokens_by_pattern(
        self, token_ids: tuple[int, ...]
    ) -> list[tuple[int, int]]:
        """
        Separate the input token ids into paragraphs based on the separator tokens.

        Args:
            token_ids: List of input token ids to separate

        Returns:
            List of tuples (start, end) indicating the start and end indices of each
            paragraph in the input token ids.
        """
        matches = self._token_matcher.match(list(token_ids))
        logger.debug(
            "Separated tokens into %d paragraphs with ranges: %s", len(matches), matches
        )
        return matches


def get_sep_tokens() -> tuple[list[int], list[int]]:
    """
    Get the separator tokens used for splitting input sequences into paragraphs.

    Returns:
        The start pattern and the end pattern in token ids for separating paragraphs.

    Environment variables:
    - `LMCACHE_BLEND_MODEL_NAME`: the model name to load the tokenizer, default
        is "openai/gpt-oss-120b"
    """
    model_name = os.getenv("LMCACHE_BLEND_MODEL_NAME", "openai/gpt-oss-120b")
    start_end_family = {
        "openai/gpt-oss-20b": ([200006], [200007]),
        "openai/gpt-oss-120b": ([200006], [200007]),
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1": ([128006], [128009]),
    }
    if model_name not in start_end_family:
        logger.error(
            "Model name %s not recognized for blend engine. Supported models: %s",
            model_name,
            list(start_end_family.keys()),
        )
        raise ValueError(
            f"Model name {model_name} not recognized for blend engine. "
            f"Supported models: {list(start_end_family.keys())}"
        )

    return start_end_family[model_name]


def add_handler_helper(
    server: MessageQueueServer, request_type: RequestType, handler_function
):
    payload_classes = get_payload_classes(request_type)
    handler_type = get_handler_type(request_type)
    server.add_handler(
        request_type,
        payload_classes,
        handler_type,
        handler_function,
    )


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    prometheus_config: PrometheusConfig,
    return_engine: bool = False,
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        prometheus_config: Configuration for the Prometheus observability stack
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheEngine)
        If return_engine is False: None (blocks until interrupted)
    """
    # Initialize global prometheus controller
    init_prometheus_controller(prometheus_config)

    sep_tokens = get_sep_tokens()

    # Initialize the engine (loggers self-register with the global controller)
    engine = BlendEngine(
        sep_tokens=sep_tokens,
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
    )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=context,
        max_workers=mp_config.max_workers,
    )

    # Add handlers for original server
    add_handler_helper(server, RequestType.REGISTER_KV_CACHE, engine.register_kv_cache)
    add_handler_helper(
        server, RequestType.UNREGISTER_KV_CACHE, engine.unregister_kv_cache
    )
    add_handler_helper(server, RequestType.STORE, engine.store)
    add_handler_helper(server, RequestType.LOOKUP, engine.lookup)
    add_handler_helper(server, RequestType.RETRIEVE, engine.retrieve)
    add_handler_helper(server, RequestType.CLEAR, engine.clear)
    add_handler_helper(server, RequestType.GET_CHUNK_SIZE, engine.get_chunk_size)
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)

    # Add handler for blend operations
    add_handler_helper(
        server, RequestType.CB_REGISTER_KV_CACHE, engine.cb_register_kv_cache
    )
    add_handler_helper(
        server, RequestType.CB_UNREGISTER_KV_CACHE, engine.cb_unregister_kv_cache
    )
    add_handler_helper(
        server, RequestType.CB_LOOKUP_PRE_COMPUTED, engine.cb_lookup_pre_computed
    )
    add_handler_helper(
        server, RequestType.CB_STORE_PRE_COMPUTED, engine.cb_store_pre_computed
    )
    add_handler_helper(
        server, RequestType.CB_RETRIEVE_PRE_COMPUTED, engine.cb_retrieve_pre_computed
    )
    add_handler_helper(server, RequestType.CB_STORE_FINAL, engine.cb_store_final)

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )
    # Start the ZMQ server
    torch.cuda.init()
    server.start()

    # Start prometheus controller after engine creation (loggers are registered)
    get_prometheus_controller().start()
    logger.info("LMCache cache blend server is running...")

    # Return server and engine if requested (for HTTP server integration)
    if return_engine:
        return server, engine

    # Dummy loop to keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        get_prometheus_controller().stop()
        server.close()
        engine.close()


if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    prometheus_config = parse_args_to_prometheus_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        prometheus_config=prometheus_config,
    )

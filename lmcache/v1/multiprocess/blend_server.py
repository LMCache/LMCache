# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import time

# Third Party
from transformers import AutoTokenizer
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ipc_keys_to_object_keys,
)
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    parse_args_to_config,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
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


class BlendEngine(MPCacheEngine):
    BLEND_HASH_PREFIX = 0xB1ED

    def __init__(
        self,
        sep_tokens: list[int],
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
    ):
        super().__init__(storage_manager_config, chunk_size, hash_algorithm="blake3")

        self._cb_gpu_contexts: dict[int, PlainGPUCacheContext] = {}

    def cb_register_kv_cache(self, instance_id: int, kv_caches: KVCache) -> None:
        """
        Register the KV cache buffer from the blend engine

        Args:
            instance_id: Unique identifier for the blend engine instance
            kv_caches: KVCache object containing the GPU buffer pointers
        """
        gpu_context = PlainGPUCacheContext(kv_caches, self.chunk_size)
        self._cb_gpu_contexts[instance_id] = gpu_context
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
            pre-computed chunks
        """
        # TODO: placeholder
        logger.warning("Looking up pre-computed chunks for key: %s", key)
        return []

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
        """
        # Compute blend-only hash for the keys
        hashed_ipc_keys = key.to_hash_keys(self.token_hasher, self.BLEND_HASH_PREFIX)
        # convert to object key
        obj_keys = ipc_keys_to_object_keys(hashed_ipc_keys)

        assert instance_id in self._cb_gpu_contexts, (
            f"Instance ID {instance_id} not registered for CB KV cache"
        )
        gpu_context = self._cb_gpu_contexts[instance_id]

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
            last_num_tokens = (key.end - key.start - 1) % self.chunk_size + 1
            cpu_shape = gpu_context.get_kv_buffer_shape(num_tokens)
            last_cpu_shape = gpu_context.get_kv_buffer_shape(last_num_tokens)

            layout_desc = MemoryLayoutDesc(
                shapes=[cpu_shape], dtypes=[gpu_context.dtype]
            )
            last_layout_desc = MemoryLayoutDesc(
                shapes=[last_cpu_shape], dtypes=[gpu_context.dtype]
            )
            logger.warning(
                "Layout desc: %s, last layout desc: %s", layout_desc, last_layout_desc
            )

            reserved_dict = self.storage_manager.reserve_write(
                obj_keys[:-1], layout_desc, "new"
            )
            reserved_dict_tail = self.storage_manager.reserve_write(
                [obj_keys[-1]], last_layout_desc, "new"
            )
            reserved_dict.update(reserved_dict_tail)

            for idx, obj_key in enumerate(obj_keys):
                if obj_key in reserved_dict:
                    memory_obj = reserved_dict[obj_key]
                else:
                    continue

                offset_start = idx * self.chunk_size + offset
                offset_end = offset_start + (
                    self.chunk_size if idx < len(obj_keys) - 1 else last_num_tokens
                )
                logger.warning(
                    "offset start and end is %d, %d", offset_start, offset_end
                )

                # Copy from GPU to CPU
                tmp_buffer = gpu_context.get_tmp_gpu_buffer(offset_end - offset_start)
                gpu_kv_slice = gpu_context.slice_kv_cache_on_tokens(
                    offset_start, offset_end
                )
                logger.warning(
                    "tmp buffer shape %s, gpu_kv_slice shape %s",
                    tmp_buffer.shape,
                    gpu_kv_slice.shape,
                )
                with self.lock:
                    tmp_buffer.copy_(gpu_kv_slice, non_blocking=True)
                    lmcache_memcpy_async_d2h(tmp_buffer, memory_obj)

        gpu_context.cupy_stream.launch_host_func(
            self.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )
        logger.info(
            "Stored pre-computed doc with %d tokens (non-skipped chunks: %d)",
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
        # TODO: placeholder
        logger.warning(
            "Retrieving pre-computed chunks for key: %s with ranges: %s at offset: %d",
            key,
            ranges,
            offset,
        )
        return bytes(), True

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
        # TODO: placeholder
        logger.warning("Storing final chunks for key: %s at offset: %d", key, offset)
        return bytes(), True


def get_sep_tokens() -> list[int]:
    """
    Get the separator tokens used for splitting input sequences into paragraphs.

    Returns:
        List of integer token ids that are used as separators.

    Environment variables:
    - `LMCACHE_BLEND_SEP_STR`: the separator string, default is " # # "
    - `LMCACHE_BLEND_MODEL_NAME`: the model name to load the tokenizer, default
        is "openai/gpt-oss-120b"
    - `LMCACHE_BLEND_TOKENIZER_OFFSET`: the offset to add to the token ids,
        default is 1
    """
    sep_tokens_str = os.getenv("LMCACHE_BLEND_SEP_STR", " # # ")
    model_name = os.getenv("LMCACHE_BLEND_MODEL_NAME", "openai/gpt-oss-120b")
    tokenizer_offset = int(os.getenv("LMCACHE_BLEND_TOKENIZER_OFFSET", "1"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sep_tokens = tokenizer.encode(sep_tokens_str)[tokenizer_offset:]

    logger.info("Got sep tokens %s", sep_tokens)

    return sep_tokens


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
    storage_manager_config: StorageManagerConfig,
    host: str = "localhost",
    port: int = 5555,
    chunk_size: int = 256,
    max_workers: int = 1,
    return_engine: bool = False,
    hash_algorithm: str = "blake3",
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        storage_manager_config: Configuration for the storage manager
        host: ZMQ server host
        port: ZMQ server port
        chunk_size: Chunk size for KV cache operations
        max_workers: Maximum number of worker threads for ZMQ server
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive
        hash_algorithm: Hash algorithm for token-based operations

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheEngine)
        If return_engine is False: None (blocks until interrupted)
    """
    sep_tokens = get_sep_tokens()

    # Initialize the engine
    engine = BlendEngine(
        sep_tokens=sep_tokens,
        storage_manager_config=storage_manager_config,
        chunk_size=chunk_size,
    )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{host}:{port}", context=context, max_workers=max_workers
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

    logger.info("LMCache ZMQ cache server is running on tcp://%s:%d", host, port)
    # Start the ZMQ server
    torch.cuda.init()
    server.start()
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
        server.close()
        engine.close()


if __name__ == "__main__":
    args = parse_args()
    storage_manager_config = parse_args_to_config(args)
    run_cache_server(
        storage_manager_config=storage_manager_config,
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        hash_algorithm=args.hash_algorithm,
    )

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
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    parse_args_to_config,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.gpu_context import PlainGPUCacheContext
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.server import MPCacheEngine, parse_args

logger = init_logger(__name__)


class BlendEngine(MPCacheEngine):
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

    def cb_store_pre_computed(self, key: IPCCacheEngineKey, offset: int) -> None:
        """
        Store the pre-computed chunks in the underlying storage for later retrieval.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the pre-computed
                chunks are stored.
            offset: The starting offset in the CB KV cache buffer where the
                pre-computed

        Note:
            The input tokens should not have any separator in it. It should just be
            one "paragraph".
        """
        # TODO: placeholder
        logger.warning(
            "Storing pre-computed chunks for key: %s at offset: %d", key, offset
        )

    def cb_retrieve_pre_computed(
        self, key: IPCCacheEngineKey, ranges: list[tuple[int, int]], offset: int
    ) -> bool:
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

        Returns:
            bool indicating the success of the retrieval and copy operation.
        """
        # TODO: placeholder
        logger.warning(
            "Retrieving pre-computed chunks for key: %s with ranges: %s at offset: %d",
            key,
            ranges,
            offset,
        )
        return True

    def cb_store_final(self, key: IPCCacheEngineKey, offset: int) -> None:
        """
        Store the final chunks in the underlying storage after processing. The stored
        chunk should be accessible for normal mode LLMs.

        Args:
            key: IPCCacheEngineKey containing the token ids for which the final chunks
                are stored.
            offset: The starting offset in the CB KV cache buffer where the final
                chunks are stored.
        """
        # TODO: placeholder
        logger.warning("Storing final chunks for key: %s at offset: %d", key, offset)


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

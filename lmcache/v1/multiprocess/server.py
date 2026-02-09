# SPDX-License-Identifier: Apache-2.0
###
# NOTE: THIS FILE IS SUBJECT TO CHANGE!!!
# TODO LIST:
# - KV Cache management
#   - Thread safe (Read/Write lock)
#   - Eviction policy
# - Double buffer for store/retrieve (5% optimization)
# - Refactor and reuse the existing LMCache classes
# - Lock and unlock
###

# Standard
import argparse
import array
import threading
import time

# Third Party
import cupy
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_keys_to_object_keys,
)
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    unwrapped_tensors = []
    for ipc_wrapper in kv_caches:
        tensor = ipc_wrapper.to_tensor()
        unwrapped_tensors.append(tensor)
    return unwrapped_tensors


def list_to_gpu_tensor(lis: list[int], device: torch.device) -> torch.Tensor:
    return torch.frombuffer(array.array("l", lis), dtype=torch.long).to(
        device, non_blocking=True
    )


class GPUCacheContext:
    """
    Manages the shape and pointers to vLLM GPU KV cache tensors.
    """

    def __init__(self, kv_caches: KVCache, lmcache_chunk_size: int = 256):
        self.kv_caches_ = unwrap_kv_cache_tensors(kv_caches)
        self.device_ = self.kv_caches_[0].device

        # Pointers
        pointers_list = [t.data_ptr() for t in self.kv_caches_]
        self.kv_cache_pointers_ = list_to_gpu_tensor(pointers_list, self.device_)

        # MLA flag
        # MLA shape: [num_blocks, block_size, hidden_dim]
        # MHA shape: [2, num_blocks, block_size, num_heads, head_size]
        self.is_mla_ = self.kv_caches_[0].ndim == 3

        # Shape related
        self.num_layers_ = len(self.kv_caches_)
        if self.is_mla_:
            self.num_blocks_ = self.kv_caches_[0].shape[0]
            self.block_size_ = self.kv_caches_[0].shape[1]
            self.hidden_dim_size_ = self.kv_caches_[0].shape[2]
        else:
            self.num_blocks_ = self.kv_caches_[0].shape[1]
            self.block_size_ = self.kv_caches_[0].shape[2]
            # hidden_dim = num_heads * head_size
            num_heads = self.kv_caches_[0].shape[3]
            head_size = self.kv_caches_[0].shape[4]
            self.hidden_dim_size_ = num_heads * head_size

        # Pre-computed slot mapping
        # shape: [num_blocks, block_size]
        block_ids = torch.arange(
            0, self.num_blocks_, dtype=torch.long, device=self.device_
        ).unsqueeze(1)
        offsets = torch.arange(
            0, self.block_size_, dtype=torch.long, device=self.device_
        ).unsqueeze(0)
        self.slot_mapping_tensor_ = (offsets + block_ids * self.block_size_).reshape(
            (self.num_blocks, self.block_size_)
        )

        # Temporary GPU buffer for transfers
        tmp_buffer_shape = self.get_kv_buffer_shape(lmcache_chunk_size)
        self.tmp_gpu_buffer_ = torch.empty(
            tmp_buffer_shape, dtype=self.dtype, device=self.device_
        )

        # Cuda streams
        self.cuda_stream_ = torch.cuda.Stream(device=self.device_)
        self.cupy_stream_ = cupy.cuda.ExternalStream(
            self.cuda_stream_.cuda_stream, self.device_.index
        )

        # Extra initialization
        self.cupy_stream_.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self.device_)
            ),
            logger,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.kv_caches_[0].dtype

    @property
    def device(self) -> torch.device:
        return self.device_

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        return self.kv_caches_

    @property
    def kv_pointers(self) -> torch.Tensor:
        """
        Returns a GPU tensor of the KV cache pointers
        """
        return self.kv_cache_pointers_

    @property
    def stream(self) -> torch.cuda.Stream:
        """
        Returns the CUDA stream for KV cache operations
        """
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> cupy.cuda.Stream:
        return self.cupy_stream_

    @property
    def block_size(self) -> int:
        """
        Returns the block size (number of tokens per block)
        """
        return self.block_size_

    @property
    def num_layers(self) -> int:
        """
        Returns the number of layers in the model
        """
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        """
        Returns the number of blocks in the KV cache
        """
        return self.num_blocks_

    @property
    def hidden_dim_size(self) -> int:
        """
        Returns the hidden dimension size of the model
        """
        return self.hidden_dim_size_

    @property
    def is_mla(self) -> bool:
        """
        Returns whether the model uses MLA
        """
        return self.is_mla_

    def get_tmp_gpu_buffer(self, num_tokens: int) -> torch.Tensor:
        """
        Returns the temporary GPU buffer for transfers
        """
        return self.tmp_gpu_buffer_[:, :, :num_tokens, :]

    @_lmcache_nvtx_annotate
    def get_slot_mapping_tensor(self, gpu_block_ids: list[int]) -> torch.Tensor:
        """
        Returns the slot mapping tensor for the KV cache on GPU
        """
        gpu_block_ids_tensor = list_to_gpu_tensor(gpu_block_ids, self.device_)
        return self.slot_mapping_tensor_[gpu_block_ids_tensor].flatten().contiguous()

    def get_kv_buffer_shape(self, num_tokens: int) -> torch.Size:
        """
        Returns the shape of the KV buffer for the given number of tokens
        """
        if self.is_mla_:
            return torch.Size((1, self.num_layers_, num_tokens, self.hidden_dim_size_))
        else:
            return torch.Size((2, self.num_layers_, num_tokens, self.hidden_dim_size_))


def update_session_for_key(
    key: IPCCacheEngineKey,
    session_manager: SessionManager,
) -> None:
    """Update session state for a token-mode key.

    For token-mode keys, sets the token sequence on the session and
    computes hashes so they are cached for resolve_keys.

    For hash-mode keys, this is a no-op.

    Args:
        key: An IPC cache engine key (token or hash mode).
        session_manager: The session manager to use.
    """
    if not key.is_token_mode():
        return
    assert key.token_ids is not None
    request_id = key.request_id
    assert request_id is not None, "Token mode requires request_id in key"
    session = session_manager.get_or_create(request_id)
    session.set_tokens(list(key.token_ids))
    session.get_hashes(key.start, key.end)


def resolve_keys(
    keys: list[IPCCacheEngineKey],
    session_manager: SessionManager,
) -> list[IPCCacheEngineKey]:
    """Convert token-mode keys to hash-mode keys.

    For token-mode keys: uses session to retrieve pre-computed rolling
    hashes, then creates hash-mode IPCCacheEngineKey instances.
    update_session_for_key must be called before this function.

    For hash-mode keys: passes through directly.

    Args:
        keys: List of IPC keys (token or hash mode).
        session_manager: The session manager to use.

    Returns:
        List of hash-mode IPCCacheEngineKey.
    """
    resolved: list[IPCCacheEngineKey] = []
    for key in keys:
        if key.is_token_mode():
            assert key.token_ids is not None
            request_id = key.request_id
            assert request_id is not None, "Token mode requires request_id in key"
            session = session_manager.get_or_create(request_id)
            hashes = session.get_hashes(key.start, key.end)
            resolved.extend(
                IPCCacheEngineKey(
                    model_name=key.model_name,
                    world_size=key.world_size,
                    worker_id=key.worker_id,
                    chunk_hash=TokenHasher.hash_to_bytes(h),
                )
                for h in hashes
            )
        else:
            resolved.append(key)
    return resolved


class MPCacheEngine:
    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ):
        # GPU ID -> KV cache tensors
        self.gpu_contexts: dict[int, GPUCacheContext] = {}

        # chunk size
        self.chunk_size = chunk_size

        # thread lock to avoid tmp buffer conflicts
        self.lock = threading.Lock()

        # storage manager
        self.storage_manager = StorageManager(storage_manager_config)

        # Token hasher and session manager for token-based operations
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self.session_manager = SessionManager(self.token_hasher)

    def register_kv_cache(self, instance_id: int, kv_caches: KVCache) -> None:
        """
        Registers the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
            kv_caches (KVCache): The KV cache tensor wrappers from vLLM.
        """
        gpu_context = GPUCacheContext(kv_caches, self.chunk_size)
        self.gpu_contexts[instance_id] = gpu_context
        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """
        Unregisters the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
        """
        if instance_id in self.gpu_contexts:
            del self.gpu_contexts[instance_id]
            logger.info("Unregistered KV cache for GPU ID %d", instance_id)
            torch.cuda.empty_cache()
        else:
            logger.warning("No KV cache found for GPU ID %d to unregister", instance_id)

    @_lmcache_nvtx_annotate
    def store(
        self,
        keys: list[IPCCacheEngineKey],
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Stores the GPU KV cache blocks to CPU.

        Args:
            keys (list[IPCCacheEngineKey]): The IPC keys for the KV cache blocks.
                All keys must have worker_id != None (worker store operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to store.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the store operation. The second
                element indicates whether the store operation was successful.
        """
        for key in keys:
            update_session_for_key(key, self.session_manager)
        ipc_keys = resolve_keys(keys, self.session_manager)

        st = time.perf_counter()

        assert all(k.worker_id is not None for k in ipc_keys), (
            "Must store with worker_id != None"
        )
        obj_keys = ipc_keys_to_object_keys(ipc_keys)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)
            slot_mapping_tensor = gpu_context.get_slot_mapping_tensor(gpu_block_ids)

            # Wait for vLLM to finish
            vllm_event = torch.cuda.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

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

                start = idx * self.chunk_size
                end = start + self.chunk_size
                slot_mapping = slot_mapping_tensor[start:end]

                # Copy from GPU to CPU
                tmp_buffer = gpu_context.get_tmp_gpu_buffer(num_tokens)
                with self.lock:
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_buffer,
                        gpu_context.kv_pointers,
                        slot_mapping,
                        gpu_context.device,
                        gpu_context.block_size * gpu_context.num_blocks,
                        True,
                        gpu_context.is_mla,
                    )

                    assert memory_obj.tensor is not None
                    lmcache_memcpy_async_d2h(tmp_buffer, memory_obj)

            event.record()

        self.gpu_contexts[instance_id].cupy_stream.launch_host_func(
            self.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )
        ed = time.perf_counter()
        if length := len(reserved_dict):
            logger.info(
                "Stored %d tokens in %.3f seconds",
                length * self.chunk_size,
                ed - st,
            )
        return event.ipc_handle(), True

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        keys: list[IPCCacheEngineKey],
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, list[bool]]:
        """
        Retrieves the CPU KV cache and put into GPU blocks.

        Args:
            keys (list[IPCCacheEngineKey]): The IPC keys for the KV cache blocks.
                All keys must have worker_id != None (worker retrieve operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to retrieve into.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.

        Returns:
            tuple[bytes, list[bool]]: The first element is the IPC handle of the event
                that signals the completion of the retrieve operation. The second
                element is a list indicating whether each IPC key was successfully
                retrieved.
        """
        for key in keys:
            update_session_for_key(key, self.session_manager)
        ipc_keys = resolve_keys(keys, self.session_manager)

        st = time.perf_counter()

        assert all(k.worker_id is not None for k in ipc_keys), (
            "Must retrieve with worker_id != None"
        )
        obj_keys = ipc_keys_to_object_keys(ipc_keys)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            for idx, (key, memory_obj) in enumerate(
                zip(keys, memory_objs, strict=False)
            ):
                start = idx * self.chunk_size
                end = start + self.chunk_size
                slot_mapping = slot_mapping_tensor[start:end]

                # Copy from CPU to GPU
                tmp_gpu_buffer_ = gpu_context.get_tmp_gpu_buffer(self.chunk_size)
                with self.lock:
                    lmcache_memcpy_async_h2d(memory_obj, tmp_gpu_buffer_)
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_gpu_buffer_,
                        gpu_context.kv_pointers,
                        slot_mapping,
                        gpu_context.device,
                        gpu_context.block_size * gpu_context.num_blocks,
                        False,
                        gpu_context.is_mla,
                    )

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            slot_mapping_tensor = gpu_context.get_slot_mapping_tensor(gpu_block_ids)

            event = torch.cuda.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            try:
                with self.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), [False] * len(obj_keys)

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    _retrieve_loop(obj_keys, memory_objs)
            except Exception as e:
                logger.warning("Cannot retrieve keys due to exception: %s", str(e))
                return event.ipc_handle(), [False] * len(obj_keys)
            finally:
                event.record()
                gpu_context.cupy_stream.launch_host_func(
                    self.storage_manager.finish_read_prefetched,
                    prefetched_keys,
                )

        tokens_retrieved = len(obj_keys) * self.chunk_size
        ed = time.perf_counter()
        logger.info(
            "Retrieved %d tokens in %.3f seconds",
            tokens_retrieved,
            ed - st,
        )

        return event.ipc_handle(), [True] * len(obj_keys)

    def lookup(
        self,
        keys: list[IPCCacheEngineKey],
    ) -> int:
        """Lookup cache hits for the given keys.

        Args:
            keys: List of cache keys.
                  request_id is embedded in each key.

        Returns:
            Number of matched chunks (prefix match count).
        """
        ipc_keys: list[IPCCacheEngineKey] = []
        for key in keys:
            if key.is_token_mode():
                ipc_keys.extend(key.to_hash_keys(self.token_hasher))
            else:
                ipc_keys.append(key)
        if not ipc_keys:
            return 0
        obj_keys = ipc_keys_to_object_keys(ipc_keys)

        handle = self.storage_manager.submit_prefetch_task(obj_keys)
        while True:
            found_count = self.storage_manager.query_prefetch_status(handle)
            if found_count is not None:
                break
        # NOTE(Kuntai): this assumes two things:
        # 1. the world size is the same between keys
        # 2. the lookup sort the keys in prefix order and breaks at the first failure
        found_count = found_count // ipc_keys[0].world_size
        return found_count

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_chunk_size(self) -> int:
        """
        Returns the chunk size used for KV cache operations.

        Returns:
            int: The chunk size.
        """
        return self.chunk_size

    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        self.session_manager.remove(request_id)

    def debug(self) -> str:
        return "OK"

    def clear(self) -> None:
        """
        Clears all stored KV cache data from the storage manager.
        """
        with self.lock:
            self.storage_manager.memcheck()
            self.storage_manager.clear()
            self.storage_manager.memcheck()

    def close(self) -> None:
        """
        Closes the MPCacheEngine and releases all resources.
        """
        # Close storage manager
        self.storage_manager.close()
        logger.info("MPCacheEngine closed")

        # Release GPU contexts
        self.gpu_contexts.clear()


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
    # Initialize the engine
    engine = MPCacheEngine(
        storage_manager_config=storage_manager_config,
        chunk_size=chunk_size,
        hash_algorithm=hash_algorithm,
    )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{host}:{port}", context=context, max_workers=max_workers
    )

    # Add handlers
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

    logger.info("LMCache ZMQ cache server is running on tcp://%s:%d", host, port)
    # Start the ZMQ server
    torch.cuda.init()
    server.start()
    logger.info("LMCache cache server is running...")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to bind the ZMQ server"
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port to bind the ZMQ server"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256, help="Chunk size for KV cache operations"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--hash-algorithm",
        type=str,
        default="blake3",
        help="Hash algorithm for token-based operations (builtin, sha256_cbor, blake3)",
    )
    parser = add_storage_manager_args(parser)
    return parser.parse_args()


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

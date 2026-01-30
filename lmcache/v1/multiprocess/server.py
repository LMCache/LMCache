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
from lmcache.v1.gpu_connector import lmcache_memcpy_async_d2h, lmcache_memcpy_async_h2d
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineHashKey,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mp_storage_manager import MPStorageManager
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


class MPCacheEngine:
    def __init__(
        self,
        chunk_size: int = 256,
        cpu_buffer_size: float = 5.0,
        disable_lazy_alloc: bool = False,
        hash_algorithm: str = "builtin",
    ):
        # GPU ID -> KV cache tensors
        self.gpu_contexts: dict[int, GPUCacheContext] = {}

        # chunk size
        self.chunk_size = chunk_size

        # thread lock to avoid tmp buffer conflicts
        self.lock = threading.Lock()

        # storage manager
        self.storage_manager = MPStorageManager(cpu_buffer_size, disable_lazy_alloc)

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
    def _store_by_hash(
        self,
        keys: list[IPCCacheEngineHashKey],
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Stores the GPU KV cache blocks to CPU (internal, hash-based keys).

        Args:
            keys (list[IPCCacheEngineHashKey]): The hash keys for the KV cache blocks.
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to store.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the store operation. The second
                element indicates whether the store operation was successful.
        """
        st = time.perf_counter()

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
            fmt = (
                MemoryFormat.KV_MLA_FMT if gpu_context.is_mla else MemoryFormat.KV_2LTD
            )
            reserve_handle, reserved_dict = self.storage_manager.reserve(
                keys, cpu_shape, gpu_context.dtype, fmt=fmt
            )

            for idx, key in enumerate(keys):
                if key in reserved_dict:
                    memory_obj = reserved_dict[key]
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
                        # memory_obj.tensor,
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
            self.storage_manager.commit, reserve_handle
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
    def _retrieve_by_hash(
        self,
        keys: list[IPCCacheEngineHashKey],
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, list[bool]]:
        """
        Retrieves the CPU KV cache and put into GPU blocks (internal, hash-based keys).

        Args:
            keys (list[IPCCacheEngineHashKey]): The hash keys for the KV cache blocks.
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to retrieve into.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.

        Returns:
            tuple[bytes, list[bool]]: The first element is the IPC handle of the event
                that signals the completion of the retrieve operation. The second
                element is a list indicating whether each key was successfully
                retrieved.

        Notes:
            - The caller must ensure that all keys are present in the storage (i.e.,
                a prior lookup should have been performed).
        """
        # NOTE: this function will only return all True or all False even if
        # there is a partial hit. This is because we are requiring all the
        # retrieves objects is pre-locked by the lookup function (so they
        # must be all found)
        st = time.perf_counter()
        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )

        gpu_context = self.gpu_contexts[instance_id]

        def _retrieve_loop(keys: list[IPCCacheEngineHashKey], memory_objs: list[MemoryObj]):
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
                        # memory_obj.tensor,
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

            try:
                with self.storage_manager.retrieve(keys) as memory_objs:
                    _retrieve_loop(keys, memory_objs)
            except Exception as e:
                logger.warning("Cannot retrieve keys: %s", str(e))
                return event.ipc_handle(), [False] * len(keys)
            finally:
                # NOTE: the event.record() should be called before
                # the event ipc handle is returned to the caller.
                event.record()
                gpu_context.cupy_stream.launch_host_func(
                    self.storage_manager.on_retrieve_finished, keys
                )

        tokens_retrieved = len(keys) * self.chunk_size
        ed = time.perf_counter()
        logger.info(
            "Retrieved %d tokens in %.3f seconds",
            tokens_retrieved,
            ed - st,
        )

        return event.ipc_handle(), [True] * len(keys)

    def get_chunk_size(self) -> int:
        """
        Returns the chunk size used for KV cache operations.

        Returns:
            int: The chunk size.
        """
        return self.chunk_size

    def _lookup_by_hash(
        self,
        keys: list[IPCCacheEngineHashKey],
        lock: bool | None = None,
    ) -> list[bool]:
        """
        Looks up the presence of keys in the storage (internal, hash-based keys).

        Args:
            keys (list[IPCCacheEngineHashKey]): The hash keys to look up.
            lock (bool | None): Whether to lock the found keys.

        Returns:
            list[bool]: A list indicating whether each key was found.

        Notes:
            - `lock` is going to be always True in the future.
            - The function does prefix-based lookup. Therefore, it
                requires that the keys are from the same request and
                are in order.
        """
        if not lock:
            logger.warning(
                "MPCacheEngine._lookup_by_hash called with lock=False, this is "
                "not recommended and may cause memory object being pinned "
                "for 5 minutes"
            )

        found_count = self.storage_manager.lookup(keys)
        return [True] * found_count + [False] * (len(keys) - found_count)

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

    # --- Public handlers (token-based) ---

    def lookup(
        self,
        request_id: str,
        key: IPCCacheEngineKey,
    ) -> int:
        """Lookup using token IDs.

        Creates/updates session with key.token_ids, computes all chunk hashes,
        converts to IPCCacheEngineHashKeys, and delegates to _lookup_by_hash().

        Args:
            request_id: Unique request identifier for session tracking.
            key: Token-based key containing full token sequence.

        Returns:
            Number of matched chunks (prefix match count).
        """
        session = self.session_manager.get_or_create(request_id)
        session.compute_all_hashes(list(key.token_ids), self.token_hasher)

        hash_keys = key.to_hash_keys(self.token_hasher)
        if not hash_keys:
            return 0

        results = self._lookup_by_hash(hash_keys, lock=True)
        return sum(results)

    def store(
        self,
        request_ids: list[str],
        keys: list[IPCCacheEngineKey],
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store using token IDs (batched).

        Iterates over request_ids/keys, computes hashes via sessions,
        combines all hash keys, then delegates to _store_by_hash().

        Args:
            request_ids: Unique request identifiers for session tracking.
            keys: Token-based keys, one per request (each contains one chunk's tokens).
            instance_id: GPU instance ID.
            gpu_block_ids: Flattened GPU block IDs.
            event_ipc_handle: IPC handle of the event to wait on.

        Returns:
            Tuple of (event IPC handle, combined success bool).
        """
        combined_hash_keys: list[IPCCacheEngineHashKey] = []

        for request_id, key in zip(request_ids, keys):
            session = self.session_manager.get_or_create(request_id)
            new_hashes = session.append_tokens_and_hash(
                list(key.token_ids), self.token_hasher
            )

            hash_keys = [
                IPCCacheEngineHashKey(
                    model_name=key.model_name,
                    world_size=key.world_size,
                    worker_id=key.worker_id,
                    chunk_hash=TokenHasher.hash_to_bytes(h),
                )
                for h in new_hashes
            ]
            combined_hash_keys.extend(hash_keys)

        if not combined_hash_keys:
            logger.warning(
                "store: no complete chunks from any request "
                "(num_requests=%d)",
                len(request_ids),
            )
            assert instance_id in self.gpu_contexts
            gpu_context = self.gpu_contexts[instance_id]
            with (
                torch.cuda.device(gpu_context.device),
                torch.cuda.stream(gpu_context.stream),
            ):
                event = torch.cuda.Event(interprocess=True)
                event.record()
            return event.ipc_handle(), False

        return self._store_by_hash(
            combined_hash_keys, instance_id, gpu_block_ids, event_ipc_handle
        )

    def retrieve(
        self,
        request_ids: list[str],
        keys: list[IPCCacheEngineKey],
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, list[bool]]:
        """Retrieve using token IDs (batched).

        Iterates over request_ids/keys, computes hashes via sessions,
        combines all hash keys, then delegates to _retrieve_by_hash().

        Args:
            request_ids: Unique request identifiers for session tracking.
            keys: Token-based keys, one per request (each contains all matched tokens).
            instance_id: GPU instance ID.
            gpu_block_ids: Flattened GPU block IDs.
            event_ipc_handle: IPC handle of the event to wait on.

        Returns:
            Tuple of (event IPC handle, combined per-chunk bools).
        """
        combined_hash_keys: list[IPCCacheEngineHashKey] = []

        for request_id, key in zip(request_ids, keys):
            session = self.session_manager.get_or_create(request_id)
            session.compute_all_hashes(list(key.token_ids), self.token_hasher)

            hash_keys = key.to_hash_keys(self.token_hasher)
            combined_hash_keys.extend(hash_keys)

        if not combined_hash_keys:
            logger.warning(
                "retrieve: no complete chunks from any request "
                "(num_requests=%d)",
                len(request_ids),
            )
            assert instance_id in self.gpu_contexts
            gpu_context = self.gpu_contexts[instance_id]
            with (
                torch.cuda.device(gpu_context.device),
                torch.cuda.stream(gpu_context.stream),
            ):
                event = torch.cuda.Event(interprocess=True)
                event.record()
            return event.ipc_handle(), []

        return self._retrieve_by_hash(
            combined_hash_keys, instance_id, gpu_block_ids, event_ipc_handle
        )


def _start_session_cleanup_timer(
    session_manager: SessionManager, interval: float = 60.0
) -> threading.Timer:
    """Start a daemon timer thread for periodic session cleanup.

    Args:
        session_manager: The SessionManager to clean up.
        interval: Cleanup interval in seconds.

    Returns:
        The started Timer thread.
    """

    def _cleanup():
        try:
            session_manager.cleanup_expired()
        except Exception:
            logger.exception("Error during session cleanup")
        # Re-schedule
        timer = threading.Timer(interval, _cleanup)
        timer.daemon = True
        timer.start()

    timer = threading.Timer(interval, _cleanup)
    timer.daemon = True
    timer.start()
    return timer


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
    host: str = "localhost",
    port: int = 5555,
    chunk_size: int = 256,
    cpu_buffer_size: float = 5.0,
    max_workers: int = 1,
    disable_lazy_alloc: bool = False,
    return_engine: bool = False,
    hash_algorithm: str = "builtin",
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        host: ZMQ server host
        port: ZMQ server port
        chunk_size: Chunk size for KV cache operations
        cpu_buffer_size: CPU buffer size in GB
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
        chunk_size, cpu_buffer_size, disable_lazy_alloc, hash_algorithm
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

    # Start periodic session cleanup
    _start_session_cleanup_timer(engine.session_manager)

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
        "--cpu-buffer-size", type=float, default=5.0, help="CPU buffer size in GB"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Maximum number of worker threads"
    )

    parser.add_argument(
        "--disable-lazy-alloc", action="store_true", help="Disable lazy allocation"
    )
    parser.add_argument(
        "--hash-algorithm",
        type=str,
        default="builtin",
        help="Hash algorithm for token-based operations (e.g., builtin, sha256_cbor)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cache_server(
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        cpu_buffer_size=args.cpu_buffer_size,
        max_workers=args.max_workers,
        disable_lazy_alloc=args.disable_lazy_alloc,
        hash_algorithm=args.hash_algorithm,
    )

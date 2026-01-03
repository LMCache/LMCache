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
from typing import Optional
import argparse
import array
import asyncio
import threading
import time

# Third Party
import cupy
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey, KVCache
from lmcache.v1.multiprocess.mp_metadata import (
    create_mp_server_metadata_from_gpu_context,
)
from lmcache.v1.multiprocess.mp_storage_manager import MPStorageManager
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
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
        config: Optional[LMCacheEngineConfig] = None,
    ):
        # GPU ID -> KV cache tensors
        self.gpu_contexts: dict[int, GPUCacheContext] = {}

        # chunk size
        self.chunk_size = chunk_size

        # thread lock to avoid tmp buffer conflicts
        self.lock = threading.Lock()

        # Store config for deferred storage manager initialization
        self._config = config
        self._cpu_buffer_size = cpu_buffer_size

        # Event loop for async storage operations (if L2 configured)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        if config is not None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._run_event_loop, daemon=True
            )
            self._loop_thread.start()

        # Storage manager (created after first GPU context registers, for metadata)
        self._storage_manager: Optional[MPStorageManager] = None

        if config is not None:
            logger.info(
                "MPCacheEngine initialized with L2 storage plugins: %s",
                config.storage_plugins or "none",
            )

    def _run_event_loop(self):
        """Run the event loop in a background thread."""
        if self._loop:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

    def _initialize_storage_manager(self, gpu_context: GPUCacheContext) -> None:
        """Initialize storage manager after first GPU context registers."""
        if self._storage_manager is not None:
            return

        metadata = None
        if self._config is not None:
            # Create metadata from GPU context for L2 storage
            metadata = create_mp_server_metadata_from_gpu_context(
                model_name="mp_server_model",
                num_layers=gpu_context.num_layers,
                hidden_dim=gpu_context.hidden_dim_size,
                kv_dtype=gpu_context.dtype,
                chunk_size=self.chunk_size,
                is_mla=gpu_context.is_mla,
            )

        self._storage_manager = MPStorageManager(
            cpu_buffer_size=self._cpu_buffer_size,
            config=self._config,
            metadata=metadata,
            loop=self._loop,
        )

        if self._config is not None and self._storage_manager.has_l2_storage():
            logger.info(
                "MPCacheEngine storage manager initialized with L2 backends: %s",
                self._storage_manager.get_l2_backend_names(),
            )

    @property
    def storage_manager(self) -> MPStorageManager:
        """Get storage manager (lazy initialization)."""
        if self._storage_manager is None:
            # Create basic storage manager without L2 support
            self._storage_manager = MPStorageManager(
                cpu_buffer_size=self._cpu_buffer_size,
            )
        return self._storage_manager

    def register_kv_cache(self, instance_id: int, kv_caches: KVCache) -> None:
        """
        Registers the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
            kv_caches (KVCache): The KV cache tensor wrappers from vLLM.
        """
        gpu_context = GPUCacheContext(kv_caches)
        self.gpu_contexts[instance_id] = gpu_context

        # Initialize storage manager with metadata from first GPU context
        self._initialize_storage_manager(gpu_context)

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
            keys (list[IPCCacheEngineKey]): The keys for the KV cache blocks.
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
                    memory_obj.tensor.copy_(tmp_buffer, non_blocking=True)

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
            keys (list[IPCCacheEngineKey]): The keys for the KV cache blocks.
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

        def _retrieve_loop(keys: list[IPCCacheEngineKey], memory_objs: list[MemoryObj]):
            for idx, (key, memory_obj) in enumerate(
                zip(keys, memory_objs, strict=False)
            ):
                start = idx * self.chunk_size
                end = start + self.chunk_size
                slot_mapping = slot_mapping_tensor[start:end]

                # Copy from CPU to GPU
                tmp_gpu_buffer_ = gpu_context.get_tmp_gpu_buffer(self.chunk_size)
                with self.lock:
                    tmp_gpu_buffer_.copy_(memory_obj.tensor, non_blocking=True)

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

    def lookup(
        self,
        keys: list[IPCCacheEngineKey],
        lock: bool | None = None,
    ) -> list[bool]:
        """
        Looks up the presence of keys in the storage. The keys
        should belongs to a single request (same prompt).

        Args:
            keys (list[IPCCacheEngineKey]): The keys to look up.
            lock (bool | None): Whether to lock the found keys.

        Returns:
            list[bool]: A list indicating whether each key was found.

        Notes:
            - `lock` is going to be always True in the future.
            - The function does prefix-based lookup. Therefore, it
                requires that the keys are from the same request and
                are in order.
        """
        # NOTE: we are doing per-request lookup, the caller need
        # to be aware of this! We need to add this to the doc!
        if not lock:
            logger.warning(
                "MPCacheEngine.lookup called with lock=False, this is "
                "not recommended and may cause memory object being pinned "
                "for 5 minutes"
            )

        found_count = self.storage_manager.lookup(keys)
        return [True] * found_count + [False] * (len(keys) - found_count)

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


def create_config_from_args(args) -> Optional[LMCacheEngineConfig]:
    """
    Create LMCacheEngineConfig from server CLI arguments.

    Returns None if no storage plugins are configured (CPU-only mode).
    Returns config if storage plugins are configured (L2 storage enabled).
    """
    # If config file is provided, use it
    if hasattr(args, "config_file") and args.config_file:
        config = LMCacheEngineConfig.from_file(args.config_file)
        logger.info("Loaded config from file: %s", args.config_file)
        return config

    # Check if any storage plugin is requested via CLI
    has_raw_block = hasattr(args, "raw_block_device") and args.raw_block_device
    has_local_disk = hasattr(args, "local_disk") and args.local_disk

    if not has_raw_block and not has_local_disk:
        # No storage plugins, use legacy MPCacheEngine
        return None

    # Build config from CLI args
    extra_config = {}
    storage_plugins = []

    if has_raw_block:
        storage_plugins.append("raw_block")
        module_path = "lmcache.v1.storage_backend.plugins.rust_raw_block_backend"
        extra_config.update(
            {
                "storage_plugin.raw_block.module_path": module_path,
                "storage_plugin.raw_block.class_name": "RustRawBlockBackend",
                "rust_raw_block.device_path": args.raw_block_device,
                "rust_raw_block.use_odirect": getattr(args, "enable_odirect", False),
                "rust_raw_block.manifest_write_interval": 1,
            }
        )
        if hasattr(args, "raw_block_capacity_gb") and args.raw_block_capacity_gb > 0:
            extra_config["rust_raw_block.capacity_bytes"] = int(
                args.raw_block_capacity_gb * (1 << 30)
            )
        logger.info(
            "Configured Raw Block backend: device=%s, odirect=%s",
            args.raw_block_device,
            getattr(args, "enable_odirect", False),
        )

    if has_local_disk:
        storage_plugins.append("local_disk")
        extra_config.update(
            {
                "local_disk": args.local_disk,
                "max_local_disk_size": getattr(args, "max_local_disk_size", 100.0),
            }
        )
        logger.info("Configured Local Disk backend: path=%s", args.local_disk)

    # Create config
    config = LMCacheEngineConfig(
        chunk_size=args.chunk_size,
        local_cpu=True,
        max_local_cpu_size=args.cpu_buffer_size,
        storage_plugins=storage_plugins if storage_plugins else None,
        extra_config=extra_config if extra_config else None,
    )

    return config


def create_cache_engine(args) -> MPCacheEngine:
    """
    Create the cache engine with optional L2 storage support.

    If storage plugins are configured (via CLI or config file),
    the engine will support tiered storage (CPU + L2).
    """
    config = create_config_from_args(args)

    if config is not None:
        logger.info(
            "Using MPCacheEngine with L2 storage plugins: %s",
            config.storage_plugins or "none",
        )
    else:
        logger.info("Using MPCacheEngine (CPU-only storage)")

    return MPCacheEngine(
        chunk_size=args.chunk_size,
        cpu_buffer_size=args.cpu_buffer_size,
        config=config,
    )


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
    args=None,
):
    """
    Run the LMCache multi-process server.

    Args:
        host: Server host address
        port: Server port
        chunk_size: KV cache chunk size
        cpu_buffer_size: CPU buffer size in GB
        max_workers: Number of worker threads
        args: Optional parsed command-line arguments (overrides other args)
    """
    # Support both keyword args and argparse namespace
    if args is None:
        # Standard
        import argparse

        args = argparse.Namespace(
            host=host,
            port=port,
            chunk_size=chunk_size,
            cpu_buffer_size=cpu_buffer_size,
            max_workers=max_workers,
        )

    # Initialize the appropriate engine
    engine = create_cache_engine(args)

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{args.host}:{args.port}",
        context=context,
        max_workers=args.max_workers,
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
    add_handler_helper(server, RequestType.NOOP, engine.debug)

    # Start the server
    torch.cuda.init()
    server.start()
    logger.info("LMCache cache server is running on %s:%d", args.host, args.port)

    # Dummy loop to keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.close()
        if hasattr(engine, "close"):
            engine.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache Multi-process Cache Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CPU-only server
  python3 -m lmcache.v1.multiprocess.server --cpu-buffer-size 10

  # Server with Raw Block backend (for TP > 1)
  python3 -m lmcache.v1.multiprocess.server \\
      --raw-block-device /dev/nvme0n1 \\
      --cpu-buffer-size 10 \\
      --enable-odirect

  # Server with config file
  python3 -m lmcache.v1.multiprocess.server --config-file my_config.yaml
""",
    )

    # Basic server options
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to bind the server (default: 5555)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for KV cache in tokens (default: 256)",
    )
    parser.add_argument(
        "--cpu-buffer-size",
        type=float,
        default=5.0,
        help="CPU buffer size in GB (default: 5.0)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of worker threads (default: 1)",
    )

    # Configuration file
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to LMCache config YAML file for storage backends",
    )

    # Raw Block backend options
    raw_block_group = parser.add_argument_group("Raw Block Backend")
    raw_block_group.add_argument(
        "--raw-block-device",
        type=str,
        default=None,
        help="Path to raw block device (e.g., /dev/nvme0n1)",
    )
    raw_block_group.add_argument(
        "--raw-block-capacity-gb",
        type=float,
        default=0,
        help="Raw block capacity in GB (0 = use full device)",
    )
    raw_block_group.add_argument(
        "--enable-odirect",
        action="store_true",
        help="Enable O_DIRECT for raw block I/O (bypasses page cache)",
    )

    # Local disk backend options
    disk_group = parser.add_argument_group("Local Disk Backend")
    disk_group.add_argument(
        "--local-disk",
        type=str,
        default=None,
        help="Path to local disk storage directory",
    )
    disk_group.add_argument(
        "--max-local-disk-size",
        type=float,
        default=100.0,
        help="Maximum local disk storage size in GB (default: 100)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cache_server(args)

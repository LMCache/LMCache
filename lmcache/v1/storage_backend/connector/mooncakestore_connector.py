# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, no_type_check
import asyncio
import json
import operator
import os

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

METADATA_BYTES_LEN = 28


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    transfer_timeout: int
    storage_root_dir: str
    prefer_local_alloc: bool = False

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
            transfer_timeout=config.get("transfer_timeout", 1),
            storage_root_dir=config.get("storage_root_dir", ""),
            prefer_local_alloc=config.get("prefer_local_alloc", False),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_file_path)

    @staticmethod
    def load_from_lmcache_config(
        config: "LMCacheEngineConfig",
    ) -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        extra_config = config.extra_config
        if extra_config is None:
            raise ValueError("The extra config is not set.")
        return MooncakeStoreConfig(
            local_hostname=extra_config["local_hostname"],
            metadata_server=extra_config["metadata_server"],
            global_segment_size=extra_config.get("global_segment_size", 3355443200),
            local_buffer_size=extra_config.get("local_buffer_size", 1073741824),
            protocol=extra_config.get("protocol", "tcp"),
            device_name=extra_config.get("device_name", ""),
            master_server_address=extra_config["master_server_address"],
            transfer_timeout=extra_config.get("transfer_timeout", 1),
            storage_root_dir=extra_config.get("storage_root_dir", ""),
            prefer_local_alloc=extra_config.get("prefer_local_alloc", False),
        )


class MooncakestoreConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        dev_name,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_config: Optional[LMCacheEngineConfig],
    ):
        try:
            # Third Party
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()
            config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
            if config_file_path is not None:
                self.config = MooncakeStoreConfig.from_file(config_file_path)
            elif lmcache_config is not None:
                self.config = MooncakeStoreConfig.load_from_lmcache_config(
                    lmcache_config
                )
            else:
                raise ValueError("MOONCAKE_CONFIG_PATH/lmcache_config must be provided")

            if host != "" and port != 0:
                self.config.master_server_address = host + ":" + str(port)
            if dev_name != "":
                self.config.device_name = dev_name
            logger.info("Mooncake Configuration loaded. config: %s", self.config)

            # Check if storage_root_dir exists and set environment variable
            if (
                self.config.storage_root_dir is not None
                and self.config.storage_root_dir != ""
            ):
                os.environ["MOONCAKE_STORAGE_ROOT_DIR"] = self.config.storage_root_dir
                logger.info(
                    "Set MOONCAKE_STORAGE_ROOT_DIR to: %s", self.config.storage_root_dir
                )

            logger.info("Setting up Mooncake store with parameters:")
            logger.info(f"  local_hostname: {self.config.local_hostname}")
            logger.info(f"  metadata_server: {self.config.metadata_server}")
            logger.info(f"  global_segment_size: {self.config.global_segment_size}")
            logger.info(f"  local_buffer_size: {self.config.local_buffer_size}")
            logger.info(f"  protocol: {self.config.protocol}")
            logger.info(f"  device_name: {self.config.device_name}")
            logger.info(f"  master_server_address: {self.config.master_server_address}")

            self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
            logger.info("Mooncake store setup completed successfully")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.registered_buffer_ptr = None

        # Initialize ReplicateConfig
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1

        # Set preferred_segment based on configuration
        if self.config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.store.get_hostname()

        # Register CPU buffer for zero-copy operations
        self._register_cpu_buffer()

        logger.info("MooncakeConnector initialized successfully.")

    def init_chunk_meta(
        self,
        config: Optional[LMCacheEngineConfig],
        metadata: Optional[LMCacheEngineMetadata],
    ) -> None:
        """Initialize chunk metadata and log the configuration."""
        super().init_chunk_meta(config, metadata)

        if self.meta_shape and self.meta_dtype and self.meta_fmt:
            logger.info("MooncakeConnector using optimized mode")
        else:
            logger.info("MooncakeConnector using legacy mode")
            logger.info(
                "Try setting 'save_chunk_meta' to False in the configuration "
                "for better performance"
            )

    def _register_cpu_buffer(self):
        """Register CPU buffer for zero-copy operations."""
        try:
            allocator = self.local_cpu_backend.memory_allocator
            if hasattr(allocator, "pin_allocator") and hasattr(
                allocator.pin_allocator, "buffer"
            ):
                buffer = allocator.pin_allocator.buffer
                self.registered_buffer_ptr = buffer.data_ptr()
                result = self.store.register_buffer(buffer.data_ptr(), buffer.numel())
                if result == 0:
                    logger.info(
                        f"Registered: {hex(buffer.data_ptr())}, {buffer.numel()} bytes"
                    )
                else:
                    logger.warning(f"Buffer registration failed: error={result}")
                    self.registered_buffer_ptr = None
            else:
                self.registered_buffer_ptr = None
        except Exception as e:
            logger.error(f"Buffer registration error: {e}")
            self.registered_buffer_ptr = None

    def _unregister_cpu_buffer(self):
        """Unregister CPU buffer."""
        if self.registered_buffer_ptr is not None:
            result = self.store.unregister_buffer(self.registered_buffer_ptr)
            if result == 0:
                logger.info(f"Unregistered buffer: {hex(self.registered_buffer_ptr)}")
            else:
                logger.warning(f"Buffer unregistration failed: error={result}")
            self.registered_buffer_ptr = None

    def support_batched_get(self) -> bool:
        """
        Check if the connector supports batched get

        Returns:
            True if batched get is supported, False otherwise
        """
        return True

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string())

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string())

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get operation - the only supported get method.
        Uses batch_get_into (with metadata) or batch_get_buffer (without metadata).
        """
        if not keys:
            return []

        # Check if we have metadata for zero-copy operations
        if self.save_chunk_meta:
            # Use legacy mode with metadata stored in remote
            return await self._batch_get_buffer(keys)
        else:
            # Use optimized mode with local metadata
            return await self._batch_get_into(keys)

    async def _batch_get_into(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Zero-copy batch get using batch_get_into when metadata is available locally.
        This is used when save_chunk_meta=False (metadata not stored remotely).
        """
        if not self.meta_shape or not self.meta_dtype or not self.meta_fmt:
            logger.error(
                f"Metadata required for batch_get_into but not available: "
                f"meta_shape={self.meta_shape}, "
                f"meta_dtype={self.meta_dtype}, "
                f"meta_fmt={self.meta_fmt}"
            )
            return [None] * len(keys)

        logger.debug(f"Using batch_get_into for {len(keys)} keys (zero-copy mode)")

        # Reserve a buffer for every requested chunk
        memory_objs: list[Optional[MemoryObj]] = []
        valid_idx: list[int] = []

        key_strs: list[str] = []
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []

        for i, _ in enumerate(keys):
            buf = self.local_cpu_backend.allocate(
                self.meta_shape, self.meta_dtype, self.meta_fmt
            )
            memory_objs.append(buf)
            buf_tensor = buf.tensor
            if buf is not None and buf_tensor is not None:
                valid_idx.append(i)

                # Prepare the argument lists for the C++ call
                key_strs.append(keys[i].to_string())
                buffer_ptrs.append(buf_tensor.data_ptr())
                buffer_sizes.append(buf_tensor.numel() * buf_tensor.element_size())

        if not valid_idx:
            logger.warning("Batch-get aborted: unable to allocate any buffers.")
            return [None] * len(keys)

        try:
            # Single RPC call for multiple chunks
            logger.debug(f"Calling batch_get_into with {len(key_strs)} keys")
            bytes_read_list = await asyncio.to_thread(
                self.store.batch_get_into, key_strs, buffer_ptrs, buffer_sizes
            )
            logger.debug(f"batch_get_into returned: {bytes_read_list}")

            # Assemble the final result list
            results: list[Optional[MemoryObj]] = [None] * len(keys)

            for i, n_read in zip(valid_idx, bytes_read_list, strict=False):
                if n_read <= 0:
                    logger.warning(
                        f"batch_get_into failed for key {keys[i]} (code={n_read})"
                    )
                    memory_objs[i].ref_count_down()  # type: ignore
                    continue

                try:
                    results[i] = self.reshape_partial_chunk(
                        memory_objs[i],  # type: ignore
                        n_read,
                    )
                except Exception as exc:
                    logger.error(f"Reshape failed for key {keys[i]}: {exc}")
                    memory_objs[i].ref_count_down()  # type: ignore

            return results

        except Exception as exc:
            logger.error(f"batch_get_into threw exception: {str(exc)}")
            # Release any buffers we successfully allocated
            for i in valid_idx:
                memory_objs[i].ref_count_down()  # type: ignore
            return [None] * len(keys)

    async def _batch_get_buffer(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get using batch_get_buffer when metadata is stored remotely.
        This is used when save_chunk_meta=True (metadata stored with data).
        """
        key_strs = [key.to_string() for key in keys]

        try:
            buffers = await asyncio.to_thread(self.store.batch_get_buffer, key_strs)
        except Exception as e:
            logger.error(f"batch_get_buffer failed: {str(e)}")
            return [None] * len(keys)

        results: list[Optional[MemoryObj]] = []
        for i, buffer in enumerate(buffers):
            if buffer is None:
                logger.warning(f"Buffer {i} is None for key {key_strs[i]}")
                results.append(None)
                continue
            try:
                memory_obj = self._process_buffer_with_metadata(buffer)
                results.append(memory_obj)
            except Exception as e:
                logger.error(
                    f"Failed to process buffer {i} for key {key_strs[i]}: {str(e)}"
                )
                results.append(None)
        return results

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Single get method - NOT SUPPORTED.
        Use batched_get instead for all operations.
        """
        logger.error("Single get operation is not supported. Use batched_get instead.")
        raise NotImplementedError(
            "Single get is not supported. Use batched_get([key]) instead."
        )

    def _process_buffer_with_metadata(self, buffer: bytes) -> Optional[MemoryObj]:
        """
        Process buffer that contains metadata + data.
        Used when save_chunk_meta=True (metadata stored remotely).
        """
        retrieved_view = memoryview(buffer)
        metadata_bytes = retrieved_view[:METADATA_BYTES_LEN]
        if metadata_bytes is None or len(metadata_bytes) != METADATA_BYTES_LEN:
            return None

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        assert len(retrieved_view) == metadata.length + METADATA_BYTES_LEN

        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        if memory_obj.tensor is not None:
            assert metadata.dtype is not None
            num_elements = reduce(operator.mul, metadata.shape)
            temp_tensor = torch.frombuffer(
                buffer,
                dtype=metadata.dtype,
                offset=METADATA_BYTES_LEN,
                count=num_elements,
            ).reshape(metadata.shape)

            memory_obj.tensor.copy_(temp_tensor)
            return memory_obj
        else:
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Put operation with metadata-consistent handling.
        Uses put_from (without metadata) or
        put_parts (with metadata) to match get behavior.
        """
        key_str = key.to_string()

        # Check metadata handling mode to match get behavior
        if self.save_chunk_meta:
            # Use put_parts with metadata stored remotely
            await self._put_with_metadata(key_str, memory_obj)
        else:
            # Use put_from without metadata (zero-copy)
            await self._put_without_metadata(key_str, memory_obj)

    async def _put_without_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Zero-copy put using put_from when metadata is not stored remotely.
        This is used when save_chunk_meta=False (matches _batch_get_into).
        """
        try:
            tensor = memory_obj.tensor
            assert tensor is not None
            buffer_ptr = tensor.data_ptr()
            buffer_size = tensor.numel() * tensor.element_size()

            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_from,
                    key_str,
                    buffer_ptr,
                    buffer_size,
                    self.replica_config,
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when putting key {key_str} using put_from. "
                "Decode instance may redo prefill."
            )
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_from: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    async def _put_with_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Put using put_parts when metadata is stored remotely.
        This is used when save_chunk_meta=True (matches _batch_get_buffer).
        """
        try:
            # Serialize data and metadata
            kv_bytes = memory_obj.byte_array
            kv_shape = memory_obj.get_shape()
            kv_dtype = memory_obj.get_dtype()
            memory_format = memory_obj.get_memory_format()

            metadata_bytes = RemoteMetadata(
                len(kv_bytes), kv_shape, kv_dtype, memory_format
            ).serialize()
            assert len(metadata_bytes) == METADATA_BYTES_LEN

            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_parts, key_str, metadata_bytes, kv_bytes
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when putting key {key_str} using put_parts. "
                "Decode instance may redo prefill."
            )
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_parts: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        # Unregister buffer before closing the store
        self._unregister_cpu_buffer()

        self.store.close()
        logger.info("Closed the mooncake store connection")

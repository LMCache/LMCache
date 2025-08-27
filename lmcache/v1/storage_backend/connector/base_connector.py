# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import abc

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)


class RemoteConnector(metaclass=abc.ABCMeta):
    """
    Interface for remote connector
    """

    save_chunk_meta: bool = True
    meta_shape: Optional[torch.Size] = None
    meta_dtype: Optional[torch.dtype] = None
    meta_fmt: Optional[MemoryFormat] = None
    full_chunk_size: Optional[int] = None
    single_token_size: Optional[int] = None

    def init_chunk_meta(
        self,
        config: Optional[LMCacheEngineConfig],
        metadata: Optional[LMCacheEngineMetadata],
    ) -> None:
        # TODO: support layerwise later
        if (
            config is None
            or metadata is None
            or config.extra_config is None
            or config.extra_config.get("save_chunk_meta", True)
            or config.use_layerwise
        ):
            return

        self.save_chunk_meta = False
        self.meta_shape = torch.Size(
            [
                metadata.kv_shape[1],
                metadata.kv_shape[0],
                metadata.kv_shape[2],
                metadata.kv_shape[3] * metadata.kv_shape[4],
            ]
        )
        self.meta_dtype = metadata.kv_dtype
        self.meta_fmt = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )
        dtype_size = torch.tensor([], dtype=metadata.kv_dtype).element_size()
        num_elements = 1
        for dim in metadata.kv_shape:
            num_elements *= dim
        self.full_chunk_size = dtype_size * num_elements
        assert self.full_chunk_size is not None
        assert self.full_chunk_size % metadata.kv_shape[2] == 0
        self.single_token_size = self.full_chunk_size // metadata.kv_shape[2]
        logger.info(
            f"init remote connector metadata info, "
            f"shape: {self.meta_shape}, "
            f"dtype: {self.meta_dtype}, "
            f"fmt: {self.meta_fmt}, "
            f"full chunk size: {self.full_chunk_size}, "
            f"single token size: {self.single_token_size}"
        )

    def reshape_partial_chunk(
        self,
        memory_obj: MemoryObj,
        bytes_read: int,
    ) -> MemoryObj:
        assert self.full_chunk_size is not None
        assert self.single_token_size is not None
        if (
            bytes_read % self.single_token_size != 0
            or bytes_read > self.full_chunk_size
        ):
            raise ValueError(
                f"bytes_read: {bytes_read} is illegal, "
                f"single_token_size: {self.single_token_size}, "
                f"full_chunk_size: {self.full_chunk_size}"
            )

        if bytes_read == self.full_chunk_size:
            # full chunk, return directly
            return memory_obj

        # NOTE: for unfull chunk, we have no way to verify
        shape_list = list(memory_obj.meta.shape)
        shape_list[2] = bytes_read // self.single_token_size
        actual_shape = torch.Size(shape_list)
        memory_obj.raw_data = memory_obj.raw_data[:bytes_read]
        memory_obj.meta.shape = actual_shape

        return memory_obj

    def post_init(self):
        """
        Post-initialization method to be called after the connector is created.
        This can be used to perform any additional setup required by the connector.
        """
        logger.info("Dummy post-initializing remote connector")

    @abc.abstractmethod
    async def exists(self, key: CacheEngineKey) -> bool:
        """
        Check if the remote server contains the key

        Input:
            key: a CacheEngineKey

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exists_sync(self, key: CacheEngineKey) -> bool:
        """
        Check if the remote server contains the key synchronized

        Input:
            key: a CacheEngineKey

        Returns:
            True if the cache engine contains the key, False otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Get the memory_obj of the corresponding key

        Input:
            key: the key of the corresponding object

        Returns:
            The memory_obj of the corresponding key
            Return None if the key does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Send the memory_obj with the corresponding key directly
        to the remote server. Will decrease the ref count after
        send finishes.

        Input:
            key: the CacheEngine key
            memory_obj: the memory_obj of the corresponding key
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def list(self) -> List[str]:
        """
        List all keys in the remote server

        Returns:
            A list of keys in the remote server
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        """
        Close remote server

        """
        raise NotImplementedError

    def support_ping(self) -> bool:
        """
        Check if the connector supports ping operation

        Returns:
            True if ping is supported, False otherwise
        """
        return False

    async def ping(self) -> int:
        """
        Ping the remote server

        Returns:
            The error code, 0 means success
        """
        raise NotImplementedError

    def support_batched_get(self) -> bool:
        """
        Check if the connector supports batched get

        Returns:
            True if batched get is supported, False otherwise
        """
        return False

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batched get the memory_objs of the corresponding keys

        Input:
            keys: the keys of the corresponding objects

        Returns:
            The memory_objs of the corresponding keys
            Return None if the key does not exist
        """
        raise NotImplementedError

    def support_batched_put(self) -> bool:
        """
        Check if the connector supports batched put
        Returns:
            True if batched put is supported, False otherwise
        """
        return False

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        """
        Batched put the memory_objs with the corresponding keys
        Input:
            keys: the keys of the corresponding objects
            memory_objs: the memory_objs of the corresponding keys
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

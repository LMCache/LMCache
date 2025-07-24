# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Iterable, Optional, Tuple
import abc

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


class LMCBackendInterface(metaclass=abc.ABCMeta):
    def __init__(
        self,
        dst_device: str = "cuda",
    ):
        """Initialize the storage backend.

        :param dst_device: the device where the retrieved KV be stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.

        :raise: RuntimeError if the device is not valid
        """
        try:
            torch.device(dst_device)
        except RuntimeError:
            raise

        self.dst_device = dst_device

    @abc.abstractmethod
    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking=True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        :param key: the key of the token chunk, in the format of
                    CacheEngineKey
        :param kv_chunk: the kv cache of the token chunk, as a big tensor.
        :param blocking: to block the call before the operation is
            completed.

        :return: None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Query if a key is in the cache or not
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve the KV cache chunk by the given key

        :param key: the key of the token chunk, including
         prefix hash and format

        :return: the kv cache of the token chunk, in the format
            of a big tensor and None if the key is not found
        """
        raise NotImplementedError

    def batched_put(
        self,
        keys_and_chunks: Iterable[Tuple[CacheEngineKey, torch.Tensor]],
        blocking=True,
    ) -> int:
        """
        Store the multiple keys and KV cache chunks into the cache engine in a
        batched manner.

        :param keys: the iterable of keys of the token chunks, in the format of
                CacheEngineKey
        :param kv_chunks: the iterable of kv cache of the token chunks, in the
                format of a big tensor
        :param blocking: whether to block the call before the operation is
                completed

        :return: the number of chunks are stored
        """
        logger.info("Using default batched implementation of the put() method")
        nchunks = 0
        for key, kv_chunk in keys_and_chunks:
            self.put(key, kv_chunk, blocking=blocking)
            nchunks += 1
        return nchunks

    def batched_get(
        self,
        keys: Iterable[CacheEngineKey],
    ) -> Iterable[Optional[torch.Tensor]]:
        """
        Retrieve the kv cache chunks by the given keys in a batched manner


        :param keys: the iterator of keys of the token chunks, including prefix
                hash and format

        :return: the iterator of kv cache of the token chunks, in the format
            of a big tensor and None if the key is not found
        """
        logger.info("Using default batched implementation of the get() method")
        for key in keys:
            if self.contains(key):  # Jiayi: This seems to be redundant?
                yield self.get(key)
            else:
                yield None

    @abc.abstractmethod
    def close(self):
        """
        Do the cleanup things
        Children classes should override this method if necessary
        """
        pass

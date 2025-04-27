import abc
from typing import Iterable, Optional, Tuple

import torch

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
        Store the KV cache chunk into the cache engine.

        :param key: The key of the token chunk (CacheEngineKey).
        :param kv_chunk: The KV cache chunk tensor. 
                         Expected shape: [2, num_layers, ...], where the specific
                         dimensions depend on the format (e.g., vLLM or\nHuggingFace)
                         and chunk size. The first dimension represents K (0) and V\n(1).
        :param blocking: Whether to block the call until the operation is\ncompleted.

        :return: None

        Note:
            The KV cache chunk should NOT have the "batch" dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Query if a key is in the cache or not.
        
        :param key: The key to check (CacheEngineKey).
        :return: True if the key exists, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve the KV cache chunk by the given key.

        :param key: The key of the token chunk (CacheEngineKey).

        :return: The KV cache chunk tensor with shape [2, num_layers, ...] 
                 if the key is found, otherwise None. The tensor will be on
                 the device specified by `self.dst_device`.
        """
        raise NotImplementedError

    def batched_put(
        self,
        keys_and_chunks: Iterable[Tuple[CacheEngineKey, torch.Tensor]],
        blocking=True,
    ) -> int:
        """
        Store multiple keys and KV cache chunks into the cache engine in a
        batched manner.

        :param keys_and_chunks: An iterable of tuples, where each tuple contains
                                (CacheEngineKey, kv_chunk_tensor). The tensor
                                should have shape [2, num_layers, ...].
        :param blocking: Whether to block the call until all operations are\ncompleted.

        :return: The number of chunks successfully stored.
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
        Retrieve KV cache chunks for the given keys in a batched manner.

        :param keys: An iterable of keys (CacheEngineKey) to retrieve.

        :return: An iterable yielding the corresponding KV cache chunk tensors 
                 (shape [2, num_layers, ...]) or None if a key is not found. 
                 Tensors will be on the device specified by `self.dst_device`.
        """
        logger.info("Using default batched implementation of the get() method")
        for key in keys:
            # Optimization: Directly call get and yield result, no need for\ncontains check
            # The get method should handle returning None if not found.
            yield self.get(key)

    @abc.abstractmethod
    def close(self):
        """
        Perform cleanup operations for the storage backend.
        Children classes should override this method if necessary.
        """
        pass

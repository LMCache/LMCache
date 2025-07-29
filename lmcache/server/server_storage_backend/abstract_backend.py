# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import abc

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class LMSBackendInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def put(
        self,
        key: str,
        kv_chunk_bytes: bytearray,
        blocking=True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache server.

        Args:
            key: the key of the token chunk, in the format of str
            kv_chunk: the kv cache (bytearray) of the token chunk,
            in the format of a big tensor
            blocking: whether to block the call before the operation is
            completed

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def contains(
        self,
        key: str,
    ) -> bool:
        """
        Query if a key is in the cache or not
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        Output:
            the kv cache of the token chunk, in the format of a big tensor
            None if the key is not found
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_keys(
        self,
    ) -> List[str]:
        """
        Retrieve the KV cache chunk by the given key

        Input:
            key: the key of the token chunk, including prefix hash and format

        Output:
            the kv cache of the token chunk, in the format of a big tensor
            None if the key is not found
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Do the cleanup things
        Children classes should override this method if necessary
        """
        pass

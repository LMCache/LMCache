# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import abc

# Third Party
import torch


@dataclass
class KVObj:
    chunk_idx: int
    size: int  # size of the obj in bytes
    data: torch.Tensor


class BasePool(metaclass=abc.ABCMeta):
    """
    Interface for mem pool
    """

    @abc.abstractmethod
    def allocate(self, kv_chunk: torch.Tensor) -> Optional[KVObj]:
        """
        Allocate a buffer memory pointer from the memory pool.

        Input:
            kv_chunk: the kv tensor to be stored

        Returns:
            KVObj with a memory pointer (torch tensor view).
            None if memory is full.

        Note:
            This does not perform the actual memory movement.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(self, kv_obj: KVObj):
        """
        Free the corresponding memory chunk

        Input:
            the KVObj to be freed
        """
        raise NotImplementedError

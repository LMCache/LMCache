import abc
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class KVObj:
    chunk_idx: int
    data: torch.Tensor

    @property
    def size(self):
        """Get size in GB"""
        size_in_bytes = self.data.numel() * self.data.element_size()
        return size_in_bytes / (1024**3)


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

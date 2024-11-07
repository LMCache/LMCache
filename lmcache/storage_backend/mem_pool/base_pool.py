import abc
import torch
from lmcache.utils import CacheEngineKey

class BaseMemPool(metaclass=abc.ABCMeta):
    """
    Interface for mem pool
    """
    
    @abc.abstractmethod
    def put():
        """
        Put the KV cache of the tokens into the memory.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get() -> torch.Tensor:
        """
        Get the KV cache of the tokens into the memory.
        """
        raise NotImplementedError
    
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Any
import torch

class BaseVirtualConnector(ABC):
    """
    A base virtual connector class that provides a common interface for different connector implementations.
    This class defines the core API that all connectors should implement.
    """

    def __init__(self, hidden_dim_size: int, num_layers: int, chunk_size: int):
        """
        Initialize the base virtual connector.
        
        Args:
            hidden_dim_size (int): The size of the hidden dimension
            num_layers (int): The number of layers in the model
            chunk_size (int): The size of each chunk for processing
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size

    @abstractmethod
    def get_hash(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prefix_hash: Optional[Any] = None
    ) -> List[Any]:
        """
        Get the hash for the given token IDs.
        
        Args:
            token_ids (torch.Tensor): The token IDs to hash
            mask (Optional[torch.Tensor]): Optional mask for the token IDs
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            List[Any]: List of hash values for the token IDs
        """
        raise NotImplementedError

    @abstractmethod
    def store_kv(
        self,
        token_ids: torch.Tensor,
        kv_caches: Union[torch.Tensor, List[torch.Tensor]],
        prefix_hash: Optional[Any] = None
    ) -> Tuple[bool, List[Any]]:
        """
        Store the KV caches.
        
        Args:
            token_ids (torch.Tensor): The token IDs
            kv_caches (Union[torch.Tensor, List[torch.Tensor]]): The KV caches to store
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            Tuple[bool, List[Any]]: Success status and list of hash values for stored KV caches
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_kv(
        self,
        token_ids: torch.Tensor,
        kv_caches: Union[torch.Tensor, List[torch.Tensor]],
        prefix_hash: Optional[Any] = None
    ) -> bool:
        """
        Retrieve the KV caches.
        
        Args:
            token_ids (torch.Tensor): The token IDs
            kv_caches (Union[torch.Tensor, List[torch.Tensor]]): The KV caches to retrieve into
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            bool: Success status of the retrieval
        """
        raise NotImplementedError

    @abstractmethod
    def store_kv_hash(
        self,
        hash_: List[Any],
        kv_caches: List[torch.Tensor]
    ) -> Tuple[bool, List[Any]]:
        """
        Store the KV caches using existing hash values.
        
        Args:
            hash_ (List[Any]): List of hash values to use for storage
            kv_caches (List[torch.Tensor]): The KV caches to store
            
        Returns:
            Tuple[bool, List[Any]]: Success status and list of hash values for stored KV caches
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_kv_hash(
        self,
        hash_: List[Any],
        kv_caches: List[torch.Tensor]
    ) -> bool:
        """
        Retrieve the KV caches using existing hash values.
        
        Args:
            hash_ (List[Any]): List of hash values to use for retrieval
            kv_caches (List[torch.Tensor]): The KV caches to retrieve into
            
        Returns:
            bool: Success status of the retrieval
        """
        raise NotImplementedError 
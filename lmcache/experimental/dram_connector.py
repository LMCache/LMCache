import abc
from typing import List, Optional, Tuple

import torch

import lmcache.c_ops as lmc_ops
from lmcache.experimental.memory_management import MemoryFormat, MemoryObj
from lmcache.utils import _lmcache_nvtx_annotate


class DramConnectorInterface(abc.ABC):

    @abc.abstractmethod
    def to_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into Dram.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from 
            Dram.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.
        """
        raise NotImplementedError


class SGLangDramNestedConnector(DramConnectorInterface):

    def __init__(self, hidden_dim_size: int, num_layers: int):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

    def to_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_BLOB:
            raise ValueError(
                "The memory object should be in KV_BLOB format in"
                " order to be processed by NestedTupleGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        kvcaches: Tuple[Tuple[torch.Tensor, ...], ...] = kwargs["kvcaches"]
        for layer_id, layer in enumerate(kvcaches):
            k, v = layer
            hidden_shape = k.shape[1:]
            k[start:end].copy_(memory_obj.tensor[0, layer_id].reshape(
                -1, *hidden_shape))
            v[start:end].copy_(memory_obj.tensor[1, layer_id].reshape(
                -1, *hidden_shape))

    def from_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        kvcaches: Tuple[Tuple[torch.Tensor, ...], ...] = kwargs["kvcaches"]
        for layer_id, layer in enumerate(kvcaches):
            k, v = layer
            hidden_shape = k.shape[1:]
            memory_obj.tensor[0, layer_id].reshape(-1, *hidden_shape).copy_(
                k[start:end])
            memory_obj.tensor[1, layer_id].reshape(-1, *hidden_shape).copy_(
                v[start:end])

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size])

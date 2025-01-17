import abc
from typing import Tuple, Union

import torch

import lmcache.c_ops as lmc_ops
from lmcache.experimental.memory_management import (BufferMemoryObj,
                                                    MemoryFormat, MemoryObj)
from lmcache.utils import _lmcache_nvtx_annotate


class GPUConnectorInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_gpu(self, memory_obj: Union[MemoryObj, BufferMemoryObj], start: int,
               end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from 
            GPU.
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


class VLLMNestedTupleGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_tokens, ...]

    The token dimension is specified by `token_dim` when constructing the
    connector.

    It will produce / consume memory object with KV_BLOB format
    """

    def __init__(self, hidden_dim_size: int, num_layers: int):
        """
        :param int gpu_token_dim: The token dimension of the GPU KV cache in
            the nested tuple.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

    # TODO(Jiayi): fix the gpu memory
    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: Union[MemoryObj, BufferMemoryObj], start: int,
               end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        """
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
                -1, *hidden_shape),
                               non_blocking=False)
            v[start:end].copy_(memory_obj.tensor[1, layer_id].reshape(
                -1, *hidden_shape),
                               non_blocking=False)

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        :raises ValueError: If 'kvcaches' is not provided in kwargs, or the 
            memory object is not in KV_BLOB format.
        :raises AssertionError: If the memory object does not have a tensor.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        kvcaches: Tuple[Tuple[torch.Tensor, ...], ...] = kwargs["kvcaches"]

        put_stream = torch.cuda.Stream()
        # Wait for all operations on the default stream to finish
        put_stream.wait_stream(torch.cuda.default_stream(
            kvcaches[0][0].device))

        for layer_id, layer in enumerate(kvcaches):
            k, v = layer
            k.record_stream(put_stream)
            v.record_stream(put_stream)

        with torch.cuda.stream(put_stream):
            for layer_id, layer in enumerate(kvcaches):
                k, v = layer
                memory_obj.tensor[1, layer_id].copy_(v[start:end].reshape(
                    -1, self.hidden_dim_size).contiguous(),
                                                     non_blocking=True)
                memory_obj.tensor[0, layer_id].copy_(k[start:end].reshape(
                    -1, self.hidden_dim_size).contiguous(),
                                                     non_blocking=True)
        put_stream.synchronize()
        memory_obj.metadata.fmt = MemoryFormat.KV_BLOB

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_BLOB format
    """

    def __init__(self, hidden_dim_size: int, num_layers: int):
        """
        :param int gpu_token_dim: The token dimension of the GPU KV cache in
            the nested tuple.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: Union[MemoryObj, BufferMemoryObj], start: int,
               end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_BLOB:
            raise ValueError(
                "The memory object should be in KV_BLOB format in"
                " order to be processed by NestedTupleGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: Tuple[Tuple[torch.Tensor, ...], ...] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        for layer_id, layer in enumerate(kvcaches):
            k, v = layer[0], layer[1]
            lmc_ops.reshape_and_cache_back_flash(memory_obj.tensor, k, v,
                                                 slot_mapping[start:end],
                                                 layer_id)

        # TODO(Jiayi): Currently, this is a blocking operation.
        # We might be able to continue other decode jobs while
        # waiting for the copy to finish.

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        :raises ValueError: If 'kvcaches' is not provided in kwargs, or the 
            memory object is not in KV_BLOB format.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "offset" in kwargs:
            start = start - kwargs["offset"]
            end = end - kwargs["offset"]
        assert start >= 0 and end >= start

        kvcaches: Tuple[Tuple[torch.Tensor, ...], ...] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        for layer_id, layer in enumerate(kvcaches):
            k, v = layer[0], layer[1]
            lmc_ops.load_and_reshape_flash(memory_obj.tensor, k, v,
                                           slot_mapping[start:end], layer_id)

        torch.cuda.synchronize()
        memory_obj.metadata.fmt = MemoryFormat.KV_BLOB

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size])

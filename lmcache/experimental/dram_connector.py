import abc

import torch

from lmcache.experimental.memory_management import MemoryFormat, MemoryObj


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

    def __init__(self, hidden_dim_size: int, num_layers: int, chunk_size: int):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size

    def to_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_BLOB:
            raise ValueError(
                "The memory object should be in KV_BLOB format in"
                " order to be processed by NestedTupleGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "retrieve_status" not in kwargs:
            raise ValueError("'retrieve_status' should be provided in kwargs \
                for sglang support")

        kvcaches = kwargs["kvcaches"]
        retrieve_status = kwargs["retrieve_status"]
        keys, values = kvcaches
        for layer_id, layer in enumerate(zip(keys, values)):
            k, v = layer
            hidden_shape = k.shape[1:]
            k[start:end].copy_(memory_obj.tensor[0, layer_id].reshape(
                -1, *hidden_shape))
            v[start:end].copy_(memory_obj.tensor[1, layer_id].reshape(
                -1, *hidden_shape))

        retrieve_status[start // self.chunk_size] = 0

    def from_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "store_status" not in kwargs:
            raise ValueError(
                "'store_status' should be provided in kwargs for sglang support"
            )

        kvcaches = kwargs["kvcaches"]
        store_status = kwargs["store_status"]
        keys, values = kvcaches
        for layer_id, layer in enumerate(zip(keys, values)):
            k, v = layer
            hidden_shape = k.shape[1:]
            memory_obj.tensor[0, layer_id].reshape(-1, *hidden_shape).copy_(
                k[start:end])
            memory_obj.tensor[1, layer_id].reshape(-1, *hidden_shape).copy_(
                v[start:end])

        store_status[start // self.chunk_size] = 0

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size])


class SGLangDramNestedConnectorInner(DramConnectorInterface):

    def __init__(self, hidden_dim_size: int, num_layers: int, chunk_size: int):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size

    def to_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_BLOB:
            raise ValueError(
                "The memory object should be in KV_BLOB format in"
                " order to be processed by NestedTupleGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "retrieve_status" not in kwargs:
            raise ValueError("'retrieve_status' should be provided in kwargs \
                for sglang support")

        kvcaches = kwargs["kvcaches"]
        retrieve_status = kwargs["retrieve_status"]
        keys, values = kvcaches
        hidden_shape = keys.shape[1:]
        keys[start:end].copy_(memory_obj.tensor[0].reshape(-1, *hidden_shape))
        values[start:end].copy_(memory_obj.tensor[1].reshape(-1, *hidden_shape))

        retrieve_status[start // self.chunk_size] = 0

    def from_dram(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "store_status" not in kwargs:
            raise ValueError(
                "'store_status' should be provided in kwargs for sglang support"
            )

        kvcaches = kwargs["kvcaches"]
        store_status = kwargs["store_status"]
        keys, values = kvcaches
        # for layer_id, layer in enumerate(zip(keys, values)):
        #     k, v = layer
        #     hidden_shape = k.shape[1:]
        #     memory_obj.tensor[0, layer_id].reshape(-1, *hidden_shape).copy_(
        #         k[start:end])
        #     memory_obj.tensor[1, layer_id].reshape(-1, *hidden_shape).copy_(
        #         v[start:end])
        hidden_shape = keys.shape[1:]
        memory_obj.tensor[0].reshape(-1, *hidden_shape).copy_(keys[start:end])
        memory_obj.tensor[1].reshape(-1, *hidden_shape).copy_(values[start:end])

        store_status[start // self.chunk_size] = 0

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, num_tokens, self.num_layers, self.hidden_dim_size])

# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import List, Optional, Tuple

import torch

import lmcache.c_ops as lmc_ops
from lmcache.experimental.memory_management import (  # noqa: E501
    GPUMemoryAllocator, MemoryFormat, MemoryObj)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class GPUConnectorInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
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

    It will produce / consume memory object with KV_2LTD format
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
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        """
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
            raise ValueError(
                "The memory object should be in KV_2LTD format in"
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
            memory object is not in KV_2LTD format.
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

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(self, hidden_dim_size: int, num_layers: int):
        """
        :param int gpu_token_dim: The token dimension of the GPU KV cache in
            the nested tuple.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
            raise ValueError(
                "The memory object should be in KV_2LTD format in"
                " order to be processed by VLLMPagedMemGPUConnector")

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
            memory object is not in KV_2LTD format.
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

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(self,
                 hidden_dim_size: int,
                 num_layers: int,
                 use_gpu: bool = False,
                 **kwargs):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this 
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(num_layers,
                                             dtype=torch.int64,
                                             device='cpu',
                                             pin_memory=True)
        self.pointers_initialized = False
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        if use_gpu:
            assert "chunk_size" in kwargs, \
                    "chunk_size should be provided to create a GPU buffer."
            assert "dtype" in kwargs, \
                    "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, \
                    "device should be provided to create a GPU buffer."
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(shape,
                                          dtype=kwargs["dtype"],
                                          device=kwargs["device"])

    def _pointers_are_good(self, kv_caches: List[torch.Tensor]):
        """
        Check if the initialized pointers are the same as the pointers in 
        the KV caches. 

        Returns:
            bool: True if the pointers are the same, False otherwise (
                including uninitialized).
        """
        if not self.pointers_initialized:
            return False

        for i in range(self.num_layers):
            if self.kv_cache_pointers[i] != kv_caches[i].data_ptr():
                return False
        return True

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]):
        for i in range(self.num_layers):
            self.kv_cache_pointers[i] = kv_caches[i].data_ptr()
        self.pointers_initialized = True
        # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
        self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[0].shape[2]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note: 
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the 
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
            raise ValueError(
                "The memory object should be in KV_2LTD format in"
                " order to be processed by VLLMPagedMemGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        if not self._pointers_are_good(kvcaches):
            self._initialize_pointers(kvcaches)

        # NOTE(ApostaC): By default, detour from a GPU buffer is slower
        # than directly copying from the CPU.
        # So disabling it for now and use direct copy from CPU to GPU.

        #if self.gpu_buffer is None or \
        #        end - start != self.gpu_buffer.shape[2]:
        #    lmc_ops.multi_layer_kv_transfer(memory_obj.tensor,
        #                                    self.kv_cache_pointers,
        #                                    slot_mapping[start:end],
        #                                    kvcaches[0].device,
        #                                    self.page_buffer_size, False)
        #else:
        #    # Memobj -> gpu_buffer -> kvcaches
        #    assert self.gpu_buffer.device == kvcaches[0].device
        #    tmp_gpu_buffer = self.gpu_buffer[:, :, :end-start, :]
        #    tmp_gpu_buffer.copy_(memory_obj.tensor, non_blocking=True)
        #    lmc_ops.multi_layer_kv_transfer(
        #        tmp_gpu_buffer,
        #        self.kv_cache_pointers,
        #        slot_mapping[start:end],
        #        kvcaches[0].device, self.page_buffer_size, False)

        lmc_ops.multi_layer_kv_transfer(memory_obj.tensor,
                                        self.kv_cache_pointers,
                                        slot_mapping[start:end],
                                        kvcaches[0].device,
                                        self.page_buffer_size, False, False)

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note: 
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the 
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        #if not self.pointers_initialized:
        if not self._pointers_are_good(kvcaches):
            self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or \
                end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer(memory_obj.tensor,
                                            self.kv_cache_pointers,
                                            slot_mapping[start:end],
                                            kvcaches[0].device,
                                            self.page_buffer_size, True, False)
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, :end - start, :]
            lmc_ops.multi_layer_kv_transfer(tmp_gpu_buffer,
                                            self.kv_cache_pointers,
                                            slot_mapping[start:end],
                                            kvcaches[0].device,
                                            self.page_buffer_size, True, False)
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.cuda.synchronize()

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [2, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemLayerwiseGPUConnector(GPUConnectorInterface):
    """
    """

    def __init__(self,
                 hidden_dim_size: int,
                 num_layers: int,
                 use_gpu: bool = False,
                 **kwargs):
        """
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        if use_gpu:
            assert "chunk_size" in kwargs, \
                    "chunk_size should be provided to create a GPU buffer."
            assert "dtype" in kwargs, \
                    "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, \
                    "device should be provided to create a GPU buffer."

            # FIXME (Jiayi): Please remove this hardcode
            max_tokens = 32000
            shape = self.get_shape(max_tokens)
            self.dtype = kwargs["dtype"]
            self.device = kwargs["device"]

            num_elements = shape.numel()

            # All sizes are in bytes
            element_size = torch.tensor([], dtype=self.dtype).element_size()
            gpu_buffer_size = num_elements * element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(gpu_buffer_size,
                                                           device=self.device)

            self.load_stream = torch.cuda.Stream()
            self.store_stream = torch.cuda.Stream()
        else:
            # TODO(Jiayi): Support `use_gpu=False` case
            pass

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory 
        objects to paged GPU memory. The first iteration will prepare some 
        related metadata. In each of the following iterations, it will first 
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer -> 
        paged GPU memory. The last iteration simply waits for the last layer 
        to finish. 
        In total, this the generator will yield num_layers + 2 times.
        
        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.
        
        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        
        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_T2D)
        assert tmp_gpu_buffer_obj is not None, \
            "Failed to allocate GPU buffer in GPUConnector"
        assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):

            memory_objs_layer = yield
            current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id-1}")

            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                for start, end, memory_obj in zip(starts,
                                                  ends,
                                                  memory_objs_layer,
                                                  strict=False):
                    assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                    tmp_gpu_buffer_obj.tensor[start - offset:end -
                                              offset].copy_(memory_obj.tensor,
                                                            non_blocking=True)

                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    kvcaches[layer_id][0],
                    kvcaches[layer_id][1],
                    slot_mapping_full,
                    False,
                )
        yield

        # synchronize the last layer
        current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(self, memory_objs: List[List[MemoryObj]],
                         starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the paged GPU 
        memory to the memory objects. The first iteration will prepare some 
        related metadata and initiate the transfer in the first layer. In each 
        of the following iterations, it will first wait until the storing of 
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU 
        memory -> GPU buffer -> memory objects. The last iteration simply waits 
        for the last layer to finish. 
        In total, this the generator will yield num_layers + 1 times.
        
        :param memory_objs: The memory objects to store the KV cache. The first 
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.
        
        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.
        
        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        
        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_T2D)
        assert tmp_gpu_buffer_obj is not None, \
            "Failed to allocate GPU buffer in GPUConnector"
        assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    kvcaches[layer_id][0],
                    kvcaches[layer_id][1],
                    slot_mapping_full,
                    True,
                )
                for start, end, memory_obj in zip(starts,
                                                  ends,
                                                  memory_objs_layer,
                                                  strict=False):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start - offset:end - offset],
                        non_blocking=True)

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])


class VLLMPagedMemGPUConnectorMLA(GPUConnectorInterface):
    """
    The GPU KV cache should be list of tensors, one for each layer
    More specifically, we have:
    - Tensor of each layer: [num_blocks, block_size, head_size]

    It will produce / consume memory object with KV_BLOB format
    """

    def __init__(self,
                 aligned_head_size: int,
                 num_layers: int,
                 use_gpu: bool = False,
                 **kwargs):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this 
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.aligned_head_size = aligned_head_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(num_layers,
                                             dtype=torch.int64,
                                             device='cpu',
                                             pin_memory=True)
        self.pointers_initialized = False

        self.gpu_buffer: Optional[torch.Tensor] = None
        if use_gpu:
            assert "chunk_size" in kwargs, \
                    "chunk_size should be provided to create a GPU buffer."
            assert "dtype" in kwargs, \
                    "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, \
                    "device should be provided to create a GPU buffer."
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(shape,
                                          dtype=kwargs["dtype"],
                                          device=kwargs["device"])

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]):

        for i in range(self.num_layers):
            self.kv_cache_pointers[i] = kv_caches[i].data_ptr()
        self.pointers_initialized = True

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note: 
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the 
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
            raise ValueError(
                "The memory object should be in KV_MLA_FMT format in"
                " order to be processed by VLLMPagedMemGPUConnector")

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        if not self.pointers_initialized:
            self._initialize_pointers(kvcaches)

        lmc_ops.multi_layer_kv_transfer(memory_obj.tensor,
                                        self.kv_cache_pointers,
                                        slot_mapping[start:end],
                                        kvcaches[0].device, 0, False, True)

        torch.cuda.synchronize(kvcaches[0].device)

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_MLA_FMT.

        Note: 
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the 
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        if not self.pointers_initialized:
            self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or \
                end - start != self.gpu_buffer.shape[1]:
            lmc_ops.multi_layer_kv_transfer(memory_obj.tensor,
                                            self.kv_cache_pointers,
                                            slot_mapping[start:end],
                                            kvcaches[0].device, 0, True, True)
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, :end - start, :]
            lmc_ops.multi_layer_kv_transfer(tmp_gpu_buffer,
                                            self.kv_cache_pointers,
                                            slot_mapping[start:end],
                                            kvcaches[0].device, 0, True, True)
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        torch.cuda.synchronize()
        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size(
            [1, self.num_layers, num_tokens, self.aligned_head_size])

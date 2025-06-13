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

# Standard
from typing import List, Optional, Tuple, Union
import abc
import threading

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.memory_management import GPUMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
import lmcache.c_ops as lmc_ops

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
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched load the data from a GPU memory into the memory objects.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data from GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens."""
        raise NotImplementedError


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        device = kv_caches[0].device
        assert device.type == "cuda", "The device should be CUDA."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_layers, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)
        if self.use_mla:
            # kv_caches[0].shape: [num_pages, page_size, head_size]
            assert kv_caches[0].dim() == 3
            self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[0].shape[1]
        else:
            # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
            assert kv_caches[0].dim() == 5
            self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[0].shape[2]

        return self.kv_cache_pointers_on_gpu[idx]

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

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            kvcaches[0].device,
            self.page_buffer_size,
            False,
            self.use_mla,
        )

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

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0].device,
                self.page_buffer_size,
                True,
                self.use_mla,
            )
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
            lmc_ops.multi_layer_kv_transfer(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0].device,
                self.page_buffer_size,
                True,
                self.use_mla,
            )
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.cuda.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMBufferLayerwiseGPUConnector(GPUConnectorInterface):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        # TODO(Jiayi): remove this hardcode
        self.cache_positions = True

        self.fused_rotary_emb = None

        assert use_gpu, "use_gpu must be true in VLLMBufferLayerwiseGPUConnector"
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        max_tokens = kwargs.get("max_tokens", 32000)
        logger.info(
            f"Using max_tokens={max_tokens} for VLLMBufferLayerwiseGPUConnector"
        )
        self.get_shape(max_tokens)  # Calculate shape (unused)
        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.buffer_mapping = {}

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV cache for the given layer ID.
        This function is used to get the KV cache from the GPU buffer.
        """
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")

        gpu_buffer = self.buffer_mapping[layer_id].tensor
        return gpu_buffer[0], gpu_buffer[1]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        buf_offset = starts[0]
        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=kvcaches[0].device
            )

        buffer_shape = self.get_shape(num_all_tokens)
        compute_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        load_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert compute_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert load_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert compute_gpu_buffer_obj.tensor is not None
        assert load_gpu_buffer_obj.tensor is not None

        # current_stream = torch.cuda.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=kvcaches[0].device
            )
        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    kvcaches[layer_id - 2][0],
                    kvcaches[layer_id - 2][1],
                    slot_mapping_full,
                    False,
                    False,  # shape is [2, num_tokens, hidden_dim]
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.old_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
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

        buf_start = 0
        slot_mapping_chunks = []
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            slot_mapping_chunks.append(slot_mapping[start:end])
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(
                        start, end, device=kvcaches[0].device, dtype=torch.int64
                    )
                )

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert tmp_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert tmp_gpu_buffer_obj.tensor is not None

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
                    False,  # shape is [2, num_tokens, hidden_dim]
                )
                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.old_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, num_tokens, self.hidden_dim_size])


class VLLMPagedMemLayerwiseGPUConnector(GPUConnectorInterface):
    """ """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        self.gpu_buffer_allocator = None

        assert "chunk_size" in kwargs, (
            "chunk_size should be provided to create a GPU buffer."
        )
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        # All sizes are in bytes
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        # Initialize streaming buffer pool with fixed small buffers
        # This eliminates dynamic allocation and provides constant memory footprint
        # Conservative buffer sizing for production stability
        self.streaming_buffer_size = 512  # tokens per buffer
        self.num_streaming_buffers = 4  # concurrent operations

        logger.info(
            f"Initializing StreamingBufferPool with {self.num_streaming_buffers} "
            f"buffers of {self.streaming_buffer_size} tokens each"
        )

        try:
            self._init_streaming_buffer_pool()
            logger.info("StreamingBufferPool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize streaming buffer pool: {e}")
            # Graceful fallback - continue without GPU buffers
            self.streaming_buffers = []
            self.buffer_available = []
            self.buffer_lock = threading.Lock()
            logger.warning("Using CPU-only fallback mode - operations will continue")

        try:
            self.load_stream = torch.cuda.Stream()
            self.store_stream = torch.cuda.Stream()
        except Exception as e:
            logger.warning(f"Failed to create CUDA streams: {e}, using default stream")
            self.load_stream = None
            self.store_stream = None

    def _init_streaming_buffer_pool(self):
        """Initialize fixed-size streaming buffer pool."""
        self.streaming_buffers = []
        self.buffer_available = []
        self.buffer_lock = threading.Lock()

        buffer_shape = self.get_shape(self.streaming_buffer_size)

        for i in range(self.num_streaming_buffers):
            try:
                # Allocate small fixed buffers that should always succeed
                buffer_tensor = torch.zeros(
                    buffer_shape, dtype=self.dtype, device=self.device
                )
                self.streaming_buffers.append(buffer_tensor)
                self.buffer_available.append(True)
                logger.debug(f"Allocated streaming buffer {i}: {buffer_shape}")
            except Exception as e:
                logger.error(f"Failed to allocate streaming buffer {i}: {e}")
                break

        if len(self.streaming_buffers) == 0:
            raise RuntimeError("Could not allocate any streaming buffers")

        logger.info(f"Initialized {len(self.streaming_buffers)} streaming buffers")

    def _acquire_streaming_buffer(self):
        """Acquire an available buffer from the pool."""
        with self.buffer_lock:
            for i, available in enumerate(self.buffer_available):
                if available:
                    self.buffer_available[i] = False
                    logger.debug(f"Acquired streaming buffer {i}")
                    return self.streaming_buffers[i]

        logger.warning("No streaming buffers available - using CPU fallback")
        return None

    def _release_streaming_buffer(self, buffer_tensor):
        """Return a buffer to the pool."""
        with self.buffer_lock:
            for i, pool_buffer in enumerate(self.streaming_buffers):
                if pool_buffer is buffer_tensor:
                    self.buffer_available[i] = True
                    logger.debug(f"Released streaming buffer {i}")
                    return

        logger.error("Attempted to release unknown buffer")

    def batched_from_gpu_streaming(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Stream-based KV cache transfer that uses fixed-size buffer pool.
        Handles requests of any size by chunking them through small buffers.
        This eliminates dynamic allocation and provides constant memory footprint.

        Generator yields num_layers + 1 times to match expected behavior:
        - Initial yield for setup
        - One yield per layer after processing
        """
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        current_stream = torch.cuda.current_stream()

        # Process each layer sequentially using streaming buffers
        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]

            # Process transfers for this layer
            if self.store_stream is not None:
                with torch.cuda.stream(self.store_stream):
                    self.store_stream.wait_stream(current_stream)
                    self._process_layer_streaming(
                        layer_id,
                        memory_objs_layer,
                        starts,
                        ends,
                        kvcaches,
                        slot_mapping,
                    )
            else:
                # Use default stream
                self._process_layer_streaming(
                    layer_id, memory_objs_layer, starts, ends, kvcaches, slot_mapping
                )

            yield
            if sync and self.store_stream is not None:
                self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # Final yield to match expected generator behavior (num_layers + 1 yields total)
        yield
        logger.debug(f"Completed streaming transfer for {self.num_layers} layers")

    def _process_layer_streaming(
        self, layer_id, memory_objs_layer, starts, ends, kvcaches, slot_mapping
    ):
        """Process a single layer using streaming buffer pool."""

        # Build full slot mapping for this layer
        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        # Try to get a streaming buffer for the entire layer
        streaming_buffer = self._acquire_streaming_buffer()

        if streaming_buffer is not None:
            try:
                # Use streaming buffer - transfer full layer at once if it fits
                total_tokens = sum(
                    end - start for start, end in zip(starts, ends, strict=False)
                )

                if total_tokens <= self.streaming_buffer_size:
                    # Full layer fits in one buffer
                    lmc_ops.single_layer_kv_transfer(
                        streaming_buffer[:total_tokens],
                        kvcaches[layer_id][0],
                        kvcaches[layer_id][1],
                        slot_mapping_full,
                        True,  # from_gpu=True
                        True,  # is_layerwise=True
                    )

                    # Copy to memory objects
                    offset = 0
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        chunk_size = end - start
                        assert memory_obj.tensor is not None
                        memory_obj.tensor.copy_(
                            streaming_buffer[offset : offset + chunk_size],
                            non_blocking=True,
                        )
                        offset += chunk_size
                else:
                    # Layer too large, process in chunks
                    self._process_layer_chunked(
                        layer_id,
                        memory_objs_layer,
                        starts,
                        ends,
                        kvcaches,
                        slot_mapping,
                        streaming_buffer,
                    )

            finally:
                self._release_streaming_buffer(streaming_buffer)
        else:
            # No streaming buffer available, use CPU fallback
            logger.warning(
                f"No streaming buffer available for layer {layer_id}, "
                f"using CPU fallback"
            )
            self._process_layer_cpu_fallback(
                layer_id, memory_objs_layer, starts, ends, kvcaches, slot_mapping_full
            )

    def _process_layer_chunked(
        self,
        layer_id,
        memory_objs_layer,
        starts,
        ends,
        kvcaches,
        slot_mapping,
        streaming_buffer,
    ):
        """Process layer in chunks when it doesn't fit in streaming buffer."""
        for start, end, memory_obj in zip(
            starts, ends, memory_objs_layer, strict=False
        ):
            assert memory_obj.tensor is not None

            offset = start
            mem_offset = 0

            # Process in chunks using streaming buffer size
            while offset < end:
                chunk_end = min(offset + self.streaming_buffer_size, end)
                chunk_tokens = chunk_end - offset

                # Create slot mapping for this chunk
                chunk_slot_mapping = slot_mapping[offset:chunk_end]

                # Transfer from GPU to streaming buffer
                lmc_ops.single_layer_kv_transfer(
                    streaming_buffer[:chunk_tokens],
                    kvcaches[layer_id][0],
                    kvcaches[layer_id][1],
                    chunk_slot_mapping,
                    True,  # from_gpu=True
                    True,  # is_layerwise=True
                )

                # Copy from streaming buffer to memory object
                memory_obj.tensor[mem_offset : mem_offset + chunk_tokens].copy_(
                    streaming_buffer[:chunk_tokens],
                    non_blocking=True,
                )

                offset = chunk_end
                mem_offset += chunk_tokens

    def _process_layer_cpu_fallback(
        self, layer_id, memory_objs_layer, starts, ends, kvcaches, slot_mapping_full
    ):
        """CPU fallback when no streaming buffers available."""
        for start, end, memory_obj in zip(
            starts, ends, memory_objs_layer, strict=False
        ):
            assert memory_obj.tensor is not None

            # Create temporary tensor for direct transfer
            chunk_tokens = end - start
            chunk_shape = self.get_shape(chunk_tokens)
            temp_tensor = torch.zeros(
                chunk_shape, dtype=memory_obj.tensor.dtype, device="cpu"
            )

            # Create slot mapping for this chunk
            chunk_slot_mapping = slot_mapping_full[start:end]

            lmc_ops.single_layer_kv_transfer(
                temp_tensor,
                kvcaches[layer_id][0],
                kvcaches[layer_id][1],
                chunk_slot_mapping,
                True,  # from_gpu=True
                True,  # is_layerwise=True
            )

            # Copy to final memory object
            memory_obj.tensor.copy_(temp_tensor, non_blocking=False)

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.
            k_cache_shape_per_layer = kv_caches[0][0].shape
            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            logger.info(f"Lazily initializing GPU buffer (max tokens={max_tokens}).")
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

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

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        current_stream = torch.cuda.current_stream()

        # Use streaming buffer for loading as well
        streaming_buffer = self._acquire_streaming_buffer()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync and self.load_stream is not None:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> streaming_buffer -> kvcaches
            if streaming_buffer is not None:
                # Use streaming buffer approach
                if self.load_stream is not None:
                    with torch.cuda.stream(self.load_stream):
                        self._load_layer_with_streaming_buffer(
                            layer_id,
                            memory_objs_layer,
                            starts,
                            ends,
                            kvcaches,
                            slot_mapping_full,
                            streaming_buffer,
                        )
                else:
                    # Use default stream
                    self._load_layer_with_streaming_buffer(
                        layer_id,
                        memory_objs_layer,
                        starts,
                        ends,
                        kvcaches,
                        slot_mapping_full,
                        streaming_buffer,
                    )
            else:
                # Direct CPU fallback for loading
                logger.warning(
                    f"No streaming buffer available for loading layer {layer_id}, "
                    f"using direct transfer"
                )
                self._load_layer_direct(
                    layer_id,
                    memory_objs_layer,
                    starts,
                    ends,
                    kvcaches,
                    slot_mapping_full,
                )

        yield

        # synchronize the last layer
        if sync and self.load_stream is not None:
            current_stream.wait_stream(self.load_stream)

        # Release streaming buffer
        if streaming_buffer is not None:
            self._release_streaming_buffer(streaming_buffer)

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    def _load_layer_with_streaming_buffer(
        self,
        layer_id,
        memory_objs_layer,
        starts,
        ends,
        kvcaches,
        slot_mapping_full,
        streaming_buffer,
    ):
        """Load a layer using streaming buffer."""
        total_tokens = sum(
            end - start for start, end in zip(starts, ends, strict=False)
        )

        if total_tokens <= self.streaming_buffer_size:
            # All fits in streaming buffer
            buffer_offset = 0
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                chunk_size = end - start
                streaming_buffer[buffer_offset : buffer_offset + chunk_size].copy_(
                    memory_obj.tensor, non_blocking=True
                )
                buffer_offset += chunk_size

            # Transfer full buffer to GPU
            lmc_ops.single_layer_kv_transfer(
                streaming_buffer[:total_tokens],
                kvcaches[layer_id][0],
                kvcaches[layer_id][1],
                slot_mapping_full,
                False,  # to_gpu=False (GPU->memory)
                True,  # is_layerwise=True
            )
        else:
            # Process in chunks
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                chunk_tokens = end - start

                # Process this memory object in streaming buffer-sized chunks
                mem_offset = 0
                while mem_offset < chunk_tokens:
                    chunk_end = min(
                        mem_offset + self.streaming_buffer_size, chunk_tokens
                    )
                    chunk_size = chunk_end - mem_offset

                    # Copy chunk to streaming buffer
                    streaming_buffer[:chunk_size].copy_(
                        memory_obj.tensor[mem_offset:chunk_end], non_blocking=True
                    )

                    # Create slot mapping for this chunk
                    chunk_slot_mapping = slot_mapping_full[
                        start + mem_offset : start + chunk_end
                    ]

                    # Transfer to GPU
                    lmc_ops.single_layer_kv_transfer(
                        streaming_buffer[:chunk_size],
                        kvcaches[layer_id][0],
                        kvcaches[layer_id][1],
                        chunk_slot_mapping,
                        False,  # to_gpu=False
                        True,  # is_layerwise=True
                    )

                    mem_offset = chunk_end

    def _load_layer_direct(
        self, layer_id, memory_objs_layer, starts, ends, kvcaches, slot_mapping_full
    ):
        """Direct loading without streaming buffer (fallback)."""
        for start, end, memory_obj in zip(
            starts, ends, memory_objs_layer, strict=False
        ):
            assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D

            # Create slot mapping for this chunk
            chunk_slot_mapping = slot_mapping_full[start:end]

            # Direct transfer from memory object to GPU
            lmc_ops.single_layer_kv_transfer(
                memory_obj.tensor,
                kvcaches[layer_id][0],
                kvcaches[layer_id][1],
                chunk_slot_mapping,
                False,  # to_gpu=False
                True,  # is_layerwise=True
            )

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Main entry point for batched GPU to memory transfer.
        Uses streaming buffer pool to handle requests of any size gracefully.
        This replaces dynamic allocation with fixed-size buffer reuse.
        """
        logger.debug("Starting batched_from_gpu with streaming buffer pool")

        # Delegate to streaming implementation
        yield from self.batched_from_gpu_streaming(memory_objs, starts, ends, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])


class SGLangGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )

        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )
            logger.info(f"GPU buffer: {self.gpu_buffer.shape}")

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        assert len(kv_caches) == self.num_kv_cache

        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        device = kv_caches[0].device
        assert device.type == "cuda", "The device should be CUDA."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_kv_cache, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        # sglang MLA kv_caches[0].shape: [num_pages * page_size, 1, head_size]
        # sglang MHA kv_caches[0].shape: [num_pages * page_size, num_heads, head_size]
        self.page_buffer_size = kv_caches[0].shape[0]
        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    f" order to be processed by {self.__class__.__name__}"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    f" order to be processed by {self.__class__.__name__}"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        offset = kwargs.get("offset", 0)

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)
        lmc_ops.multi_layer_kv_transfer_unilateral(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start - offset : end - offset],
            kvcaches[0][0].device,
            self.page_buffer_size,
            False,
            self.use_mla,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
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

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer_unilateral(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                True,
                self.use_mla,
            )
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0][0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
            lmc_ops.multi_layer_kv_transfer_unilateral(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                True,
                self.use_mla,
            )
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.cuda.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Yuwei): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

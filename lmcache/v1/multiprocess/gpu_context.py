# SPDX-License-Identifier: Apache-2.0
"""
GPU Cache Context management for LMCache multiprocessing.

This module provides GPU-side KV cache management functionality, including:
- GPUCacheContext: Manages shape and pointers to vLLM GPU KV cache tensors
- Helper functions for tensor operations and key resolution
"""

# Standard
import array
import threading
from typing import Any

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector.utils import (
    discover_gpu_kv_format,
    get_block_size,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_layers,
    get_num_heads,
    get_page_buffer_size,
    is_mla,
    is_sglang_mha,
)
from lmcache.v1.multiprocess.custom_types import (
    KVCache,
)

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: Any) -> Any:
    if isinstance(kv_caches, list):
        return [unwrap_kv_cache_tensors(kv_cache) for kv_cache in kv_caches]
    if isinstance(kv_caches, tuple):
        return tuple(unwrap_kv_cache_tensors(kv_cache) for kv_cache in kv_caches)
    return kv_caches.to_tensor()


def flatten_kv_cache_tensors(kv_caches: Any) -> list[torch.Tensor]:
    if isinstance(kv_caches, list | tuple):
        flattened: list[torch.Tensor] = []
        for kv_cache in kv_caches:
            flattened.extend(flatten_kv_cache_tensors(kv_cache))
        return flattened
    return [kv_caches]


def list_to_gpu_tensor(lis: list[int], device: torch.device) -> torch.Tensor:
    return torch.frombuffer(array.array("l", lis), dtype=torch.long).to(
        device, non_blocking=True
    )


class GPUCacheContext:
    """
    Manages the shape and pointers to vLLM GPU KV cache tensors.
    """

    def __init__(
        self,
        kv_caches: Any,
        lmcache_chunk_size: int = 256,
        engine_type: EngineType = EngineType.VLLM,
        block_size: int | None = None,
    ):
        self.kv_caches_ = unwrap_kv_cache_tensors(kv_caches)
        self.engine_type_ = engine_type
        self.flat_kv_caches_ = flatten_kv_cache_tensors(self.kv_caches_)
        self.device_ = self.flat_kv_caches_[0].device

        # Pointers
        pointers_list = [t.data_ptr() for t in self.flat_kv_caches_]
        self.kv_cache_pointers_ = list_to_gpu_tensor(pointers_list, self.device_)

        self.gpu_kv_format_ = discover_gpu_kv_format(self.kv_caches_, self.engine_type_)
        self.is_mla_ = is_mla(self.gpu_kv_format_)
        self.num_layers_ = get_num_layers(self.kv_caches_, self.gpu_kv_format_)
        self.page_buffer_size_ = get_page_buffer_size(self.kv_caches_, self.gpu_kv_format_)
        if block_size is None:
            block_size = get_block_size(self.kv_caches_, self.gpu_kv_format_)
        self.block_size_ = block_size
        if self.page_buffer_size_ % self.block_size_ != 0:
            raise ValueError(
                "page_buffer_size must be divisible by block_size, got "
                f"{self.page_buffer_size_} and {self.block_size_}"
            )
        try:
            expected_num_blocks = get_num_blocks(self.kv_caches_, self.gpu_kv_format_)
        except ValueError:
            expected_num_blocks = self.page_buffer_size_ // self.block_size_
        self.num_blocks_ = self.page_buffer_size_ // self.block_size_
        if expected_num_blocks != self.num_blocks_:
            raise ValueError(
                "Registration block_size does not match discovered KV cache "
                f"layout: expected {expected_num_blocks} blocks, "
                f"got {self.num_blocks_}"
            )
        self.hidden_dim_size_ = get_hidden_dim_size(
            self.kv_caches_, self.gpu_kv_format_
        )

        # Pre-computed slot mapping
        # shape: [num_blocks, block_size]
        block_ids = torch.arange(
            0, self.num_blocks_, dtype=torch.long, device=self.device_
        ).unsqueeze(1)
        offsets = torch.arange(
            0, self.block_size_, dtype=torch.long, device=self.device_
        ).unsqueeze(0)
        self.slot_mapping_tensor_ = (offsets + block_ids * self.block_size_).reshape(
            (self.num_blocks, self.block_size_)
        )

        # Temporary GPU buffer for transfers
        tmp_buffer_shape = self.get_kv_buffer_shape(lmcache_chunk_size)
        self.tmp_gpu_buffer_ = torch.empty(
            tmp_buffer_shape, dtype=self.dtype, device=self.device_
        )

        # Cuda streams
        self.cuda_stream_ = torch.cuda.Stream(device=self.device_)
        self.cupy_stream_ = cupy.cuda.ExternalStream(
            self.cuda_stream_.cuda_stream, self.device_.index
        )

        _, high_priority = torch.cuda.Stream.priority_range()
        self.high_priority_cuda_stream_ = torch.cuda.Stream(
            device=self.device_, priority=high_priority
        )
        self.high_priority_cupy_stream_ = cupy.cuda.ExternalStream(
            self.high_priority_cuda_stream_.cuda_stream, self.device_.index
        )

        # Per-device lock to serialise GPU↔CPU data transfers
        # on the same device without blocking transfers on other
        # devices.  Replaces the old global ``MPCacheEngine.lock``
        # to avoid deadlocks with the implicit CUDA driver lock.
        self.transfer_lock = threading.Lock()

        # Extra initialization
        self.cupy_stream_.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self.device_)
            ),
            logger,
        )

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype(self.kv_caches_, self.gpu_kv_format_)

    @property
    def gpu_kv_format(self):
        return self.gpu_kv_format_

    @property
    def device(self) -> torch.device:
        return self.device_

    @property
    def kv_tensors(self) -> Any:
        return self.kv_caches_

    @property
    def engine_type(self) -> EngineType:
        return self.engine_type_

    @property
    def kv_pointers(self) -> torch.Tensor:
        """
        Returns a GPU tensor of the KV cache pointers
        """
        return self.kv_cache_pointers_

    @property
    def stream(self) -> torch.cuda.Stream:
        """
        Returns the CUDA stream for KV cache operations
        """
        return self.cuda_stream_

    @property
    def cupy_stream(self) -> cupy.cuda.Stream:
        return self.cupy_stream_

    @property
    def high_priority_stream(self) -> torch.cuda.Stream:
        return self.high_priority_cuda_stream_

    @property
    def high_priority_cupy_stream(self) -> cupy.cuda.Stream:
        return self.high_priority_cupy_stream_

    @property
    def block_size(self) -> int:
        """
        Returns the block size (number of tokens per block)
        """
        return self.block_size_

    @property
    def num_layers(self) -> int:
        """
        Returns the number of layers in the model
        """
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        """
        Returns the number of blocks in the KV cache
        """
        return self.num_blocks_

    @property
    def hidden_dim_size(self) -> int:
        """
        Returns the hidden dimension size of the model
        """
        return self.hidden_dim_size_

    @property
    def is_mla(self) -> bool:
        """
        Returns whether the model uses MLA
        """
        return self.is_mla_

    @property
    def is_sglang_mha(self) -> bool:
        return is_sglang_mha(self.gpu_kv_format_)

    @property
    def page_buffer_size(self) -> int:
        return self.page_buffer_size_

    def get_layerwise_kv_tensors(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_sglang_mha:
            raise ValueError(
                "Layerwise KV tensor views are only available for "
                "the SGLang MHA GPUKVFormat"
            )
        assert isinstance(self.kv_caches_, list) and len(self.kv_caches_) == 2
        key_layers, value_layers = self.kv_caches_
        key_tensor = key_layers[layer_id]
        value_tensor = value_layers[layer_id]
        num_heads = get_num_heads(self.kv_caches_, self.gpu_kv_format_)
        head_size = get_head_size(self.kv_caches_, self.gpu_kv_format_)
        view_shape = (self.num_blocks_, self.block_size_, num_heads, head_size)
        return key_tensor.view(view_shape), value_tensor.view(view_shape)

    def get_tmp_gpu_buffer(self, num_tokens: int) -> torch.Tensor:
        """
        Returns the temporary GPU buffer for transfers
        """
        return self.tmp_gpu_buffer_[:, :, :num_tokens, :]

    @_lmcache_nvtx_annotate
    def get_slot_mapping_tensor(self, gpu_block_ids: list[int]) -> torch.Tensor:
        """
        Returns the slot mapping tensor for the KV cache on GPU
        """
        gpu_block_ids_tensor = list_to_gpu_tensor(gpu_block_ids, self.device_)
        return self.slot_mapping_tensor_[gpu_block_ids_tensor].flatten().contiguous()

    def get_kv_buffer_shape(self, num_tokens: int) -> torch.Size:
        """
        Returns the shape of the KV buffer for the given number of tokens
        """
        if self.is_mla_:
            return torch.Size((1, self.num_layers_, num_tokens, self.hidden_dim_size_))
        else:
            return torch.Size((2, self.num_layers_, num_tokens, self.hidden_dim_size_))


class PlainGPUCacheContext:
    """
    A plain GPU cache context that have a single contiguous 2LTD buffer
    """

    def __init__(self, kv_caches: KVCache, lmcache_chunk_size: int = 256):
        assert len(kv_caches) == 1, (
            "PlainGPUCacheContext only supports a single KV cache tensor"
        )

        # KV cache basics
        self._kv_cache = unwrap_kv_cache_tensors(kv_caches)[0]
        self._device = self._kv_cache.device

        # Shape related
        shape = self._kv_cache.shape
        assert len(shape) == 4, "Expected [2, L, T, D] for plain GPU cache"

        self._num_layers = shape[1]
        self._num_tokens = shape[2]
        self._hidden_dim_size = shape[3]

        # Temporary buffer
        tmp_buffer_shape = self.get_kv_buffer_shape(lmcache_chunk_size)
        self._tmp_gpu_buffer = torch.empty(
            tmp_buffer_shape, dtype=self.dtype, device=self.device
        )

        # Cuda streams
        self._cuda_stream = torch.cuda.Stream(device=self._device)
        self._cupy_stream = cupy.cuda.ExternalStream(
            self._cuda_stream.cuda_stream, self._device.index
        )

        _, high_priority = torch.cuda.Stream.priority_range()
        self._high_priority_cuda_stream = torch.cuda.Stream(
            device=self._device, priority=high_priority
        )
        self._high_priority_cupy_stream = cupy.cuda.ExternalStream(
            self._high_priority_cuda_stream.cuda_stream, self._device.index
        )

        # Extra initialization
        self._cupy_stream.launch_host_func(
            lambda logger: logger.info(
                "Initialized cuda stream on device %s", str(self._device)
            ),
            logger,
        )

    def get_kv_buffer_shape(self, num_tokens: int) -> torch.Size:
        """
        Returns the shape of the KV buffer for the given number of tokens
        """
        return torch.Size((2, self._num_layers, num_tokens, self._hidden_dim_size))

    def get_tmp_gpu_buffer(self, num_tokens: int) -> torch.Tensor:
        """
        Returns the temporary GPU buffer for transfers
        """
        return self._tmp_gpu_buffer[:, :, :num_tokens, :]

    def slice_kv_cache_on_tokens(self, start: int, end: int) -> torch.Tensor:
        """
        Slices the KV cache tensor on the token dimension
        """
        return self._kv_cache[:, :, start:end, :]

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_cache.dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def stream(self) -> torch.cuda.Stream:
        return self._cuda_stream

    @property
    def cupy_stream(self) -> cupy.cuda.Stream:
        return self._cupy_stream

    @property
    def high_priority_stream(self) -> torch.cuda.Stream:
        return self._high_priority_cuda_stream

    @property
    def high_priority_cupy_stream(self) -> cupy.cuda.Stream:
        return self._high_priority_cupy_stream

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def hidden_dim_size(self) -> int:
        return self._hidden_dim_size

    @property
    def kv_cache_tensor(self) -> torch.Tensor:
        return self._kv_cache

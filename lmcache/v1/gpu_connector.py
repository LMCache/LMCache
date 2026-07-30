# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, Union
import abc
import logging
import os

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.memory_management import GPUMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotary embedding helper: [-x2, x1, -x4, x3, ...]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _mrope_delta_rotate_k(
    k: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply DELTA rotation to cached K for mRoPE models.

    Unlike the fused kernel (which un-rotates from old_pos then re-rotates to
    new_pos), this function directly applies rotation by (new_pos - old_pos).
    For mRoPE, the fused un-rotate/re-rotate approach fails because the
    per-token 1D position doesn't match the per-section mRoPE positions used
    during original encoding.  However, the per-axis mRoPE base offsets shift
    uniformly, so a direct delta rotation is correct.

    k: (num_tokens, num_kv_heads * head_size)
    old_positions, new_positions: (num_tokens,) -- 1D sequential positions
    cos_sin_cache: (max_position, rotary_dim) from model's RotaryEmbedding
    head_size: per-head dimension
    """
    num_tokens = k.shape[0]
    num_kv_heads = k.shape[1] // head_size
    rotary_dim = cos_sin_cache.shape[-1]
    rot_dim_half = rotary_dim // 2

    delta = new_positions - old_positions  # (T,)
    abs_delta = delta.abs().clamp(max=cos_sin_cache.shape[0] - 1)

    cs = cos_sin_cache[abs_delta]  # (T, rotary_dim)
    cos_d = cs[:, :rot_dim_half]
    sin_d = cs[:, rot_dim_half:]

    neg_mask = (delta < 0).unsqueeze(-1)
    sin_d = torch.where(neg_mask, -sin_d, sin_d)

    cos_full = torch.cat([cos_d, cos_d], dim=-1)  # (T, rotary_dim)
    sin_full = torch.cat([sin_d, sin_d], dim=-1)

    k_view = k.view(num_tokens, num_kv_heads, head_size)
    k_rot = k_view[..., :rotary_dim]
    k_pass = k_view[..., rotary_dim:]

    cos_e = cos_full.unsqueeze(1)
    sin_e = sin_full.unsqueeze(1)

    k_rotated = k_rot * cos_e + _rotate_half(k_rot) * sin_e

    if k_pass.shape[-1] > 0:
        out = torch.cat([k_rotated, k_pass], dim=-1)
    else:
        out = k_rotated
    return out.view(num_tokens, -1)


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
    def batched_to_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched store the data from the memory objects to GPU kv cache.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data to GPU.
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

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]


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

        self.kvcaches: Optional[List[torch.Tensor]] = None

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

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        idx = self.device.index
        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]
        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
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

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

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

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.device,
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

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        with torch.cuda.stream(self.store_stream):
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.gpu_buffer.device == self.kvcaches[0].device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.multi_layer_kv_transfer(
                    tmp_gpu_buffer,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                )
                memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(self.load_stream)
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "batched_to_gpu: evicted chunk (%d, %d); skipping", start, end,
                    )
                    continue
                self.to_gpu(memory_obj, start, end, **kwargs)
        end_event.record(self.load_stream)
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        logger.debug("batched_to_gpu cost %.3f ms", elapsed_ms)

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

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # TODO(Jiayi): remove this hardcode
        self.cache_positions = True

        self.fused_rotary_emb = None

        assert use_gpu, "use_gpu must be true in VLLMBufferLayerwiseGPUConnector"
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.rope_stream = torch.cuda.Stream()
        self.to_page_mem_stream = torch.cuda.Stream()

        self.load_done_event_list = []
        self.rope_done_event_list = []
        for _ in range(self.num_layers):
            self.load_done_event_list.append(torch.cuda.Event())
            self.rope_done_event_list.append(torch.cuda.Event())    

        self.buffer_mapping: dict[int, MemoryObj] = {}

        # track gap positions between blended chunks
        self.current_gap_positions = None

        # Measurement (LMCACHE_BLEND_TIMING): count global device barriers issued
        # by batched_to_gpu. Serial per-request fetch => N*num_layers barriers per
        # step; a coalesced fetch collapses this to num_layers. The adapter reads
        # the delta across the fetch phase. Never gates behavior.
        self.global_sync_count = 0
        self.use_gpu = use_gpu
        self.gpu_buffer_allocator = None
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

    def get_page_stream(self):
        return self.to_page_mem_stream
    
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

            # flash attention: [num_layers, 2, num_blocks, block_size,
            # num_heads, head_size]
            # flash infer: [num_layers, num_blocks, 2, block_size, num_heads, head_size]
            assert kv_caches[0].shape[0] == 2 or kv_caches[0].shape[1] == 2, (
                "The kv_caches should have shape [num_layers, 2, num_blocks, "
                "block_size, num_heads, head_size] or "
                "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
            )

            self.vllm_two_major = kv_caches[0].shape[0] == 2

            if self.vllm_two_major:
                k_cache_shape_per_layer = kv_caches[0][0].shape
            else:
                k_cache_shape_per_layer = kv_caches[0][:, 0].shape
            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]

            logger.info(f"Lazily initializing GPU buffer (max tokens={max_tokens}).")
            num_elements = k_cache_shape_per_layer.numel() * 2
            # B1 (LMCACHE_COALESCED_FETCH=1): batched_to_gpu_multi needs TWO
            # buffers alive at once, each sized to the SUM of every packed
            # request's tokens (not one request's own span, like the old
            # per-request path) -- so a burst of up to MAXSEQS large windows
            # coalesced together can need far more than this pool's default
            # size, which was never chosen with that in mind (job 15005984:
            # AssertionError: Failed to allocate GPU buffer -> EngineDeadError
            # -> every subsequent request failed). Bounded, deterministic fix:
            # scale the pool by a multiplier tied to the KNOWN MAXSEQS cap
            # (vLLM already never schedules more concurrent requests than
            # that). Default 1 = byte-identical to before unless explicitly
            # set. The adapter-side pre-check in _batched_blend_load_kv is the
            # defense-in-depth for when the sizing assumption behind the
            # chosen multiplier is violated (e.g. an unexpectedly large
            # window) -- it falls back to per-request fetch instead of
            # crashing, rather than relying on this multiplier alone.
            buffer_mult = float(os.environ.get("LMCACHE_COALESCED_BUFFER_MULT", "1"))
            gpu_buffer_size = int(num_elements * self.element_size * buffer_mult)
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )
            # Pool capacity, in token-equivalents, is `max_tokens * buffer_mult`
            # (gpu_buffer_size scales linearly with max_tokens by construction
            # above); a coalesced call needs TWO buffers of `num_all_tokens`
            # alive at once, so `num_all_tokens` is only safe up to half of
            # that. The adapter reads this to decide whether to take the
            # coalesced path for a given pending group, instead of hardcoding
            # a copy of this arithmetic (or an assumed dataset-specific
            # ceiling) on the adapter side.
            self.max_coalesced_tokens = int(max_tokens * buffer_mult / 2)
            logger.info(
                f"GPU buffer pool: max_tokens={max_tokens} buffer_mult="
                f"{buffer_mult} -> max_coalesced_tokens={self.max_coalesced_tokens}"
            )

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV cache for the given layer ID.
        This function is used to get the KV cache from the GPU buffer.
        """
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")

        gpu_buffer = self.buffer_mapping[layer_id].tensor
        assert gpu_buffer is not None
        return gpu_buffer[0], gpu_buffer[1]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # B1: coalesced multi-request layerwise fetch (LMCACHE_COALESCED_FETCH=1)
    # ------------------------------------------------------------------
    @staticmethod
    def _coalesced_layout(req_starts, req_ends):
        """Index arithmetic for the coalesced staging buffer. PURE + CPU-testable.

        Today each request gets its OWN batched_to_gpu generator, hence its own
        pair of staging buffers and its own device barrier per layer:
        ``fetch_syncs = num_layers * batchN`` (measured 48/96/192 at batchN=1/2/4).
        B1 concatenates all requests into ONE buffer so a layer costs ONE store,
        ONE barrier and ONE RoPE call regardless of N -- the num_layers*N ->
        num_layers collapse. Justification (job 15005032): fetch grew 4.70x for a
        5x batch (near-LINEAR, does NOT amortize) while recompute grew 3.68x, so
        fetch is the phase cross-request batching provably cannot help.

        Request r occupies buffer rows [off_r, off_r + n_r) where n_r is its
        contiguous span ends_r[-1] - starts_r[0]. Correctness rests on the
        POSITIONAL 1:1 correspondence between buffer rows and slot_mapping
        entries -- the single-request path allocates get_shape(ends[-1]-starts[0])
        and passes slot_mapping[starts[0]:ends[-1]], same length and same order --
        so concatenating both in the same order preserves it.

        Returns (num_all_tokens, segments, chunk_slices, gap_ranges):
          segments     [(buf_off, src_start, length)]  one per request
          chunk_slices [(bs, be)] one per (request, chunk), in the EXACT order the
                       caller must send memory objects for each layer
          gap_ranges   [(bs, be)] buffer rows covered by NO chunk (zeroed post-RoPE)
        """
        segments, chunk_slices, gap_ranges = [], [], []
        off = 0
        for starts, ends in zip(req_starts, req_ends):
            s0, e1 = starts[0], ends[-1]
            n_r = e1 - s0
            segments.append((off, s0, n_r))
            cursor = s0
            for s, e in zip(starts, ends):
                if s > cursor:                       # hole before this chunk
                    gap_ranges.append((off + cursor - s0, off + s - s0))
                chunk_slices.append((off + s - s0, off + e - s0))
                cursor = e
            if cursor < e1:                          # trailing hole
                gap_ranges.append((off + cursor - s0, off + e1 - s0))
            off += n_r
        return off, segments, chunk_slices, gap_ranges

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
        # Per-layer timing requires two extra full stream syncs per layer
        # (load_stream + current stream) just to read elapsed_time for a
        # debug log. Only pay that when DEBUG logging is actually enabled.
        timing = logger.isEnabledFor(logging.DEBUG)
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb
            self._is_mrope = getattr(self.lmc_model, "is_mrope", False)
            if self._is_mrope:
                logger.info("mRoPE model detected: will use mRoPE-aware delta rotation instead of 1D fused kernel.")
            elif self.fused_rotary_emb is not None:
                self.lmc_model.rope_cache_to_device(self.device)

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        # Generator-local staging state. Several retrieve_layer generators can be
        # advanced concurrently on this *shared* connector (e.g. multiple requests
        # stepped in lockstep by wait_for_layer_load at N>=4). A shared
        # self.buffer_mapping keyed by layer_id let them clobber and `del` each
        # other's entries -> "KeyError: <layer>" and a dead engine. Keep the
        # staging dict / gap positions local to each generator; still mirror them
        # onto self so get_kv() and the blender (serial blend path) read them as
        # before -- that path drains one generator at a time, so no race.
        buffer_mapping: dict = {}
        self.buffer_mapping = buffer_mapping

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        gap_positions = torch.where(gap_mask)[0]
        self.current_gap_positions = gap_positions

        buf_offset = starts[0]
        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kvcaches[0].device
            )

        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
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

        stream = torch.cuda.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kvcaches[0].device
            )
        for layer_id in range(self.num_layers + 2):
            store_events = None
            rope_events = None
            load_events = None
            if layer_id > 1:
                if timing:
                    store_start = torch.cuda.Event(enable_timing=True)
                    store_end = torch.cuda.Event(enable_timing=True)
                    store_start.record(stream)
                lmc_ops.single_layer_kv_transfer(
                    buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )
                if timing:
                    store_end.record(stream)
                    store_events = (store_start, store_end)
                del buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done.
                # Default: global barrier (batch-path safe). Async-overlap
                # prototype (#async): a global sync here would barrier the
                # whole device and defeat blend<->prefill overlap, so use
                # scoped cross-stream waits instead (the load_stream feeds
                # rope on `stream`; both directions covered for the 2-buffer
                # ping-pong). Validate output byte-identical when enabling.
                #
                # Fix E (LMCACHE_SCOPED_STREAM_SYNC=1): use those same scoped
                # waits on the EAGER path too. The `else` branch below is a FULL
                # DEVICE `torch.cuda.synchronize()` executed once per layer per
                # request -- measured at `fetch_syncs = num_layers * batchN` per
                # blend step (48/96/192 for batchN=1/2/4 on InternVL3-14B). The
                # scoped pair orders exactly the two streams that actually share
                # the ping-pong buffers, which is the same ordering guarantee
                # without stalling the whole device. Semantics are unchanged;
                # gate it so the claim can be proven with
                # `EQ_AXIS=scopedsync` before it becomes the default.
                if (
                    os.environ.get("VLLM_CODECSIGHT_ASYNC_OVERLAP", "0") == "1"
                    or os.environ.get("LMCACHE_BATCHED_BLEND_OVERLAP", "0") == "1"
                    or os.environ.get("LMCACHE_SCOPED_STREAM_SYNC", "0") == "1"
                ):
                    stream.wait_stream(self.load_stream)
                    self.load_stream.wait_stream(stream)
                else:
                    self.global_sync_count += 1
                    torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if timing:
                    rope_start = torch.cuda.Event(enable_timing=True)
                    rope_end = torch.cuda.Event(enable_timing=True)
                    rope_start.record(stream)
                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    if getattr(self, "_is_mrope", False):
                        rotary_emb = self.lmc_model.layers[0].self_attn.rotary_emb
                        compute_gpu_buffer_obj.tensor[0] = _mrope_delta_rotate_k(
                            compute_gpu_buffer_obj.tensor[0],
                            old_positions_full,
                            new_positions_full,
                            rotary_emb.cos_sin_cache,
                            rotary_emb.head_size,
                        )
                    else:
                        compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                            old_positions_full,
                            new_positions_full,
                            compute_gpu_buffer_obj.tensor[0],
                        )

                # gap zeroing after RoPE
                if gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, gap_positions] = 0.0

                buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                if timing:
                    rope_end.record(stream)
                    rope_events = (rope_start, rope_end)
                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                c = 0
                evicted_ranges = []
                with torch.cuda.stream(self.load_stream):
                    if timing:
                        load_start = torch.cuda.Event(enable_timing=True)
                        load_end = torch.cuda.Event(enable_timing=True)
                        load_start.record(self.load_stream)
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        c += 1
                        s = start - buf_offset
                        e = end - buf_offset
                        if memory_obj is None:
                            evicted_ranges.append((s, e))
                            continue
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None

                        load_gpu_buffer_obj.tensor[:, s:e].copy_(memory_obj.tensor, non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[s:e] = memory_obj.metadata.cached_positions
                    if timing:
                        load_end.record(self.load_stream)
                        load_events = (load_start, load_end)
                if evicted_ranges and layer_id == 0:
                    new_gap_indices = []
                    for gs, ge in evicted_ranges:
                        new_gap_indices.append(
                            torch.arange(gs, ge, device=gap_positions.device)
                        )
                    extra_gaps = torch.cat(new_gap_indices)
                    gap_positions = torch.cat(
                        [gap_positions, extra_gaps]
                    ).unique()
                    self.current_gap_positions = gap_positions
                    logger.warning(
                        "batched_to_gpu: %d evicted chunk(s) on layer %d; "
                        "added %d positions to gap mask",
                        len(evicted_ranges), layer_id, extra_gaps.numel(),
                    )

            elif layer_id == self.num_layers:
                yield

            if timing:
                # Ensure events on both streams are completed before timing.
                self.load_stream.synchronize()
                stream.synchronize()

                store_ms = (
                    store_events[0].elapsed_time(store_events[1])
                    if store_events is not None
                    else 0.0
                )
                rope_ms = (
                    rope_events[0].elapsed_time(rope_events[1])
                    if rope_events is not None
                    else 0.0
                )
                load_ms = (
                    load_events[0].elapsed_time(load_events[1])
                    if load_events is not None
                    else 0.0
                )
                total_ms = store_ms + rope_ms + load_ms

                logger.debug(
                    "batched_to_gpu iter=%d store_layer=%s rope_layer=%s load_layer=%s "
                    "store_ms=%.3f rope_ms=%.3f load_ms=%.3f total_ms=%.3f, c=%d",
                    layer_id,
                    str(layer_id - 2) if layer_id > 1 else "NA",
                    str(layer_id - 1) if 0 < layer_id <= self.num_layers else "NA",
                    str(layer_id) if layer_id < self.num_layers else "NA",
                    store_ms,
                    rope_ms,
                    load_ms,
                    total_ms,
                    c,
                )


        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # ------------------------------------------------------------------
    # B1: coalesced multi-request layerwise fetch (LMCACHE_COALESCED_FETCH=1,
    # default OFF). Correctness gated (jobs 15005761 + 15005982); measured
    # no latency benefit at N=8 (see BLEND_OPT_IMPLEMENTATION.md).
    # ------------------------------------------------------------------
    @_lmcache_nvtx_annotate
    def batched_to_gpu_multi(
        self,
        req_starts: List[List[int]],
        req_ends: List[List[int]],
        req_slot_mappings: List[torch.Tensor],
        **kwargs,
    ):
        """
        Multi-request counterpart of `batched_to_gpu`. Field-for-field the
        same generator, except it stages ALL requests in ONE buffer via
        `_coalesced_layout`, so each layer costs ONE store, ONE barrier, ONE
        RoPE call and ONE gap-zero regardless of request count N -- collapsing
        `fetch_syncs = num_layers * N` to `num_layers` (see `_coalesced_layout`
        docstring for the justification and the buffer-row <-> slot_mapping
        positional invariant this relies on).

        `req_starts[r]` / `req_ends[r]` are request r's own chunk starts/ends
        -- exactly what a lone `batched_to_gpu` call would receive as
        `starts`/`ends`. `req_slot_mappings[r]` is request r's OWN slot
        mapping tensor (its physical KV-cache block assignment) -- unlike
        `req_starts`/`req_ends`, these CANNOT be concatenated from one shared
        tensor: two different requests' token position `k` maps to unrelated
        physical slots, so each request contributes its own tensor, sliced by
        its own `(s0, n_r)` from `_coalesced_layout`'s `segments`. Per layer,
        the caller must `send()` a FLAT list of memory objects covering every
        request's chunks for that layer, in the exact order
        `_coalesced_layout(req_starts, req_ends)` returns as `chunk_slices`
        (request-major, then chunk order within each request).

        Yields `num_layers + 2` times, identical cadence to `batched_to_gpu`,
        so a caller stepping this generator needs no change to its
        step-counting versus stepping one per-request generator.
        """
        timing = logger.isEnabledFor(logging.DEBUG)
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        assert len(req_slot_mappings) == len(req_starts) == len(req_ends), (
            "batched_to_gpu_multi: req_slot_mappings must be parallel to "
            "req_starts/req_ends, one tensor per request."
        )

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb
            self._is_mrope = getattr(self.lmc_model, "is_mrope", False)
            if self._is_mrope:
                logger.info("mRoPE model detected: will use mRoPE-aware delta rotation instead of 1D fused kernel.")
            elif self.fused_rotary_emb is not None:
                self.lmc_model.rope_cache_to_device(self.device)

        self._lazy_initialize_buffer(self.kvcaches)

        # Generator-local staging state -- see the identical note in
        # `batched_to_gpu`: keep it local so concurrent generators on this
        # shared connector can't clobber each other; still mirror onto self
        # for get_kv() / the serial blend path.
        buffer_mapping: dict = {}
        self.buffer_mapping = buffer_mapping

        num_all_tokens, segments, chunk_slices, gap_ranges = self._coalesced_layout(
            req_starts, req_ends
        )

        # Each request's OWN slot mapping, sliced by its OWN (s0, n_r) --
        # request r's tokens [s0, s0+n_r) map into req_slot_mappings[r], never
        # into another request's tensor (see docstring: positions across
        # requests are unrelated, unlike the single-request case where one
        # shared tensor covers the whole span).
        slot_mapping_full = torch.cat(
            [
                req_slot_mappings[r][s0 : s0 + n_r]
                for r, (_, s0, n_r) in enumerate(segments)
            ],
            dim=0,
        )

        if gap_ranges:
            gap_positions = torch.cat(
                [
                    torch.arange(bs, be, device=slot_mapping_full.device)
                    for bs, be in gap_ranges
                ]
            )
        else:
            gap_positions = torch.empty(
                0, dtype=torch.long, device=slot_mapping_full.device
            )
        self.current_gap_positions = gap_positions

        if self.cache_positions:
            new_positions_full = torch.cat(
                [
                    torch.arange(
                        s0, s0 + n_r, dtype=torch.int64, device=self.kvcaches[0].device
                    )
                    for _, s0, n_r in segments
                ]
            )

        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
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

        stream = torch.cuda.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kvcaches[0].device
            )
        for layer_id in range(self.num_layers + 2):
            store_events = None
            rope_events = None
            load_events = None
            if layer_id > 1:
                if timing:
                    store_start = torch.cuda.Event(enable_timing=True)
                    store_end = torch.cuda.Event(enable_timing=True)
                    store_start.record(stream)
                lmc_ops.single_layer_kv_transfer(
                    buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )
                if timing:
                    store_end.record(stream)
                    store_events = (store_start, store_end)
                del buffer_mapping[layer_id - 2]

                logger.debug(
                    f"Finished loading layer {layer_id - 2} into paged memory (coalesced)"
                )

            if layer_id > 0 and layer_id <= self.num_layers:
                # See `batched_to_gpu` for the full rationale on this branch;
                # identical here since it's the same 2-buffer ping-pong, just
                # over the coalesced (multi-request) buffer.
                if (
                    os.environ.get("VLLM_CODECSIGHT_ASYNC_OVERLAP", "0") == "1"
                    or os.environ.get("LMCACHE_BATCHED_BLEND_OVERLAP", "0") == "1"
                    or os.environ.get("LMCACHE_SCOPED_STREAM_SYNC", "0") == "1"
                ):
                    stream.wait_stream(self.load_stream)
                    self.load_stream.wait_stream(stream)
                else:
                    self.global_sync_count += 1
                    torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if timing:
                    rope_start = torch.cuda.Event(enable_timing=True)
                    rope_end = torch.cuda.Event(enable_timing=True)
                    rope_start.record(stream)
                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    if getattr(self, "_is_mrope", False):
                        rotary_emb = self.lmc_model.layers[0].self_attn.rotary_emb
                        compute_gpu_buffer_obj.tensor[0] = _mrope_delta_rotate_k(
                            compute_gpu_buffer_obj.tensor[0],
                            old_positions_full,
                            new_positions_full,
                            rotary_emb.cos_sin_cache,
                            rotary_emb.head_size,
                        )
                    else:
                        compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                            old_positions_full,
                            new_positions_full,
                            compute_gpu_buffer_obj.tensor[0],
                        )

                # gap zeroing after RoPE
                if gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, gap_positions] = 0.0

                buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                if timing:
                    rope_end.record(stream)
                    rope_events = (rope_start, rope_end)
                logger.debug(
                    f"Finished loading layer {layer_id - 1} into buffer (coalesced)"
                )

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer. `chunk_slices` entries are already
                # buffer-relative (unlike `batched_to_gpu`'s starts/ends,
                # which need `- buf_offset`), and cover every request's
                # chunks for this layer in one flat, request-major list.
                c = 0
                evicted_ranges = []
                with torch.cuda.stream(self.load_stream):
                    if timing:
                        load_start = torch.cuda.Event(enable_timing=True)
                        load_end = torch.cuda.Event(enable_timing=True)
                        load_start.record(self.load_stream)
                    for (s, e), memory_obj in zip(
                        chunk_slices, memory_objs_layer, strict=False
                    ):
                        c += 1
                        if memory_obj is None:
                            evicted_ranges.append((s, e))
                            continue
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None

                        load_gpu_buffer_obj.tensor[:, s:e].copy_(memory_obj.tensor, non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[s:e] = memory_obj.metadata.cached_positions
                    if timing:
                        load_end.record(self.load_stream)
                        load_events = (load_start, load_end)
                if evicted_ranges and layer_id == 0:
                    new_gap_indices = []
                    for gs, ge in evicted_ranges:
                        new_gap_indices.append(
                            torch.arange(gs, ge, device=gap_positions.device)
                        )
                    extra_gaps = torch.cat(new_gap_indices)
                    gap_positions = torch.cat(
                        [gap_positions, extra_gaps]
                    ).unique()
                    self.current_gap_positions = gap_positions
                    logger.warning(
                        "batched_to_gpu_multi: %d evicted chunk(s) on layer %d; "
                        "added %d positions to gap mask",
                        len(evicted_ranges), layer_id, extra_gaps.numel(),
                    )

            elif layer_id == self.num_layers:
                yield

            if timing:
                # Ensure events on both streams are completed before timing.
                self.load_stream.synchronize()
                stream.synchronize()

                store_ms = (
                    store_events[0].elapsed_time(store_events[1])
                    if store_events is not None
                    else 0.0
                )
                rope_ms = (
                    rope_events[0].elapsed_time(rope_events[1])
                    if rope_events is not None
                    else 0.0
                )
                load_ms = (
                    load_events[0].elapsed_time(load_events[1])
                    if load_events is not None
                    else 0.0
                )
                total_ms = store_ms + rope_ms + load_ms

                logger.debug(
                    "batched_to_gpu_multi iter=%d store_layer=%s rope_layer=%s load_layer=%s "
                    "store_ms=%.3f rope_ms=%.3f load_ms=%.3f total_ms=%.3f, c=%d",
                    layer_id,
                    str(layer_id - 2) if layer_id > 1 else "NA",
                    str(layer_id - 1) if 0 < layer_id <= self.num_layers else "NA",
                    str(layer_id) if layer_id < self.num_layers else "NA",
                    store_ms,
                    rope_ms,
                    load_ms,
                    total_ms,
                    c,
                )

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(buffer_mapping) == 0, (
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

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

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
                        start, end, device=self.kvcaches[0].device, dtype=torch.int64
                    )
                )

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None
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
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    True,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
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
                        memory_obj.metadata.cached_positions = old_positions

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

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # All sizes are in bytes
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

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

            # flash attention: [num_layers, 2, num_blocks, block_size,
            # num_heads, head_size]
            # flash infer: [num_layers, num_blocks, 2, block_size, num_heads, head_size]
            assert kv_caches[0].shape[0] == 2 or kv_caches[0].shape[1] == 2, (
                "The kv_caches should have shape [num_layers, 2, num_blocks, "
                "block_size, num_heads, head_size] or "
                "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
            )

            self.vllm_two_major = kv_caches[0].shape[0] == 2

            if self.vllm_two_major:
                k_cache_shape_per_layer = kv_caches[0][0].shape
            else:
                k_cache_shape_per_layer = kv_caches[0][:, 0].shape
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

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj: Optional[MemoryObj] = (
                self.gpu_buffer_allocator.allocate(
                    buffer_shape, self.dtype, MemoryFormat.KV_T2D
                )
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    if memory_obj is None:
                        logger.warning(
                            "Layer %d, chunk (%d, %d) evicted; skipping",
                            layer_id, start, end,
                        )
                        continue
                    assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                    start_event.record(self.load_stream)
                    if self.use_gpu:
                        tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                            memory_obj.tensor, non_blocking=True
                        )
                    else:
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping_full,
                            False,
                            True,
                            self.vllm_two_major,
                        )
                    end_event.record(self.load_stream)
                    end_event.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)
                    logger.debug(
                        "Layer %d, chunk (%d, %d) transfer to GPU buffer cost %.3f ms",
                        layer_id,
                        start,
                        end,
                        elapsed_ms,
                    )
                start_event.record(self.load_stream)
                if self.use_gpu:
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        False,
                        True,
                        self.vllm_two_major,
                    )
                end_event.record(self.load_stream)
                end_event.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                logger.debug(
                    "Layer %d transfer from GPU buffer to paged memory cost %.3f ms",
                    layer_id,
                    elapsed_ms,
                )
        yield
        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if self.use_gpu:
            assert tmp_gpu_buffer_obj is not None
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
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

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj: Optional[MemoryObj] = (
                self.gpu_buffer_allocator.allocate(
                    buffer_shape, self.dtype, MemoryFormat.KV_T2D
                )
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                if self.use_gpu:
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        True,
                        True,
                        self.vllm_two_major,
                    )
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    assert memory_obj.tensor is not None
                    logger.debug(f"Memory obj device: {memory_obj.tensor.device}")
                    if self.use_gpu:
                        memory_obj.tensor.copy_(
                            tmp_gpu_buffer_obj.tensor[start - offset : end - offset],
                            non_blocking=True,
                        )
                    else:
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            True,
                            True,
                            self.vllm_two_major,
                        )

            yield
            if sync:
                self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        assert tmp_gpu_buffer_obj is not None
        tmp_gpu_buffer_obj.ref_count_down()
        yield

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
            if memory_obj is None:
                logger.warning(
                    "batched_to_gpu: evicted chunk (%d, %d); skipping", start, end,
                )
                continue
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Yuwei): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)


class SGLangLayerwiseGPUConnector(GPUConnectorInterface):
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
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        self.dtype = kwargs["dtype"]
        assert "device" in kwargs, "device should be provided to create a GPU buffer."
        self.device = kwargs["device"]

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )
        self.use_gpu = use_gpu
        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            k_cache_shape_per_layer = kv_caches[0][0].shape
            max_tokens = k_cache_shape_per_layer[0]
            logger.info(f"Lazily initializing GPU buffer (max tokens={max_tokens}).")
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
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

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        False,
                        True,
                    )

            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape

                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    False,
                    True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
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

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape
                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    True,
                    True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None
                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        True,
                        True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])

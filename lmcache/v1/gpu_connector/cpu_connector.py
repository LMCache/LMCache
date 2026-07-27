# SPDX-License-Identifier: Apache-2.0
"""
Pure-PyTorch CPU KV connector for vLLM.

This is the CPU counterpart of ``VLLMPagedMemGPUConnectorV2``. The GPU/XPU/HPU
connectors move KV between vLLM's paged buffer and an LMCache ``MemoryObj`` via a
compiled CUDA/SYCL kernel (``lmcache.c_ops.multi_layer_kv_transfer``) driven by
``torch.cuda.Stream``. Neither the kernel nor CUDA streams exist on a CPU-only
build, so this connector reimplements the same gather/scatter with plain torch
advanced indexing — no native extension, no streams, no device pointers.

On CPU, ``lmcache.c_ops`` resolves to the pure-torch ``CpuDeviceOps`` baseline
(there is no CUDA/SYCL kernel to dispatch to), so this connector does the
gather/scatter itself with torch advanced indexing instead of a device kernel.

Supported KV layout: vLLM non-MLA per-layer caches, i.e. a list of per-layer
tensors. vLLM's CPU attention backend uses HND physical layout
``[2, num_blocks, num_heads, block_size, head_size]``; NHD
``[2, num_blocks, block_size, num_heads, head_size]`` is also handled.

MemoryObj format produced/consumed: ``KV_2LTD`` = ``[2, num_layers, T, NH*HS]``.
"""

# Standard
from typing import TYPE_CHECKING, Any, List, Tuple, cast
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.types import LayoutHints

logger = init_logger(__name__)


class VLLMCPUConnector(GPUConnectorInterface):
    """CPU gather/scatter between vLLM paged KV and an LMCache MemoryObj."""

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_mla: bool = False,
        kv_layout: str | None = None,
        head_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_mla = use_mla
        # head_size splits vLLM's fused-KV CPU layout (last dim = 2*head_size).
        # Optional: when unset, _kv_token_views infers it from the cache shape.
        self.head_size = head_size
        # Physical KV layout precedence: explicit kv_layout argument (direct
        # construction only -- from_metadata forces None on CPU because vLLM
        # misreports its layout there), then the LMCACHE_CPU_KV_LAYOUT env
        # override, then the vLLM CPU attention backend default (HND).
        resolved_layout = kv_layout or os.environ.get("LMCACHE_CPU_KV_LAYOUT") or "HND"
        self.kv_layout = resolved_layout.upper()
        if self.kv_layout not in ("HND", "NHD"):
            raise ValueError(
                f"Unsupported kv_layout '{resolved_layout}'; expected 'HND' or "
                "'NHD' (case-insensitive)."
            )
        self.kvcaches: List[torch.Tensor] | None = None
        self._logged = False

        if self.use_mla:
            raise NotImplementedError(
                "VLLMCPUConnector does not support MLA models yet."
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: torch.device | None = None,
        layout_hints: "LayoutHints | None" = None,
    ) -> "VLLMCPUConnector":
        """Build a connector from LMCache engine metadata.

        Args:
            metadata: Engine metadata. ``kv_shape`` is
                ``(num_layers, 2 or 1, chunk_size, num_kv_head, head_size)`` and
                ``use_mla`` selects MLA (unsupported).
            use_gpu: Unused on CPU; accepted for interface parity.
            device: Unused on CPU; accepted for interface parity.
            layout_hints: Ignored on CPU -- vLLM's CPU attention backend
                misreports its layout, so HND is forced (mirrors the KV-format
                detector). Accepted only for interface compatibility.

        Returns:
            A configured :class:`VLLMCPUConnector`.
        """
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        # vLLM's CPU attention backend stores KV in HND but misreports its
        # layout, so the KV-format detector forces HND on CPU (see
        # kv_format/detectors/vllm.py). Mirror that: ignore layout_hints and let
        # __init__ resolve the env override (LMCACHE_CPU_KV_LAYOUT) or the HND
        # default. layout_hints is accepted only for interface compatibility.
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_mla=metadata.use_mla,
            kv_layout=None,
            head_size=head_size,
        )

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    def initialize_kvcaches_ptr(self, **kwargs: Any) -> None:
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]

    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """STORE: gather vLLM paged KV (tokens ``[start, end)``) into memory_obj.

        Args:
            memory_obj: Destination LMCache buffer; filled and tagged KV_2LTD.
            start: Start token index (inclusive) within this chunk.
            end: End token index (exclusive) within this chunk.
            **kwargs: Expected keys:
                ``kvcaches`` (list[torch.Tensor]): the vLLM paged KV caches.
                ``slot_mapping`` (torch.Tensor[int]): per-token slot ids;
                negative values are padding sentinels and are skipped.

        Raises:
            ValueError: If ``memory_obj.tensor`` is None or ``slot_mapping`` is
                not provided.
            RuntimeError: If ``kvcaches`` were not provided or initialized.
        """
        if memory_obj.tensor is None:
            raise ValueError("memory_obj.tensor must not be None")
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise RuntimeError(
                "kvcaches are not initialized; pass 'kvcaches' via kwargs"
            )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided")
        self._log_once()

        slot_mapping = kwargs["slot_mapping"]
        mt = memory_obj.tensor
        T = end - start
        k0, _ = self._kv_token_views(0)
        bs = k0.shape[1]
        block, off, valid = self._block_offsets(slot_mapping, start, end, bs)
        all_valid = bool(valid.all())

        for layer in range(self.num_layers):
            kview, vview = self._kv_token_views(layer)
            g_k = kview[block, off]  # [T, NH, HS] (clamped; invalid rows are bogus)
            g_v = vview[block, off]
            if not all_valid:
                # advanced indexing above already returns fresh copies, so
                # zeroing sentinel/padding rows here does not touch the cache.
                g_k[~valid] = 0
                g_v[~valid] = 0
            mt[0, layer, :T, :].copy_(g_k.reshape(T, -1))
            mt[1, layer, :T, :].copy_(g_v.reshape(T, -1))

        memory_obj.metadata.fmt = MemoryFormat.KV_2LTD

    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """LOAD: scatter memory_obj KV back into the vLLM paged buffer.

        Args:
            memory_obj: Source LMCache buffer; must be in KV_2LTD format.
            start: Start token index (inclusive) within this chunk.
            end: End token index (exclusive) within this chunk.
            **kwargs: Expected keys:
                ``kvcaches`` (list[torch.Tensor]): the vLLM paged KV caches.
                ``slot_mapping`` (torch.Tensor[int]): per-token slot ids;
                negative values are padding sentinels and are skipped.
                ``vllm_cached_tokens`` (int, optional): count of leading tokens
                already present in vLLM's paged cache; those are not rewritten.

        Raises:
            ValueError: If ``memory_obj.tensor`` is None, ``slot_mapping`` is not
                provided, or the memory object is not in KV_2LTD format.
            RuntimeError: If ``kvcaches`` were not provided or initialized.
        """
        if memory_obj.tensor is None:
            raise ValueError("memory_obj.tensor must not be None")
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise RuntimeError(
                "kvcaches are not initialized; pass 'kvcaches' via kwargs"
            )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided")
        if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
            raise ValueError(
                "memory object must be in KV_2LTD format for VLLMCPUConnector"
            )
        self._log_once()

        slot_mapping = kwargs["slot_mapping"]
        mt = memory_obj.tensor
        T = end - start

        # Skip tokens already present in vLLM's own paged cache (matches the
        # CUDA path's skip_prefix_n_tokens), avoids redundant writes.
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip = min(T, max(0, vllm_cached - start))

        k0, _ = self._kv_token_views(0)
        bs = k0.shape[1]
        block, off, valid = self._block_offsets(slot_mapping, start, end, bs)
        # only scatter real positions (skip prefix + sentinel/padding slots)
        sel = valid.clone()
        sel[:skip] = False
        b_sel, o_sel = block[sel], off[sel]
        # row indices into mt for the selected (valid, non-skipped) tokens
        rows = torch.nonzero(sel, as_tuple=True)[0]

        for layer in range(self.num_layers):
            kview, vview = self._kv_token_views(layer)
            nh, hs = kview.shape[2], kview.shape[3]
            kview[b_sel, o_sel] = mt[0, layer, rows, :].reshape(-1, nh, hs)
            vview[b_sel, o_sel] = mt[1, layer, rows, :].reshape(-1, nh, hs)

    def batched_to_gpu(
        self,
        memory_objs: List[List[MemoryObj]] | List[MemoryObj] | List[int] | None = None,
        starts: List[int] | None = None,
        ends: List[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if memory_objs is None or starts is None or ends is None:
            return
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)  # type: ignore[arg-type]

    def batched_from_gpu(
        self,
        memory_objs: List[List[MemoryObj]] | List[MemoryObj],
        starts: List[int],
        ends: List[int],
        **kwargs: Any,
    ) -> None:
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)  # type: ignore[arg-type]

    # ---- private helpers -----------------------------------------------------

    def _kv_token_views(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (k_view, v_view), each a [NB, BS, NH, HS] view of the layer's
        paged cache, indexable as ``view[block_idx, off_idx] -> [.., NH, HS]``.
        Views alias storage so assignment writes through.

        Operates on the *raw* vLLM CPU paged caches (not detector-normalized
        ``EngineKVFormat`` tensors), so K/V is split on the leading dim (5-D) or
        the last dim (4-D fused) as vLLM's CPU backend lays them out.

        Handles two vLLM CPU layouts:
          * 5-D ``[2, NB, NH, BS, HS]`` (HND) / ``[2, NB, BS, NH, HS]`` (NHD):
            K/V split on the leading dim.
          * 4-D ``[NB, NH, BS, 2*HS]`` (HND) / ``[NB, BS, NH, 2*HS]`` (NHD):
            vLLM's CPU attention backend fuses K and V into the last dim
            (first head_size = K, last head_size = V).
        """
        if self.kvcaches is None:
            raise RuntimeError("kvcaches not initialized")
        # Recover the physical layout via the shared contiguity helper (single
        # source of truth); a no-op when the tensor is already contiguous.
        t = cast(torch.Tensor, attempt_permute_to_contiguous_view(self.kvcaches[layer]))
        if t.dim() == 5:
            k, v = t[0], t[1]
        elif t.dim() == 4 and t.shape[-1] % 2 == 0:
            # Fused K/V on the last dim ([..., 2*HS]); infer HS from the shape so
            # this does not depend on head_size being supplied at construction.
            hs = t.shape[-1] // 2
            if self.head_size is not None and hs != self.head_size:
                raise ValueError(
                    f"4-D fused cache last dim {t.shape[-1]} implies head_size="
                    f"{hs}, but connector was created with head_size="
                    f"{self.head_size}"
                )
            k, v = t[..., :hs], t[..., hs:]
        else:
            raise ValueError(
                f"Unexpected CPU KV cache shape {tuple(t.shape)}; expected 5-D "
                f"[2,NB,NH,BS,HS] or 4-D [NB,NH,BS,2*HS]"
            )
        if self.kv_layout == "HND":
            # [NB, NH, BS, HS] -> [NB, BS, NH, HS]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        # NHD already [NB, BS, NH, HS]
        return k, v

    def _block_offsets(
        self, slot_mapping: torch.Tensor, start: int, end: int, bs: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Negative slots are sentinels (PAD_SLOT_ID == -1, prefix-cache padding).
        # The CUDA kernel and python fallback SKIP them; we must too. Without a
        # guard, slot<0 makes block/off negative which silently wraps to the last
        # block/offset in torch indexing -> reads/writes the WRONG KV. We build a
        # validity mask and clamp indices to 0 so advanced indexing stays in-bounds;
        # callers must only touch positions where `valid` is True.
        slots = slot_mapping[start:end].to(torch.int64)
        valid = slots >= 0
        safe = slots.clamp(min=0)
        block = torch.div(safe, bs, rounding_mode="floor")
        off = safe - block * bs
        return block, off, valid

    def _log_once(self) -> None:
        if self._logged:
            return
        self._logged = True
        if self.kvcaches is None:
            raise RuntimeError("kvcaches not initialized")
        t = self.kvcaches[0]
        ct = cast(torch.Tensor, attempt_permute_to_contiguous_view(t))
        logger.debug(
            "VLLMCPUConnector active: layers=%d hidden=%d layout=%s "
            "layer0 raw shape=%s stride=%s -> view shape=%s",
            self.num_layers,
            self.hidden_dim_size,
            self.kv_layout,
            tuple(t.shape),
            tuple(t.stride()),
            tuple(ct.shape),
        )

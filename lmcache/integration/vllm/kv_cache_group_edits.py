# SPDX-License-Identifier: Apache-2.0
"""Centralized edits to vLLM kv cache specs.

This is needed to mask out attention-specific details while making sure that
LMCache can still store / load KV cache correctly.

The registry holds structural view edits, one per
:class:`KVCacheGroupEdit` subclass below. Rules cover Mamba state pages and
attention tensors whose kernel paging differs from the scheduler's logical
block granularity, in fused, split-plane, and MLA forms; the registry is
consulted whenever a
``KVCacheConfig`` is available, and each rule decides independently
whether it applies. :func:`validate_kv_cache_groups`
additionally rejects, at startup, group specs the transfer path cannot serve
correctly yet (see its docstring).

Reference design: vLLM PR #42828 ("[KVConnector][DSV4] HMA support for
Mooncake store connector") solves the same problem class for an external KV
store. Check it again when working on any of the following:

- Extending :func:`validate_kv_cache_groups` (its connector validates and
  rejects unsupported specs up front, with one aggregated error).
- Per-group store/load masks (SWA / Mamba tail-only transfer, the
  "sliding-window load-plan trimming" deferred in the hybrid design doc):
  its ``MooncakeStoreCoordinator.store_mask`` / ``load_mask`` derive masks
  from vLLM's own per-spec managers. Requires per-group objects in LMCache
  (``ObjectKey.object_group_id``, LMCache #3608).
- Hit-length computation for hybrid models: its ``ExternalCachedBlockPool``
  duck-types vLLM's ``BlockPool`` over a ``(group_id, hash)`` existence set
  and reuses ``KVCacheSpecRegistry`` manager classes, so promised hits are
  consumable by construction. NOTE: LMCache's lookup doubles as prefetch, so
  vLLM's lookup-first-then-trim flow does not map directly -- this needs its
  own design before borrowing.
- Eagle + HMA: see its ``apply_eagle`` notes (the eagle last-block prune must
  be applied exactly once between hit-length and mask computation).
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from typing import TypeAlias

# Third Party
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.gpu_connector.kv_format.types import KV_LAYOUT_NAMES
from lmcache.v1.gpu_connector.utils import LayoutHints

logger = init_logger(__name__)

# One registered cache value: a paged KV tensor, [conv_state, ssm_state] for
# Mamba layers, or (k_plane, v_plane) on split attention caches.
RegisteredKVCache: TypeAlias = (
    torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
)

# Synthetic head count for a reinterpreted page. The page is opaque bytes, so
# one "head" holding the whole per-(K/V) slab is enough; head_size is derived
# to fill the page.
_SYNTHETIC_NUM_HEADS = 1

# Standard-paged (non-MLA) attention kinds eligible for the sub-paged edit.
_SUBPAGEABLE_ATTENTION_KINDS = frozenset(
    {
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.SLIDING_WINDOW,
        KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION,
        KVCacheSpecKind.SINK_FULL_ATTENTION,
    }
)


def _declares_slot_compression(spec: KVCacheSpec) -> bool:
    """Return whether a spec declares slot compression (must not be edited).

    Covers ``tokens_per_state > 1`` (DeepSeek-V4 slot packing; ``compress_ratio``
    on older vLLM) and ``TQFullAttentionSpec.tq_slot_size > 0`` (TurboQuant
    slots); such groups belong to the compression path in
    ``lmcache.v1.kv_layer_groups``.
    """
    return (
        getattr(spec, "tokens_per_state", 1) > 1
        or getattr(spec, "compress_ratio", 1) > 1
        or getattr(spec, "tq_slot_size", 0) > 0
    )


def _tokens_per_state(spec: KVCacheSpec) -> int:
    """Return the declared number of tokens packed into one stored state.

    ``tokens_per_state`` on current vLLM; older specs declare the packing
    as ``compress_ratio``, or via ``num_states``, where it is
    ``block_size / num_states``.
    """
    tokens_per_state = getattr(spec, "tokens_per_state", None)
    if tokens_per_state is not None:
        return tokens_per_state
    num_states = getattr(spec, "num_states", None)
    if num_states is not None and num_states > 0:
        return max(1, spec.block_size // num_states)
    return max(1, getattr(spec, "compress_ratio", 1))


def _leaf_specs(spec: KVCacheSpec) -> list[KVCacheSpec]:
    """Return a group spec's leaf specs, unwrapping ``UniformTypeKVCacheSpecs``."""
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict):
        return list(inner.values())
    return [spec]


def validate_kv_cache_groups(kv_cache_config: KVCacheConfig | None) -> None:
    """Reject KV cache group specs the transfer path cannot serve correctly.

    Rejected, with one aggregated error listing every offending group:

    - ``CrossAttentionSpec`` (encoder-decoder caches).
    - Mamba groups with ``mamba_cache_mode`` other than ``"align"`` or
      ``"all"``: the remaining mode (``"none"``) keeps no reusable per-block
      state snapshots.

    Specs declaring slot compression (``compress_ratio > 1`` /
    ``tq_slot_size > 0``, e.g. DeepSeek-V4) are NOT rejected: they are served
    by the compression path in ``lmcache.v1.kv_layer_groups`` and merely
    skipped by the edits here (see ``_declares_slot_compression``).

    Args:
        kv_cache_config: vLLM ``KVCacheConfig``; ``None`` skips validation
            (callers without the config validate again at registration).

    Raises:
        ValueError: Listing every unsupported group and why.
    """
    if kv_cache_config is None:
        return
    unsupported: list[str] = []
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        for spec in _leaf_specs(group.kv_cache_spec):
            kind = get_kv_cache_spec_kind(spec)
            if kind == KVCacheSpecKind.CROSS_ATTENTION:
                unsupported.append(f"group {group_idx}: CrossAttentionSpec")
            elif kind == KVCacheSpecKind.MAMBA and getattr(
                spec, "mamba_cache_mode", "none"
            ) not in ("align", "all"):
                unsupported.append(
                    f"group {group_idx}: MambaSpec with mamba_cache_mode="
                    f"'{getattr(spec, 'mamba_cache_mode', 'none')}' "
                    f"(only 'align' and 'all' keep reusable state snapshots)"
                )
    if unsupported:
        raise ValueError(
            "LMCache cannot serve this model's KV cache groups: "
            + "; ".join(unsupported)
            + ". See lmcache/integration/vllm/kv_cache_group_edits.py."
        )


def _synthetic_attention_shape(elems_per_page: int, block_size: int) -> tuple[int, int]:
    """Factor a page's element count into the synthetic attention layout.

    Args:
        elems_per_page: Total elements in one page (one logical block).
        block_size: Logical block size (tokens per page).

    Returns:
        ``(num_heads, head_size)`` such that
        ``2 * block_size * num_heads * head_size == elems_per_page``.

    Raises:
        ValueError: If the page size does not factor into the target shape.
    """
    denom = 2 * block_size * _SYNTHETIC_NUM_HEADS
    if elems_per_page % denom != 0:
        raise ValueError(
            f"page ({elems_per_page} elems) does not factor into "
            f"(2, block_size={block_size}, num_heads={_SYNTHETIC_NUM_HEADS}, head_size)"
        )
    return _SYNTHETIC_NUM_HEADS, elems_per_page // denom


def _synthetic_attention_page_view(
    base: torch.Tensor,
    num_blocks: int,
    block_size: int,
    page_size_bytes: int,
) -> torch.Tensor:
    """Re-view ``base`` as one opaque logical page per block.

    The result is ``(num_blocks, 2, block_size, 1, head_size)`` in ``base``'s
    dtype, starting at ``base``'s storage offset. Axis 1 holds two equal byte
    slabs of the page, not semantic K and V. A list of these tensors detects
    as ``NL_X_NB_TWO_BS_NH_HS`` (format 2).

    Raises:
        ValueError: If the page does not factor into the synthetic shape.
    """
    elems_per_page = page_size_bytes // base.element_size()
    num_heads, head_size = _synthetic_attention_shape(elems_per_page, block_size)
    flat = base.as_strided(
        (num_blocks, elems_per_page),
        (elems_per_page, 1),
        storage_offset=int(base.storage_offset()),
    )
    return flat.reshape(num_blocks, 2, block_size, num_heads, head_size)


class KVCacheGroupEdit(ABC):
    """One structural edit rule for a KV cache group's registered cache.

    ``matches`` must be side-effect free and decide purely from the vLLM spec
    and the registered cache value; ``apply`` must return a view over the same
    storage (never a copy). ``name`` labels the rule in logs.
    """

    name: str

    @abstractmethod
    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        manager_block_size: int,
    ) -> bool:
        """Return whether this rule applies to the layer's registered cache.

        Args:
            spec: The layer's leaf vLLM KV cache spec.
            kv_cache: The layer's registered cache value -- a tensor, or a
                list/tuple of tensors for split and Mamba layers.
            manager_block_size: Logical block width used by the group's block
                manager. It can differ from the leaf spec's kernel page
                width.
        """

    @abstractmethod
    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        layout_hints: LayoutHints,
        manager_block_size: int,
    ) -> torch.Tensor:
        """Return the edited view for a layer this rule matched.

        Args:
            spec: The layer's leaf vLLM KV cache spec.
            kv_cache: The layer's registered cache value.
            layout_hints: The layout hints from vLLM (see :class:`LayoutHints`).
            manager_block_size: Logical block width used by the group's block
                manager.

        Raises:
            ValueError: If the cache's layout violates the rule's invariants.
        """


class _MambaPageViewEdit(KVCacheGroupEdit):
    """Convert a Mamba page to its equivalent: a sliding-window-style layer.

    A Mamba layer registers ``[conv_state, ssm_state]`` -- two tensors with
    different shapes and dtypes sharing one padded page per block
    (``conv | ssm | pad``) -- which LMCache's attention-shaped transfer path
    cannot represent. Each page is one recurrent state snapshot, equivalent
    for caching purposes to one block of a sliding-window attention layer with
    window == block_size: only the last matched block is ever consumed. The
    edit reinterprets the page buffer as one
    ``[#blocks, 2, block_size, 1, head_size]`` tensor in the conv state's
    dtype, with ``head_size`` derived to fill the page exactly.

    The view's dims are addressing metadata only; the bytes are opaque
    (conv | ssm | pad, not K/V), so content-aware processing does not apply.
    """

    name = "mamba-page-view"

    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _manager_block_size: int,
    ) -> bool:
        # Conv's per-block stride must be the full page. A slice-sized stride
        # passes through; apply re-strides only the packed layout.
        return (
            get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA
            and isinstance(kv_cache, list)
            and bool(kv_cache)
            and kv_cache[0].stride(0) * kv_cache[0].element_size()
            == spec.page_size_bytes
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
        _manager_block_size: int,
    ) -> torch.Tensor:
        # vLLM lays out one padded page per block as (conv | ssm | pad). The
        # conv state's leading dim is the block count and its per-block stride
        # is one full page; it may start at a non-zero storage offset when the
        # pool is shared across layers. Re-striding covers the whole page, ssm
        # and pad included, without including bytes before that offset.
        if not isinstance(kv_cache, list) or not kv_cache:
            raise ValueError(
                f"expected a Mamba [conv_state, ssm_state] tensor list, "
                f"got {type(kv_cache).__name__}"
            )
        conv_state = kv_cache[0]
        if conv_state.stride(0) * conv_state.element_size() != spec.page_size_bytes:
            raise ValueError(
                f"Mamba conv state per-block stride "
                f"({conv_state.stride(0) * conv_state.element_size()} bytes) "
                f"does not equal the page size ({spec.page_size_bytes} bytes)"
            )
        return _synthetic_attention_page_view(
            conv_state,
            int(conv_state.shape[0]),
            int(spec.block_size),
            int(spec.page_size_bytes),
        )


class _MambaSlicePlaneEdit(KVCacheGroupEdit):
    """Re-view slice-strided ``[conv_state, ssm_state]`` as uint8 planes.

    Some hybrid backends (e.g., vLLM-Ascend <= v0.29) register conv and SSM
    as separate contiguous slices carved from a shared pool. Each slice's
    per-block stride is its own byte size, not the padded page, so
    :class:`_MambaPageViewEdit` does not apply. The transfer path expects
    one uint8 plane per slice, ``[num_blocks, block_size, 1, width]`` with
    ``width = nbytes_per_block // block_size``. ``matches`` and ``apply``
    require each state to be a dense contiguous slice with the same
    ``num_blocks`` and require ``spec.block_size`` (scheduler block size) to
    divide that per-block byte count; otherwise ``width`` is not integral and
    the edit does not match (``apply`` raises if forced).
    """

    name = "mamba-slice-planes"

    @staticmethod
    def _dense_block_bytes(state: torch.Tensor) -> int | None:
        """Dense per-block byte size, or ``None`` if the slice is not one."""
        num_blocks = int(state.shape[0])
        if num_blocks <= 0 or not state.is_contiguous():
            return None
        return int(state.numel()) * int(state.element_size()) // num_blocks

    @staticmethod
    def _uint8_plane(state: torch.Tensor, block_size: int) -> torch.Tensor:
        """Re-view one state slice as one uint8 plane."""
        nbytes = _MambaSlicePlaneEdit._dense_block_bytes(state)
        if nbytes is None:
            raise ValueError(
                "Mamba state must be a non-empty contiguous slice, "
                f"got shape {tuple(state.shape)} stride {tuple(state.stride())}"
            )
        if nbytes % block_size != 0:
            raise ValueError(
                f"scheduler block size {block_size} does not divide the "
                f"{nbytes}-byte state block"
            )
        width = nbytes // block_size
        num_blocks = int(state.shape[0])
        raw = state.view(torch.uint8)
        base = int(raw.storage_offset())
        return raw.as_strided(
            (num_blocks, block_size, 1, width),
            (nbytes, width, width, 1),
            storage_offset=base,
        )

    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _manager_block_size: int,
    ) -> bool:
        if (
            get_kv_cache_spec_kind(spec) != KVCacheSpecKind.MAMBA
            or not isinstance(kv_cache, list)
            or not kv_cache
            or not all(isinstance(state, torch.Tensor) for state in kv_cache)
        ):
            return False
        block_size = int(spec.block_size)
        if block_size <= 0:
            return False
        block_counts: set[int] = set()
        for state in kv_cache:
            nbytes = self._dense_block_bytes(state)
            if nbytes is None or nbytes % block_size != 0:
                return False
            block_counts.add(int(state.shape[0]))
        if len(block_counts) != 1:
            return False
        stride_bytes = kv_cache[0].stride(0) * kv_cache[0].element_size()
        return stride_bytes != spec.page_size_bytes

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
        _manager_block_size: int,
    ) -> list[torch.Tensor]:
        if not isinstance(kv_cache, list) or not kv_cache:
            raise ValueError(
                f"expected a Mamba [conv_state, ssm_state] tensor list, "
                f"got {type(kv_cache).__name__}"
            )
        num_blocks = {int(state.shape[0]) for state in kv_cache}
        if len(num_blocks) != 1:
            raise ValueError(
                f"Mamba states cover different block counts {sorted(num_blocks)}"
            )
        block_size = int(spec.block_size)
        return [self._uint8_plane(state, block_size) for state in kv_cache]


class _SubpagedAttentionViewEdit(KVCacheGroupEdit):
    """Re-view a kernel-paged attention tensor as logical-block pages.

    For a Mamba-hybrid model vLLM inflates the attention block size to align
    with the Mamba page (``vllm/platforms/interface.py:_align_hybrid_block_size``;
    e.g. 544 for Qwen3.5-0.8B), and that size is used for all prefix-caching
    logic at the scheduler. But the backend can only page the physical tensor
    at a *kernel* block size it actually supports: ``select_common_block_size``
    (``vllm/v1/worker/utils.py``) returns the largest advertised size dividing
    the logical one, so a backend advertising fixed sizes (FlashInfer's
    ``[16, 32, 64]``, ROCm AITER FA's ``[16, 32]``) re-pages the tensor as

        ``[#blocks, 2, 32, #heads, head_size]``

    which makes LMCache detect block size 32, mistake the group for a
    DeepSeek-compression layer (``block size < scheduler block size``), and
    corrupt the store/retrieve. Fix: re-view the tensor at the scheduler
    block size (one logical block = its 17 contiguous kernel pages),

        ``[#blocks / 17, 2, 544, 1, head_size']``

    Backends advertising ``MultipleOf(16)`` (FlashAttention) page at the
    logical size directly and never need the edit.

    Registered non-MLA attention KV layouts (the block dim is index 2 in both,
    so only the rank tells them apart):

        rank 5 (<= 0.25.x): ``(num_blocks, 2, block_size, num_heads, head_size)``
        rank 4 (>= 0.26.0): ``(num_blocks, num_heads, block_size, 2 * head_size)``

    Both stay reachable: ``hpc_attn`` still registers rank 5 in 0.26.0.

    Cost: before this fix ``kv_caches[:, 0]`` is just the K tensor; after, it
    interleaves K and V at kernel-page granularity. The bytes round-trip
    correctly (store and retrieve share the mapping), but the dims are no
    longer semantic, so content-aware processing does not apply.
    """

    name = "subpaged-attention-view"

    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _manager_block_size: int,
    ) -> bool:
        return (
            # Standard-paged attention only; MLA layouts and declared slot
            # compression (DeepSeek) belong to other transfer paths.
            get_kv_cache_spec_kind(spec) in _SUBPAGEABLE_ATTENTION_KINDS
            and not _declares_slot_compression(spec)
            # A rank-5 or rank-4 layout whose block dim disagrees with the
            # scheduler block-id unit, i.e. the backend re-paged the tensor at
            # its kernel block size.
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim in (5, 4)
            and kv_cache.shape[2] != spec.block_size
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
        _manager_block_size: int,
    ) -> torch.Tensor:
        """Re-view ``kv_cache`` at logical-block granularity.

        The tensor is kernel-paged, either as ``(num_kernel_pages, 2,
        kernel_block_size, num_kv_heads, head_size)`` (vLLM <= 0.25.x) or as
        ``(num_kernel_pages, num_kv_heads, kernel_block_size, 2 * head_size)``
        (vLLM >= 0.26.0); the result is ``(num_logical_blocks, 2,
        spec.block_size, num_heads, head_size)`` over the same storage.

        A rank-4 tensor is re-viewed in memory order, since vLLM registers it
        as a permute view of the backend's physical layout. A rank-5 tensor
        must already be contiguous in logical order.

        Raises:
            ValueError: If the layout is not a recognized kernel-paged shape,
                the sizes do not divide evenly, the pages are not contiguous
                (in memory order for rank 4, logical order for rank 5), or the
                kernel pages of one logical block do not tile its page bytes
                exactly (which would indicate an undeclared packed layout that
                must not be edited).
        """
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim not in (5, 4):
            got = (
                tuple(kv_cache.shape)
                if isinstance(kv_cache, torch.Tensor)
                else type(kv_cache).__name__
            )
            raise ValueError(
                f"expected a (num_blocks, 2, block_size, num_heads, head_size) "
                f"or (num_blocks, num_heads, block_size, 2 * head_size) "
                f"attention KV tensor, got {got}"
            )
        if kv_cache.ndim == 5 and kv_cache.shape[1] != 2:
            raise ValueError(
                f"expected a (num_blocks, 2, block_size, num_heads, head_size) "
                f"attention KV tensor, got {tuple(kv_cache.shape)}"
            )
        logical_block_size = spec.block_size
        kernel_block_size = kv_cache.shape[2]
        if logical_block_size % kernel_block_size != 0:
            raise ValueError(
                f"logical block size {logical_block_size} is not a multiple of "
                f"kernel block size {kernel_block_size}"
            )
        ratio = logical_block_size // kernel_block_size

        num_kernel_pages = kv_cache.shape[0]
        if num_kernel_pages % ratio != 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} is not a multiple of "
                f"the logical/kernel block ratio {ratio}"
            )
        kernel_page_bytes = kv_cache.shape[1:].numel() * kv_cache.element_size()
        if kernel_page_bytes * ratio != spec.page_size_bytes:
            raise ValueError(
                f"{ratio} kernel pages ({kernel_page_bytes * ratio} bytes) do "
                f"not tile the logical page ({spec.page_size_bytes} bytes)"
            )
        if kv_cache.ndim == 4:
            # vLLM registers the rank-4 tensor as a permute view of the
            # backend's physical layout (get_kv_cache_stride_order), so it need
            # not be contiguous. Pages only tile by byte range in memory order:
            # re-view dims outermost-first by stride (a no-op if contiguous).
            ordered = attempt_permute_to_contiguous_view(kv_cache)
            if not isinstance(ordered, torch.Tensor) or not ordered.is_contiguous():
                raise ValueError(
                    "kernel-paged attention KV tensor must be contiguous in "
                    "memory order to re-view as logical pages (shape "
                    f"{tuple(kv_cache.shape)}, strides {tuple(kv_cache.stride())})"
                )
            if ordered.shape[0] != num_kernel_pages:
                raise ValueError(
                    f"expected a num-blocks-first KV cache layout; outermost "
                    f"memory dim is {ordered.shape[0]}, not the kernel page "
                    f"count {num_kernel_pages}"
                )
        else:
            # Rank 5 deliberately keeps the stricter logical-order guard.
            # Extending the rank-4 memory-order normalization to rank 5 would
            # let a permuted tensor (vLLM <= 0.25.x under HND) re-view
            # successfully and then be misread downstream: the edited view is
            # NHD-shaped, so the HND detector branch reads its block size from
            # the synthetic ``num_heads`` axis and resolves 1. Failing loudly
            # here is strictly better than that silent corruption.
            ordered = kv_cache
            if not ordered.is_contiguous():
                raise ValueError(
                    "kernel-paged attention KV tensor must be contiguous to "
                    "re-view as logical pages"
                )

        return _synthetic_attention_page_view(
            ordered,
            num_kernel_pages // ratio,
            logical_block_size,
            int(spec.page_size_bytes),
        )


######################
# vLLM 0.26.0 or later
######################


class _MambaUnifiedViewEdit(KVCacheGroupEdit):
    """Re-view mamba's unified state as a per-token paged tensor.

    For vLLM >= 0.26.0, where the unified KV cache layout is implemented.
    """

    name = "mamba-unified-view"

    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _manager_block_size: int,
    ) -> bool:
        """
        Matches the mamba kv cache with unified layout.
        """
        return (
            get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 4
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        layout_hints: LayoutHints,
        _manager_block_size: int,
    ) -> torch.Tensor:
        """Re-view the per-block state row as block_size tokens.

        Input: [num_blocks, 1, 1, row] with strides (S, row, row, 1),
        where the row is this layer's state and S >= row (the page
        padding, and any sibling layers on a shared pool, live between
        row and S).

        Output for NHD / BLNHC: [num_blocks, block_size, 1, head_size] with
        strides (S, head_size, head_size, 1). HND / BLHNC swaps dims 1 and 2:
        [num_blocks, 1, block_size, head_size] with strides
        (S, block_size * head_size, head_size, 1). Only stride(0) and
        spec.page_size_bytes are read, so a blocks-first layout (layer dim
        inside the block) views exactly like its layers-first twin.

        head_size = ceil(row / block_size), rounded up to the kernels'
        vector alignment, and block_size * head_size may exceed the row
        by at most this layer's own page padding
        (spec.page_size_bytes), never reaching sibling bytes.
        """
        assert isinstance(kv_cache, torch.Tensor), (
            "single-layer KV cache must be a torch.Tensor"
        )
        kv_layout = layout_hints.get("kv_layout", "none")
        if kv_layout not in KV_LAYOUT_NAMES:
            raise ValueError(
                f"Unsupported kv_layout: {kv_layout}. "
                f"Supported: {', '.join(KV_LAYOUT_NAMES)}."
            )
        num_blocks = kv_cache.shape[0]
        row = kv_cache[0].numel()
        block_size = spec.block_size
        elem = kv_cache.element_size()
        block_step = kv_cache.stride(0)
        base = -(-row // block_size)
        # The transfer kernels vectorize by token width, so round it up to
        # the widest alignment that divides the block step. The spill must
        # stay inside this layer's own padded page -- on a shared pool
        # stride(0) spans sibling layers, so it is not the bound.
        page_bytes = spec.page_size_bytes
        head_size = 0
        for align_bytes in (16, 8, 4, 2):
            if align_bytes % elem != 0 or (block_step * elem) % align_bytes != 0:
                continue
            step = align_bytes // elem
            candidate = -(-base // step) * step
            if candidate * block_size * elem <= page_bytes:
                head_size = candidate
                break
        if head_size == 0:
            raise ValueError(
                f"cannot tile a {row}-element state row into {block_size} "
                f"aligned tokens within the {page_bytes}-byte page"
            )
        # BLNHC is blocks-first NHD (tokens before heads), BLHNC blocks-first HND.
        if kv_layout in ("NHD", "BLNHC"):
            inner = (block_size, 1, head_size)
            inner_strides = (head_size, head_size, 1)
        else:
            inner = (1, block_size, head_size)
            inner_strides = (block_size * head_size, head_size, 1)
        return kv_cache.as_strided(
            (num_blocks, *inner), (kv_cache.stride(0), *inner_strides)
        )


class _SubpagedMLAAttentionViewEdit(KVCacheGroupEdit):
    """Re-view a kernel-paged MLA cache at manager-logical granularity.

    For vLLM 0.26 or later. Covers the rank-3 ``[NB, states, C]`` cache
    (Kimi K3: ``[N * 12, 64, 576]`` -> ``[N, 768, 576]``), the unified
    rank-4 cache with one head slot in either order (GLM-5.3-Flash:
    sparse MLA at 64 rows and the kpool indexer with
    ``tokens_per_state=4`` at 32 rows under a 1152-token block), and
    pure MLA models whose scheduler keeps logical block IDs at a width
    wider than the kernel page (GLM-5.3 NVFP4 DSA: 256-token logical
    blocks over 64-slot ``FLASHMLA_SPARSE`` kernel pages). The target
    states per stored page are derived from the *group manager's* logical
    block, divided by the leaf spec's declared ``tokens_per_state``:
    declared slot compression is preserved, and a leaf spec whose
    geometry already matches its kernel pages passes through untouched.

    The kernel states axis is identified by matching bytes per stored
    state against the leaf spec instead of trusting the layout hint, so a
    stale hint cannot misplace the singleton head axis of a rank-4 cache.
    """

    name = "subpaged-mla-attention-view"

    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        manager_block_size: int,
    ) -> bool:
        target_states = manager_block_size // _tokens_per_state(spec)
        return (
            get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MLA_ATTENTION
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim in (3, 4)
            and (
                (kv_cache.ndim == 3 and kv_cache.shape[1] != target_states)
                or (
                    kv_cache.ndim == 4
                    and target_states not in (kv_cache.shape[1], kv_cache.shape[2])
                )
            )
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
        manager_block_size: int,
    ) -> torch.Tensor:
        """Re-view ``kv_cache`` at manager-logical-block granularity.

        The tensor is kernel-paged with ``kernel_states`` states per page;
        the result replaces that dim with the manager-scaled target and
        divides the page count accordingly, over the same storage.

        Raises:
            ValueError: If the sizes do not divide evenly, the states axis
                cannot be identified uniquely from the leaf spec's bytes
                per stored state, or the tensor is not contiguous.
        """
        assert isinstance(kv_cache, torch.Tensor)
        tokens_per_state = _tokens_per_state(spec)
        if manager_block_size % tokens_per_state != 0:
            raise ValueError(
                f"logical block size {manager_block_size} is not divisible "
                f"by the declared tokens-per-state {tokens_per_state}"
            )
        target_states = manager_block_size // tokens_per_state

        if kv_cache.ndim == 3:
            candidate_axes = (1,)
        elif kv_cache.ndim == 4:
            candidate_axes = (1, 2)
        else:
            raise ValueError(
                f"expected a rank-3 or rank-4 MLA cache, got rank {kv_cache.ndim}"
            )

        kernel_page_bytes = kv_cache.shape[1:].numel() * kv_cache.element_size()
        # bytes-per-state identity on the leaf spec: a kernel page holds
        # shape[axis] states at kernel_page_bytes / shape[axis] bytes each,
        # and the leaf page covers spec.block_size // tokens_per_state
        # states at page_size_bytes / (same) bytes each. Matching the two
        # locates the states axis for rank-4 singleton-head layouts where
        # the kv_layout hint may lie, and fails closed on packed layouts
        # the leaf spec does not describe.
        leaf_states = spec.block_size // tokens_per_state
        matching_axes = [
            axis
            for axis in candidate_axes
            if kernel_page_bytes * leaf_states
            == spec.page_size_bytes * kv_cache.shape[axis]
        ]
        if len(matching_axes) != 1:
            candidate_sizes = {axis: kv_cache.shape[axis] for axis in candidate_axes}
            raise ValueError(
                "could not identify exactly one MLA block axis: kernel page "
                f"has {kernel_page_bytes} bytes, leaf spec has "
                f"{spec.page_size_bytes} bytes / {leaf_states} states, "
                f"candidate sizes are {candidate_sizes}; the kernel page "
                "and the leaf spec declare different bytes per token, or "
                "the candidate axes are ambiguous"
            )
        block_axis = matching_axes[0]
        kernel_states = kv_cache.shape[block_axis]
        if target_states % kernel_states != 0:
            raise ValueError(
                f"logical states {target_states} is not a multiple of "
                f"kernel states {kernel_states}"
            )
        ratio = target_states // kernel_states

        num_kernel_pages = kv_cache.shape[0]
        if num_kernel_pages % ratio != 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} is not a multiple of "
                f"the logical/kernel page ratio {ratio}"
            )
        if not kv_cache.is_contiguous():
            raise ValueError(
                "kernel-paged MLA cache must be contiguous to re-view as logical pages"
            )

        shape = list(kv_cache.shape)
        shape[0] = num_kernel_pages // ratio
        shape[block_axis] = target_states
        return kv_cache.view(shape)


def _attention_kv_plane_pair(
    kv_cache: RegisteredKVCache,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return split K/V planes when ``kv_cache`` is a two-tensor sequence."""
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 2:
        k_plane, v_plane = kv_cache[0], kv_cache[1]
        if isinstance(k_plane, torch.Tensor) and isinstance(v_plane, torch.Tensor):
            return k_plane, v_plane
    return None


def _unpadded_attention_page_bytes(spec: KVCacheSpec) -> int:
    """Bytes of one logical K+V page stored in the attention tensors.

    ``page_size_bytes`` may be larger: vLLM-Ascend hybrid models set
    ``page_size_padded`` to the Mamba page, which adds conv state the
    contiguous K/V views do not contain. Only the contiguous-plane rule uses
    this. The interleaved rule tiles the padded ``page_size_bytes`` span.
    """
    real_page_size_bytes = getattr(spec, "real_page_size_bytes", None)
    if isinstance(real_page_size_bytes, int):
        return real_page_size_bytes
    return int(spec.page_size_bytes)


def _subpaged_plane_pair_ratio(
    spec: KVCacheSpec,
    k_plane: torch.Tensor,
    v_plane: torch.Tensor,
) -> int | None:
    """Return the logical/kernel block ratio for a sub-paged rank-4 pair.

    ``None`` means the pair is not a standard-paged attention cache whose
    kernel block evenly divides the scheduler block, or K/V differ in dtype
    or device.
    """
    if get_kv_cache_spec_kind(
        spec
    ) not in _SUBPAGEABLE_ATTENTION_KINDS or _declares_slot_compression(spec):
        return None
    if k_plane.ndim != 4 or v_plane.ndim != 4 or k_plane.shape != v_plane.shape:
        return None
    if k_plane.dtype != v_plane.dtype or k_plane.device != v_plane.device:
        return None
    kernel_block_size = int(k_plane.shape[1])
    logical_block_size = int(spec.block_size)
    if (
        kernel_block_size <= 0
        or kernel_block_size >= logical_block_size
        or logical_block_size % kernel_block_size != 0
    ):
        return None
    ratio = logical_block_size // kernel_block_size
    if int(k_plane.shape[0]) % ratio != 0:
        return None
    return ratio


def _dense_trailing_nhd_row(plane: torch.Tensor) -> bool:
    """Whether ``plane`` has a dense NHD row on axes 1..3.

    Dim 0 may be gapped (interleaved K/V kernel pages); only the trailing
    ``(kernel_block, num_heads, head_size)`` slice must be dense.
    """
    if plane.ndim != 4:
        return False
    num_heads, head_size = int(plane.shape[2]), int(plane.shape[3])
    return (
        int(plane.stride(3)) == 1
        and int(plane.stride(2)) == head_size
        and int(plane.stride(1)) == num_heads * head_size
    )


def _is_interleaved_kernel_kv_pair(
    k_plane: torch.Tensor,
    v_plane: torch.Tensor,
    *,
    ratio: int,
    page_size_bytes: int,
) -> bool:
    """Return whether K/V are v0.30 interleaved kernel pages in one pool."""
    if not (_dense_trailing_nhd_row(k_plane) and _dense_trailing_nhd_row(v_plane)):
        return False
    row_elems = int(k_plane.shape[1:].numel())
    row_bytes = row_elems * int(k_plane.element_size())
    kernel_stride_bytes = int(k_plane.stride(0)) * int(k_plane.element_size())
    if kernel_stride_bytes != 2 * row_bytes or int(v_plane.stride(0)) != int(
        k_plane.stride(0)
    ):
        return False
    if k_plane.untyped_storage().data_ptr() != v_plane.untyped_storage().data_ptr():
        return False
    if v_plane.storage_offset() != k_plane.storage_offset() + row_elems:
        return False
    ptr_gap = v_plane.data_ptr() - k_plane.data_ptr()
    if ptr_gap != row_elems * int(k_plane.element_size()):
        return False
    return kernel_stride_bytes * ratio == page_size_bytes


class _SubpagedAttentionInterleavedPlaneViewEdit(KVCacheGroupEdit):
    """Re-view interleaved K/V kernel pages as one synthetic logical page.

    vLLM-Ascend v0.30 pages hybrid attention as two rank-4 tensors at the
    kernel block size (``select_common_block_size``), while the scheduler
    uses the inflated hybrid block (``_align_hybrid_block_size``). K and V
    share one pool:

        ``[K0][V0][K1][V1]...``

    K's ``stride(0)`` is one K row plus the following V row, and V starts at
    the next row. The K+V pairs of one scheduler block are one contiguous
    byte span, so the edit re-views that span from K in the original dtype:

        ``(num_logical_blocks, 2, scheduler_block_size, 1, head_size)``

    That is the same opaque rank-5 page produced by
    :class:`_SubpagedAttentionViewEdit`, so registration detects
    ``NL_X_NB_TWO_BS_NH_HS`` (format 2). Axis 1 is the first and second half
    of each logical page, not semantic K and V. Separate dense K-only and
    V-only views at the scheduler block size are not possible without a copy.

    The tile check uses ``page_size_bytes``, the full interleaved span.
    """

    name = "subpaged-attention-interleaved-plane-view"

    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _manager_block_size: int,
    ) -> bool:
        planes = _attention_kv_plane_pair(kv_cache)
        if planes is None:
            return False
        k_plane, v_plane = planes
        ratio = _subpaged_plane_pair_ratio(spec, k_plane, v_plane)
        if ratio is None:
            return False
        return _is_interleaved_kernel_kv_pair(
            k_plane,
            v_plane,
            ratio=ratio,
            page_size_bytes=int(spec.page_size_bytes),
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
        _manager_block_size: int,
    ) -> torch.Tensor:
        """Re-view interleaved kernel pages at logical-block granularity.

        Raises:
            ValueError: If ``kv_cache`` is not a two-tensor pair, or the
                interleaved kernel pages do not tile ``page_size_bytes``.
        """
        planes = _attention_kv_plane_pair(kv_cache)
        if planes is None:
            raise ValueError(
                f"expected a (K, V) attention pair, got {type(kv_cache).__name__}"
            )
        k_plane, _v_plane = planes
        logical_block_size = int(spec.block_size)
        kernel_block_size = int(k_plane.shape[1])
        if kernel_block_size <= 0 or logical_block_size % kernel_block_size != 0:
            raise ValueError(
                f"logical block size {logical_block_size} is not a multiple of "
                f"kernel block size {kernel_block_size}"
            )
        ratio = logical_block_size // kernel_block_size
        page_size_bytes = int(spec.page_size_bytes)
        kernel_stride_bytes = int(k_plane.stride(0)) * int(k_plane.element_size())
        logical_block_stride_bytes = ratio * kernel_stride_bytes
        if logical_block_stride_bytes != page_size_bytes:
            raise ValueError(
                f"{ratio} interleaved kernel pages ({logical_block_stride_bytes} "
                f"bytes) do not tile the logical page ({page_size_bytes} bytes)"
            )
        num_kernel_pages = int(k_plane.shape[0])
        if num_kernel_pages % ratio != 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} is not a multiple of "
                f"the logical/kernel block ratio {ratio}"
            )
        return _synthetic_attention_page_view(
            k_plane,
            num_kernel_pages // ratio,
            logical_block_size,
            page_size_bytes,
        )


class _SubpagedAttentionContiguousPlaneViewEdit(KVCacheGroupEdit):
    """Re-view two contiguous kernel-paged planes at the scheduler block.

    vLLM-Ascend <= v0.29 uses the same rank-4 shape as v0.30, but K is every
    kernel page packed together and V starts after the entire K half (or in
    a distinct allocation). Each plane is dense, so it can be re-viewed in
    place without copying:

        ``(num_kernel_pages / ratio, scheduler_block_size, num_kv_heads, head_size)``

    The pair stays two tensors and detects as ``NL_X_TWO_X_NB_BS_NH_HS``
    (format 16). Collapsing the two allocations into one byte plane would
    copy the cache.

    The tile check uses ``real_page_size_bytes``. Hybrid
    ``page_size_bytes`` includes Mamba conv padding that is not in these
    views.
    """

    name = "subpaged-attention-contiguous-plane-view"

    def matches(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _manager_block_size: int,
    ) -> bool:
        planes = _attention_kv_plane_pair(kv_cache)
        if planes is None:
            return False
        k_plane, v_plane = planes
        ratio = _subpaged_plane_pair_ratio(spec, k_plane, v_plane)
        if ratio is None:
            return False
        if not (k_plane.is_contiguous() and v_plane.is_contiguous()):
            return False
        row_bytes = int(k_plane.shape[1:].numel()) * int(k_plane.element_size())
        return 2 * row_bytes * ratio == _unpadded_attention_page_bytes(spec)

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
        _manager_block_size: int,
    ) -> list[torch.Tensor]:
        """Return K and V re-viewed at the scheduler block size.

        Raises:
            ValueError: If ``kv_cache`` is not a two-tensor pair, or the
                kernel pages of one logical block do not tile
                ``real_page_size_bytes``.
        """
        planes = _attention_kv_plane_pair(kv_cache)
        if planes is None:
            raise ValueError(
                f"expected a (K, V) attention pair, got {type(kv_cache).__name__}"
            )
        k_plane, v_plane = planes
        logical_block_size = int(spec.block_size)
        kernel_block_size = int(k_plane.shape[1])
        if kernel_block_size <= 0 or logical_block_size % kernel_block_size != 0:
            raise ValueError(
                f"logical block size {logical_block_size} is not a multiple of "
                f"kernel block size {kernel_block_size}"
            )
        ratio = logical_block_size // kernel_block_size
        row_bytes = int(k_plane.shape[1:].numel()) * int(k_plane.element_size())
        page_size_bytes = _unpadded_attention_page_bytes(spec)
        if 2 * row_bytes * ratio != page_size_bytes:
            raise ValueError(
                f"{ratio} kernel pages of K and V ({2 * row_bytes * ratio} "
                f"bytes) do not tile the logical page ({page_size_bytes} bytes)"
            )
        num_kernel_pages = int(k_plane.shape[0])
        if num_kernel_pages % ratio != 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} is not a multiple of "
                f"the logical/kernel block ratio {ratio}"
            )
        logical_shape = (
            num_kernel_pages // ratio,
            logical_block_size,
            int(k_plane.shape[2]),
            int(k_plane.shape[3]),
        )
        return [k_plane.view(logical_shape), v_plane.view(logical_shape)]


# Rule registry, in match priority order.
_EDITS: tuple[KVCacheGroupEdit, ...] = (
    _MambaUnifiedViewEdit(),
    _MambaPageViewEdit(),
    _MambaSlicePlaneEdit(),
    _SubpagedMLAAttentionViewEdit(),
    _SubpagedAttentionViewEdit(),
    _SubpagedAttentionInterleavedPlaneViewEdit(),
    _SubpagedAttentionContiguousPlaneViewEdit(),
)


def apply_kv_cache_group_edits(
    kv_cache_config: KVCacheConfig | None,
    kv_caches: Mapping[str, RegisteredKVCache],
    layout_hints: LayoutHints,
) -> dict[str, RegisteredKVCache]:
    """Apply all KV cache group metadata edits for LMCache registration.

    Each layer is checked against the ``_EDITS`` rules (first match wins) and
    re-viewed by the matching rule; layers matching no rule pass through
    unchanged. A ``None`` config is returned as-is (as a dict) because no
    per-group specs are available to match safely.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` (read for per-group specs).
        kv_caches: Registered tensors keyed by layer name. Mamba entries are
            ``[conv_state, ssm_state]`` lists; others are single tensors.
        layout_hints: layout hints from vLLM (see :class:`LayoutHints`).

    Returns:
        A new ``dict`` with edited layers re-viewed, others untouched.

    Raises:
        ValueError: If the groups fail :func:`validate_kv_cache_groups`, or a
            matched layer's cache layout violates its rule's invariants (see
            each rule's ``apply``).
    """
    # Backstop for connectors initialized without a kv_cache_config.
    validate_kv_cache_groups(kv_cache_config)
    if kv_cache_config is None:
        return dict(kv_caches)

    if "subpaged_dsa_opaque" in layout_hints:
        raise ValueError(
            "subpaged_dsa_opaque is an internal LMCache hint and must not be "
            "provided by the serving engine"
        )

    edited = dict(kv_caches)
    counts: Counter[str] = Counter()
    opaque_dsa_layers: set[str] = set()
    all_dsa_layers: set[str] = set()
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        group_spec = group.kv_cache_spec
        per_layer_specs = getattr(group_spec, "kv_cache_specs", None)
        manager_block_size = getattr(group_spec, "block_size", None)
        for name in group.layer_names:
            spec = per_layer_specs[name] if per_layer_specs else group_spec
            block_size = manager_block_size
            if block_size is None:
                block_size = spec.block_size
            original_cache = kv_caches[name]
            applied_edit: KVCacheGroupEdit | None = None
            for edit in _EDITS:
                if edit.matches(spec, original_cache, block_size):
                    edited[name] = edit.apply(
                        spec, original_cache, layout_hints, block_size
                    )
                    applied_edit = edit
                    counts[edit.name] += 1
                    break

            edited_cache = edited[name]
            if (
                get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MLA_ATTENTION
                and isinstance(edited_cache, torch.Tensor)
                and edited_cache.dtype == torch.uint8
                and edited_cache.ndim in (3, 4)
                and int(edited_cache.shape[-1]) == 132
            ):
                # A DSA values/scales plane (128 fp8 values plus one scale
                # row). Native logical pages and opaque subpaged views look
                # alike, so track every occurrence.
                all_dsa_layers.add(name)
            if (
                applied_edit is not None
                and applied_edit.name == "subpaged-mla-attention-view"
                and _tokens_per_state(spec) == 1
                and isinstance(original_cache, torch.Tensor)
                and original_cache.dtype == torch.uint8
                and original_cache.ndim in (3, 4)
                and int(original_cache.shape[-1]) == 132
            ):
                # Kernel-paged DSA planes joined into manager-logical pages
                # are NOT native logical DSA pages: k kernel scale blocks
                # are not one native logical page. They must transfer as
                # opaque bytes instead of the values/scales format.
                if name not in all_dsa_layers:
                    raise ValueError(
                        f"subpaged DSA layer {name} did not normalize to "
                        "a uint8 [..., 132] plane"
                    )
                opaque_dsa_layers.add(name)
        logger.info(
            "KV cache group edit geometry: group=%d, manager_block_size=%s, "
            "layers=%d, edits=%s",
            group_idx,
            manager_block_size,
            len(group.layer_names),
            dict(counts) if counts else "none",
        )

    if opaque_dsa_layers:
        native_dsa_layers = all_dsa_layers - opaque_dsa_layers
        if native_dsa_layers:
            raise ValueError(
                "cannot mix opaque subpaged DSA and native logical-page DSA "
                f"in one registration; native layers: {sorted(native_dsa_layers)}"
            )
        # Registration-wide hint: the detector must move these planes to the
        # generic opaque MLA byte transfer instead of the native
        # values/scales format.
        layout_hints["subpaged_dsa_opaque"] = True
        logger.info(
            "Marked %d subpaged DSA layers for opaque uint8 transfer",
            len(opaque_dsa_layers),
        )

    logger.info(
        "KV cache group edits applied: %s",
        dict(counts) if counts else "none",
    )
    return edited

# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache, LayoutHints

logger = init_logger(__name__)

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

DEFAULT_LAYER_NAME_PREFIX = "model.layers."

# ------------------------------------------------------------------ #
#  dtype mapping                                                       #
# ------------------------------------------------------------------ #

DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


# The 8-tuple that uniquely identifies a set of kernel-equivalent layers:
# ``(kv_size, num_heads, head_size, block_size, logical_block_size,
# kv_cache_group_id, sliding_window, dtype)``. Two layers share a
# transfer-kernel launch iff they share this identity — see the
# grouping loop in :meth:`KVLayerGroupsManager.__init__` for the
# derivation.
#
# The fifth slot (``logical_block_size``) is the *scheduler-side*
# block size — i.e. the granularity at which the serving engine hands
# out block IDs for this layer. It comes from
# :data:`~lmcache.v1.gpu_connector.utils.LayoutHints.per_layer_logical_block_size`
# when the engine supplies it, otherwise it defaults to the physical
# block size (``bs``). On DeepSeek-V4 with vLLM's *hybrid manager
# active*, several physical-equivalent layer sets receive block IDs at
# different scheduler grids (e.g. dense-MLA layers use a 256-token
# grid while the SWA layers use a 64-token grid). Two layers with the
# same physical layout but different scheduler grids are
# kernel-incompatible — one ``LoadStoreOp`` cannot index both — so the
# logical block size must participate in the identity tuple.
#
# The sixth slot (``kv_cache_group_id``) is the engine's per-layer
# block-ID namespace handle. Two layers may share an LMCache transfer-
# kernel group only if a single ``LoadStoreOp``'s ``block_ids``
# correctly indexes both — which is true iff they share a namespace.
# It comes from
# :data:`~lmcache.v1.gpu_connector.utils.LayoutHints.per_layer_kv_cache_group_id`
# when the engine supplies it, otherwise every layer is treated as
# namespace 0. On DeepSeek-V4 with the hybrid manager active, the
# even-indexed-plus-MTP SWA layers (vLLM gid 1) and the odd-indexed
# SWA layers (vLLM gid 2) have identical ``KVCacheSpec`` field values
# — same ``block_size``, ``sliding_window``, ``head_size``, dtype, and
# physical ``shape[1]`` — but pull block IDs from disjoint pools.
# Without this slot they merge into one LMCache group and a STORE op
# carrying gid 1's block IDs would silently mis-index gid 2's layers.
#
# The seventh slot (``sliding_window``) is the SWA window size in
# tokens, or 0 for full-attention layers. It comes from
# :data:`~lmcache.v1.gpu_connector.utils.LayoutHints.per_layer_sliding_window`.
# When non-zero, the SWA-suffix-only optimization kicks in for that
# group (store/retrieve the last ``ceil(window/logical_bs)`` blocks
# per chunk instead of all). Layers with different windows have
# different per-chunk byte budgets and so cannot share a transfer
# group. When absent, every layer is treated as ``sliding_window = 0``
# and the 8-tuple collapses to the prior 7-tuple identity.
#
# When neither hint set is provided, every layer's logical block size
# defaults to its physical ``bs``, every layer's namespace defaults to
# 0, and every layer's sliding_window defaults to 0, so the 8-tuple
# collapses to the prior 5-tuple grouping (kv_size, nh, hs, bs, dtype).
LayerGroupIdentity = tuple[int, int, int, int, int, int, int, torch.dtype]


@dataclass
class KVLayerGroupInfo:
    """A single transfer-kernel dispatch unit: a set of KV layers that can
    ride one kernel launch with one ``PageBufferShapeDesc``.

    Membership is decided by :class:`KVLayerGroupsManager` according to
    :data:`LayerGroupIdentity`; every layer referenced by
    ``layer_indices`` shares the same
    ``(kv_size, num_heads, head_size, block_size, dtype)`` signature.
    Consumers use ``layer_indices`` to pull the matching device pointers
    out of ``kv_caches`` (via
    :func:`~lmcache.v1.gpu_connector.utils.get_group_data_ptrs`) and
    feed them to the kernel alongside ``shape_desc``.

    ``dtype`` is carried alongside ``shape_desc`` because
    ``PageBufferShapeDesc.element_size`` is a byte width, which cannot
    distinguish dtypes that share a byte count (e.g. bfloat16 and
    float16 are both 2 bytes). Kernel template instantiation keys on the
    torch dtype, not the byte width, so we keep it explicit.

    Treat instances as immutable after construction; callers may hold
    references for the lifetime of the manager.
    """

    layer_indices: list[int]
    """0-based layer indices belonging to this group, in the order the
    kernel should iterate them. Fed to ``get_group_data_ptrs`` to build
    the per-group pointer array."""
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    """Kernel-facing shape descriptor shared by every layer in the group.
    All eight fields (``kv_size, nl, nb, bs, nh, hs, element_size,
    block_stride_elems``) are stamped once at construction. Note that
    ``shape_desc.bs`` carries the **physical** block size (the on-GPU
    tensor's per-block slot count); the *logical* (scheduler) block
    size lives on :attr:`logical_block_size` and is used by chunking
    arithmetic, which runs upstream of the kernel."""
    dtype: torch.dtype
    """Torch dtype of the KV cache tensors for this group. Used for
    kernel template instantiation; see class docstring for why we keep
    this alongside ``shape_desc.element_size``."""
    compress_ratio: int = 1
    """Logical-tokens-per-physical-slot for this group. ``1`` for
    non-compressed groups (one logical token per physical slot);
    greater than ``1`` for compressed groups where each physical slot
    packs ``compress_ratio`` logical tokens (e.g. DeepSeek V4
    compressor / indexer caches). Derived from this group's
    :attr:`logical_block_size` and ``shape_desc.bs`` at
    :class:`KVLayerGroupsManager` construction time:
    ``compress_ratio = logical_block_size // shape_desc.bs``."""
    physical_chunk_size: int = 0
    """Number of *physical* slots in one LMCache chunk for this group
    (= ``lmcache_logical_chunk_size // compress_ratio``). This is what
    the block-level transfer kernel must be told, not the logical
    ``lmcache_logical_chunk_size`` which counts vLLM tokens. ``0``
    means the field has not been populated yet; ``GPUCacheContext``
    fills it in after construction once ``lmcache_logical_chunk_size``
    is known."""
    logical_block_size: int = 0
    """Scheduler-side tokens-per-block for this group, i.e. the
    granularity at which the serving engine hands out block IDs for
    these layers. Equals ``shape_desc.bs`` (physical) when the engine
    does not provide a per-layer hint via
    :data:`~lmcache.v1.gpu_connector.utils.LayoutHints.per_layer_logical_block_size`,
    so existing single-group call sites are unaffected. When the hint
    is present, this field carries the engine's per-layer
    ``KVCacheSpec.block_size`` for the represented layer — which is
    what chunking math (``blocks_per_chunk_g = lmcache_chunk_size //
    logical_block_size``) must use, and what the connector's
    ``LoadStoreOp.block_ids`` are stride-compatible with. ``0``
    indicates the field has not been populated yet (left in for
    backwards compatibility with no-arg constructors); the manager
    always sets a positive value at construction."""
    kv_cache_group_id: int = 0
    """Engine-side block-ID namespace handle for this group. Stamped
    from
    :data:`~lmcache.v1.gpu_connector.utils.LayoutHints.per_layer_kv_cache_group_id`
    at registration. Two LMCache groups with different
    ``kv_cache_group_id`` values reflect layers whose
    ``LoadStoreOp.block_ids`` come from disjoint engine-side
    ``BlockPool``-allocated namespaces — they cannot be merged into one
    transfer-kernel launch even when every other field matches.
    Defaults to 0 when the engine does not provide the hint, in which
    case all layers share namespace 0 and grouping behaves identically
    to the prior 6-tuple identity."""
    sliding_window: int = 0
    """Sliding-window attention window size in tokens, or ``0`` for
    full-attention groups. When non-zero, the SWA-suffix-only
    optimization activates for this group: store/retrieve only the
    last ``ceil(sliding_window / logical_block_size)`` blocks per
    chunk instead of all ``lmcache_chunk_size // logical_block_size``
    blocks. Stamped from
    :data:`~lmcache.v1.gpu_connector.utils.LayoutHints.per_layer_sliding_window`
    at registration. Two groups with different ``sliding_window``
    values cannot share a transfer-kernel launch (their per-chunk
    byte budgets differ); the field is the 7th component of
    :data:`LayerGroupIdentity`."""

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        sd = self.shape_desc
        return (
            f"KVLayerGroupInfo(layers={len(self.layer_indices)}, "
            f"indices={indices_repr}, "
            f"shape_desc=(kv={sd.kv_size}, nl={sd.nl}, nb={sd.nb}, "
            f"bs={sd.bs}, nh={sd.nh}, hs={sd.hs}, "
            f"element_size={sd.element_size}, "
            f"block_stride_elems={sd.block_stride_elems}), "
            f"dtype={self.dtype}, "
            f"compress_ratio={self.compress_ratio}, "
            f"physical_chunk_size={self.physical_chunk_size}, "
            f"logical_bs={self.logical_block_size}, "
            f"kv_cache_group_id={self.kv_cache_group_id}, "
            f"sliding_window={self.sliding_window})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in this group."""
        return len(self.layer_indices)

    @property
    def hidden_dim_size(self) -> int:
        """Hidden dimension size (``num_heads * head_size``)."""
        return self.shape_desc.nh * self.shape_desc.hs


class KVLayerGroupsManager:
    """Partition a model's KV layers into transfer-kernel dispatch units.

    At construction time, every layer in ``kv_caches`` is bucketed by its
    :data:`LayerGroupIdentity` (``(kv_size, num_heads, head_size,
    block_size, logical_block_size, kv_cache_group_id, sliding_window,
    dtype)``). Each bucket becomes one :class:`KVLayerGroupInfo`
    holding the layer indices, a shared :class:`PageBufferShapeDesc`,
    and the group's torch dtype.

    Downstream consumers (``VLLMPagedMemGPUConnectorV3``,
    ``GPUCacheContext``, the multiprocess server) iterate
    ``self.kv_layer_groups`` and issue one transfer-kernel launch per
    group. The manager itself is a pure metadata object — it does not
    own any GPU buffers or perform any transfers.

    Layout parsing is delegated entirely to
    :mod:`lmcache.v1.gpu_connector.utils`; this class only drives the
    grouping and look-up.
    """

    def __init__(
        self,
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_blocks: int,
        layout_hints: "LayoutHints | None" = None,
        lmcache_logical_chunk_size: int = 256,
    ) -> None:
        """Partition layers into groups keyed by
        :data:`LayerGroupIdentity`.

        For each layer ``i`` in ``kv_caches``, read
        ``(kv_size, num_heads, head_size, dtype)`` via the format-aware
        accessors in ``utils.py``. Layers with identical identities are
        bucketed together; each bucket becomes one
        :class:`KVLayerGroupInfo`.

        Groups are emitted in the order of their first-appearing layer,
        so group indices are deterministic across runs.

        Args:
            kv_caches: KV cache structure accepted by
                :func:`normalize_kv_and_discover_format`.
            gpu_kv_format: Format returned by
                :func:`normalize_kv_and_discover_format`.
            num_blocks: Number of paged blocks. Stamped into every
                ``shape_desc.nb``. Each group's ``shape_desc.bs`` is
                discovered per-layer via :func:`get_block_size`, so
                compressed and non-compressed groups can coexist.
            layout_hints: Engine-provided hints. The manager reads:

                * ``inference_engine_logical_block_size`` (single
                  global value, used as the *fallback* per-layer
                  logical block size when the per-layer hint is
                  absent and as the legacy single-source-of-truth
                  for non-hybrid engines).
                * ``per_layer_logical_block_size`` (optional list of
                  length ``num_layers``). When present, each
                  layer's scheduler-side block size is used to key
                  layer grouping (so layers with same physical
                  shape but different scheduler grids end up in
                  separate groups) and to derive that group's
                  ``compress_ratio = logical_bs_g // physical_bs_g``.
                  Required when the engine has KV layer groups with
                  mixed scheduler block sizes (vLLM hybrid manager
                  active on DeepSeek-V4).
                * ``per_layer_kv_cache_group_id`` (optional list of
                  length ``num_layers``). When present, each layer's
                  engine-side block-ID *namespace* handle joins the
                  identity tuple, so layers whose ``KVCacheSpec``
                  field values fully match but pull block IDs from
                  disjoint engine-side namespaces (e.g.
                  DeepSeek-V4's vLLM gids 1 and 2: even+MTP vs odd
                  SWA layers) end up in different LMCache groups.
                * ``per_layer_sliding_window`` (optional list of
                  length ``num_layers``). When present, each layer's
                  SWA window size joins the identity tuple, so
                  layers with different windows end up in different
                  LMCache groups. Non-zero entries activate the
                  SWA-suffix-only optimization for that group.

                ``None`` (or a hints dict without any of the above)
                means every group is treated as non-compressed
                (``compress_ratio == 1``, ``logical_block_size ==
                shape_desc.bs``, ``kv_cache_group_id == 0``,
                ``sliding_window == 0``).
            lmcache_logical_chunk_size: Logical tokens per LMCache chunk
                (one logical token = one inference-engine token).
                Together with ``compress_ratio`` it determines each
                group's ``physical_chunk_size =
                lmcache_logical_chunk_size // compress_ratio``, the
                number of *physical* slots per chunk fed to the
                block-level transfer kernel.
        """
        # Import here to break a circular import via
        # lmcache.v1.gpu_connector.__init__ → metadata → kv_layer_groups.
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_num_layers,
            make_page_buffer_shape_desc,
            resolve_block_stride_and_log_layout,
        )

        # Pull the inference-engine logical block size out of
        # ``layout_hints`` once; ``None`` means no compression info
        # available and every group is treated as non-compressed below.
        # The attribute is finalised after the group-building loop
        # below, where ``None`` is replaced by the first group's
        # physical ``bs`` so the public ``int`` contract holds.
        self.inference_engine_logical_block_size_: "int | None" = (
            layout_hints.get("inference_engine_logical_block_size")
            if layout_hints
            else None
        )
        self.kv_layer_groups: list[KVLayerGroupInfo] = []

        num_layers = get_num_layers(kv_caches, gpu_kv_format)
        if num_layers == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        # Resolve the per-layer logical block size hint, if present.
        # When absent, every layer's logical block size will default to
        # its physical ``bs`` (effectively reproducing the prior
        # 5-tuple grouping behavior on engines that don't supply the
        # hint). When present, the per-layer values participate in
        # :data:`LayerGroupIdentity` so layers with same physical
        # shape but different scheduler grids land in distinct LMCache
        # groups (DeepSeek-V4 hybrid manager active).
        per_layer_logical_bs: "list[int] | None" = None
        per_layer_namespace: "list[int] | None" = None
        per_layer_sliding_window: "list[int] | None" = None
        if layout_hints is not None:
            logical_hint = layout_hints.get("per_layer_logical_block_size")
            if logical_hint is not None:
                if len(logical_hint) != num_layers:
                    raise ValueError(
                        "per_layer_logical_block_size length "
                        f"({len(logical_hint)}) does not match "
                        f"num_layers ({num_layers}); each registered "
                        "layer must have exactly one logical block "
                        "size entry"
                    )
                bad = [i for i, bs in enumerate(logical_hint) if bs <= 0]
                if bad:
                    raise ValueError(
                        "per_layer_logical_block_size has invalid "
                        f"(non-positive) entries at positions {bad[:8]}"
                        + ("..." if len(bad) > 8 else "")
                        + "; every registered layer must be covered by "
                        "exactly one engine-side block size"
                    )
                per_layer_logical_bs = list(logical_hint)

            ns_hint = layout_hints.get("per_layer_kv_cache_group_id")
            if ns_hint is not None:
                if len(ns_hint) != num_layers:
                    raise ValueError(
                        "per_layer_kv_cache_group_id length "
                        f"({len(ns_hint)}) does not match num_layers "
                        f"({num_layers}); each registered layer must "
                        "have exactly one block-ID namespace entry"
                    )
                bad = [i for i, ns in enumerate(ns_hint) if ns < 0]
                if bad:
                    raise ValueError(
                        "per_layer_kv_cache_group_id has invalid "
                        f"(negative) entries at positions {bad[:8]}"
                        + ("..." if len(bad) > 8 else "")
                        + "; namespace IDs must be non-negative"
                    )
                per_layer_namespace = list(ns_hint)

            sw_hint = layout_hints.get("per_layer_sliding_window")
            if sw_hint is not None:
                if len(sw_hint) != num_layers:
                    raise ValueError(
                        "per_layer_sliding_window length "
                        f"({len(sw_hint)}) does not match num_layers "
                        f"({num_layers}); each registered layer must "
                        "have exactly one sliding-window entry"
                    )
                bad = [i for i, sw in enumerate(sw_hint) if sw < 0]
                if bad:
                    raise ValueError(
                        "per_layer_sliding_window has invalid "
                        f"(negative) entries at positions {bad[:8]}"
                        + ("..." if len(bad) > 8 else "")
                        + "; sliding windows must be non-negative "
                        "(0 = full attention)"
                    )
                per_layer_sliding_window = list(sw_hint)

        groups_dict = self._group_layers_by_identity(
            kv_caches,
            gpu_kv_format,
            num_layers,
            per_layer_logical_bs,
            per_layer_namespace,
            per_layer_sliding_window,
        )

        # Emit groups in order of their first-appearing layer so that group
        # indices remain deterministic across runs.
        for group_idx, (
            (
                _,
                _,
                _,
                bs,
                group_logical_bs,
                group_namespace,
                group_sliding_window,
                dt,
            ),
            indices,
        ) in enumerate(sorted(groups_dict.items(), key=lambda kv: kv[1][0])):
            block_stride_elems = resolve_block_stride_and_log_layout(
                kv_caches,
                gpu_kv_format,
                layer_idx=indices[0],
                group_idx=group_idx,
            )
            shape_desc = make_page_buffer_shape_desc(
                kv_caches,
                gpu_kv_format,
                layer_idx=indices[0],
                num_layers_in_group=len(indices),
                num_blocks=num_blocks,
                block_size=bs,
                block_stride_elems=block_stride_elems,
            )

            compress_ratio, physical_chunk_size = self._derive_compression_metadata(
                group_idx=group_idx,
                bs=bs,
                logical_block_size=group_logical_bs,
                ie_logical_block_size=self.inference_engine_logical_block_size_,
                lmcache_logical_chunk_size=lmcache_logical_chunk_size,
            )

            self.kv_layer_groups.append(
                KVLayerGroupInfo(
                    layer_indices=indices,
                    shape_desc=shape_desc,
                    dtype=dt,
                    compress_ratio=compress_ratio,
                    physical_chunk_size=physical_chunk_size,
                    logical_block_size=group_logical_bs,
                    kv_cache_group_id=group_namespace,
                    sliding_window=group_sliding_window,
                )
            )

        self.inference_engine_logical_block_size_ = (
            self.inference_engine_logical_block_size_
            or self.kv_layer_groups[0].shape_desc.bs
        )

        logger.info("KV layer groups: %s", self.kv_layer_groups)

    @staticmethod
    def _derive_compression_metadata(
        group_idx: int,
        bs: int,
        logical_block_size: int,
        ie_logical_block_size: "int | None",
        lmcache_logical_chunk_size: int,
    ) -> tuple[int, int]:
        """Resolve ``(compress_ratio, physical_chunk_size)`` for one group.

        ``compress_ratio`` is per-group:
        ``compress_ratio_g = logical_block_size // bs``. Each layer
        group has its own ``logical_block_size`` (carried by the
        identity tuple); when ``per_layer_logical_block_size`` is not
        provided the manager defaults each group's
        ``logical_block_size`` to its physical ``bs``, so
        ``compress_ratio == 1`` and the per-group formula is identical
        to the prior global formula.

        ``ie_logical_block_size`` is consulted only when its presence
        differs from the per-layer hint's: if the engine provides
        neither the per-layer hint nor a global value (i.e. a
        non-vLLM engine), ``compress_ratio`` falls back to 1 even
        when ``logical_block_size > bs`` — preserving the old
        non-compressed default for tools and tests that don't
        provide hints.

        ``physical_chunk_size`` is then
        ``lmcache_logical_chunk_size // compress_ratio``, the per-chunk
        physical slot count fed to the block-level transfer kernel.

        Args:
            group_idx: Group index (used in error messages and logs).
            bs: This group's physical block size (``shape_desc.bs``).
            logical_block_size: This group's logical (scheduler-side)
                block size.
            ie_logical_block_size: Optional global hint — if both this
                and the per-layer hint are absent, the group falls
                back to ``compress_ratio == 1``.
            lmcache_logical_chunk_size: Logical tokens per LMCache
                chunk (one logical token = one engine token).

        Returns:
            ``(compress_ratio, physical_chunk_size)`` tuple.

        Raises:
            ValueError: If ``logical_block_size`` is not a multiple of
                ``bs``, or if ``lmcache_logical_chunk_size`` is not a
                multiple of the resulting ``compress_ratio``.
        """
        if ie_logical_block_size is None and logical_block_size == bs:
            # Neither the global nor the per-layer hint was supplied
            # for this group's layers (per-layer defaults to physical
            # bs when no hint is provided): treat as non-compressed.
            compress_ratio = 1
        else:
            if logical_block_size % bs != 0:
                raise ValueError(
                    f"group {group_idx}: logical block size "
                    f"{logical_block_size} must be a multiple of "
                    f"physical slot count {bs}"
                )
            compress_ratio = logical_block_size // bs
        if lmcache_logical_chunk_size % compress_ratio != 0:
            raise ValueError(
                f"lmcache_logical_chunk_size {lmcache_logical_chunk_size} "
                f"must be a multiple of compress_ratio {compress_ratio} "
                f"(group {group_idx})"
            )
        physical_chunk_size = lmcache_logical_chunk_size // compress_ratio
        if compress_ratio != 1:
            logger.info(
                "group %d: compressed "
                "(logical_block_size=%d -> physical_bs=%d, "
                "compress_ratio=%d, physical_chunk_size=%d)",
                group_idx,
                logical_block_size,
                bs,
                compress_ratio,
                physical_chunk_size,
            )
        return compress_ratio, physical_chunk_size

    @staticmethod
    def _group_layers_by_identity(
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_layers: int,
        per_layer_logical_bs: "list[int] | None" = None,
        per_layer_namespace: "list[int] | None" = None,
        per_layer_sliding_window: "list[int] | None" = None,
    ) -> dict[LayerGroupIdentity, list[int]]:
        """Partition layer indices by :data:`LayerGroupIdentity`.

        Linear single pass over ``kv_caches``; layers sharing the same
        ``(kv_size, num_heads, head_size, block_size,
        logical_block_size, kv_cache_group_id, sliding_window, dtype)``
        signature land in the same bucket. When ``per_layer_logical_bs``
        is None, each layer's ``logical_block_size`` defaults to its
        physical block size; when ``per_layer_namespace`` is None,
        each layer's namespace defaults to 0; when
        ``per_layer_sliding_window`` is None, every layer is treated
        as ``sliding_window = 0`` (full attention). With all three
        defaults, the 8-tuple is effectively a 5-tuple (preserving
        prior grouping behavior on engines without these hints). The
        returned dict's value lists are later passed by reference into
        :class:`KVLayerGroupInfo` instances, so the dict itself is
        garbage-collected after ``__init__`` returns while the lists
        stay alive on each group.
        """
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_block_size,
            get_dtype,
            get_head_size,
            get_num_heads,
            is_mla,
        )

        mla = is_mla(gpu_kv_format)
        kv_size = 1 if mla else 2
        groups_dict: dict[LayerGroupIdentity, list[int]] = defaultdict(list)
        for idx in range(num_layers):
            nh = 1 if mla else get_num_heads(kv_caches, gpu_kv_format, idx)
            hs = get_head_size(kv_caches, gpu_kv_format, idx)
            dt = get_dtype(kv_caches, gpu_kv_format, idx)
            bs = get_block_size(kv_caches, gpu_kv_format, idx)
            logical_bs = (
                per_layer_logical_bs[idx] if per_layer_logical_bs is not None else bs
            )
            namespace = (
                per_layer_namespace[idx] if per_layer_namespace is not None else 0
            )
            sliding_window = (
                per_layer_sliding_window[idx]
                if per_layer_sliding_window is not None
                else 0
            )
            groups_dict[
                (kv_size, nh, hs, bs, logical_bs, namespace, sliding_window, dt)
            ].append(idx)
        return groups_dict

    @property
    def num_groups(self) -> int:
        """Number of :class:`KVLayerGroupInfo` entries.

        Zero if ``kv_caches`` had no layers at construction time.
        """
        return len(self.kv_layer_groups)

    @property
    def inference_engine_logical_block_size(self):
        """Inference-engine-side logical block size.

        Taken from ``layout_hints`` at construction time, or falls back
        to the first group's physical ``bs`` when no hint is provided
        (non-vLLM engines, or vLLM without mixed-compression KV groups),
        in which case every group is treated as non-compressed.
        """
        return (
            self.inference_engine_logical_block_size_
            or self.kv_layer_groups[0].shape_desc.bs
        )

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Return the :class:`PageBufferShapeDesc` for *group_idx*.

        Equivalent to ``self.kv_layer_groups[group_idx].shape_desc``.

        Args:
            group_idx: 0-based group index.

        Raises:
            IndexError: If *group_idx* is out of range.
        """
        return self.kv_layer_groups[group_idx].shape_desc

    def get_physical_chunk_size(self, group_idx: int) -> int:
        """Return the per-chunk *physical* slot count for *group_idx*.

        Equivalent to
        ``self.kv_layer_groups[group_idx].physical_chunk_size``.
        For non-compressed groups this equals
        ``lmcache_logical_chunk_size``; for compressed groups it equals
        ``lmcache_logical_chunk_size // compress_ratio`` and is what the
        block-level transfer kernel must be told (the logical chunk size
        in *vLLM tokens* is not what the kernel addresses).

        Args:
            group_idx: 0-based group index.

        Raises:
            IndexError: If *group_idx* is out of range.
        """
        return self.kv_layer_groups[group_idx].physical_chunk_size


# ------------------------------------------------------------------ #
#  CLI shape-spec parser                                               #
# ------------------------------------------------------------------ #


def parse_kvcache_shape_spec(
    spec_str: str,
) -> list[KVLayerGroupInfo]:
    """Parse a ``--kvcache-shape-spec`` string into layer groups.

    **Grammar** (EBNF-ish)::

        spec        := group { ";" group }
        group       := "(" shape ")" ":" dtype ":" layer_count
        shape       := kv_size "," NB "," BS "," NH "," HS
        dtype       := "float16" | "float32" | "bfloat16" | "uint8"
        layer_count := positive integer

    **Field semantics** (names aligned with ``GPUKVFormat``; see
    :func:`lmcache.v1.gpu_connector.utils.get_gpu_kv_shape_description`):

    * ``kv_size`` -- leading dim (``2`` for standard K/V, ``1`` for MLA).
    * ``NB`` -- ``num_blocks``: paged-KV block count.
    * ``BS`` -- ``block_size``: tokens per paged-KV block.
    * ``NH`` -- ``num_heads``: attention heads per layer.
    * ``HS`` -- ``head_size``: per-head hidden dim.
    * ``dtype`` -- element dtype (case-insensitive). ``uint8`` is used
      by FP8-quantized layouts.
    * ``layer_count`` -- number of consecutive layers sharing this
      group's geometry. Groups are concatenated in declaration order;
      ``layer_indices`` are assigned sequentially starting from 0.

    When consumed by the ``lmcache bench kvcache`` CLI, ``NB``/``BS``
    from the spec take precedence over ``--num-blocks`` / ``--block-size``
    CLI flags when set to a positive value.

    **Examples**::

        # Single homogeneous group: 32 layers of standard K/V
        (2,1024,16,8,128):float16:32

        # Heterogeneous model: 30 dense layers + 2 MLA-ish layers
        (2,1024,16,8,128):float16:30;(1,1024,16,4,64):bfloat16:2

        # FP8-quantized KV cache
        (2,1024,16,8,128):uint8:32

    See also :func:`format_kvcache_shape_spec` for the inverse -- it
    turns a parsed group list back into a human-readable spec string
    (handy for CLI echo-back / debug logging).

    Returns:
        A list of :class:`KVLayerGroupInfo`, one per group.

    Raises:
        ValueError: Malformed spec, unknown dtype, or a shape with a
            wrong number of dimensions.
    """
    if not spec_str:
        raise ValueError("KV shape specification cannot be empty")

    groups: list[KVLayerGroupInfo] = []
    layer_offset = 0

    for group_spec in spec_str.split(";"):
        group_spec = group_spec.strip()
        if not group_spec:
            continue

        if not (group_spec.startswith("(") and "):" in group_spec):
            raise ValueError("Invalid group spec format: %s" % group_spec)

        shape_end = group_spec.find(")")
        shape_str = group_spec[1:shape_end]

        remaining = group_spec[shape_end + 2 :]  # Skip "):"
        parts = remaining.split(":")
        if len(parts) != 2:
            raise ValueError("Invalid group spec format: %s" % group_spec)

        dtype_str = parts[0].strip()
        layer_count_str = parts[1].strip()

        dtype_key = dtype_str.lower()
        if dtype_key not in DTYPE_MAP:
            raise ValueError(
                "Unrecognized dtype '%s' in group spec: %s. "
                "Supported: %s" % (dtype_str, group_spec, list(DTYPE_MAP.keys()))
            )
        try:
            shape = tuple(int(p.strip()) for p in shape_str.split(","))
            layer_count = int(layer_count_str)
        except ValueError as exc:
            raise ValueError("Invalid number in group spec: %s" % group_spec) from exc
        dtype = DTYPE_MAP[dtype_key]

        if len(shape) != 5:
            raise ValueError(
                "Shape must be a 5-tuple (kv_size,nb,bs,nh,hs): %s" % group_spec
            )
        kv_size, nb, bs, nh, hs = shape
        shape_desc = lmc_ops.PageBufferShapeDesc()
        shape_desc.kv_size = kv_size
        shape_desc.nl = layer_count
        shape_desc.nb = nb
        shape_desc.bs = bs
        shape_desc.nh = nh
        shape_desc.hs = hs
        shape_desc.element_size = dtype.itemsize

        indices = list(range(layer_offset, layer_offset + layer_count))
        groups.append(
            KVLayerGroupInfo(
                layer_indices=indices,
                shape_desc=shape_desc,
                dtype=dtype,
                logical_block_size=bs,
            )
        )
        layer_offset += layer_count

    if not groups:
        raise ValueError("No valid layer groups found in spec")

    return groups


def format_kvcache_shape_spec(groups: list[KVLayerGroupInfo]) -> str:
    """Format layer groups back into a ``--kvcache-shape-spec`` string.

    This is the inverse of :func:`parse_kvcache_shape_spec`; the
    result is round-trip safe (i.e. ``parse(format(x)) == x`` for any
    ``x`` that ``parse`` would produce).

    The returned string is also human-readable and is used by the
    ``lmcache bench kvcache`` CLI to echo the resolved KV cache
    geometry at startup, so operators can verify that their spec was
    interpreted as intended.

    Example::

        >>> groups = parse_kvcache_shape_spec(
        ...     "(2,1024,16,8,128):float16:30;"
        ...     "(1,1024,16,4,64):bfloat16:2"
        ... )
        >>> format_kvcache_shape_spec(groups)
        '(2,1024,16,8,128):float16:30;(1,1024,16,4,64):bfloat16:2'

    Args:
        groups: Layer groups as returned by
            :func:`parse_kvcache_shape_spec`.

    Raises:
        ValueError: If *groups* is empty or contains an unsupported
            dtype (one that is not present in :data:`DTYPE_MAP`).
    """
    if not groups:
        raise ValueError("Cannot format an empty layer group list")

    # Invert DTYPE_MAP once: torch.dtype -> canonical string name.
    dtype_names = {v: k for k, v in DTYPE_MAP.items()}

    parts: list[str] = []
    for g in groups:
        sd = g.shape_desc
        try:
            dtype_str = dtype_names[g.dtype]
        except KeyError as exc:
            raise ValueError("dtype %s is not present in DTYPE_MAP" % g.dtype) from exc
        parts.append(
            "(%d,%d,%d,%d,%d):%s:%d"
            % (sd.kv_size, sd.nb, sd.bs, sd.nh, sd.hs, dtype_str, sd.nl)
        )
    return ";".join(parts)

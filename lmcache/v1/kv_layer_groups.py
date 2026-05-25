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


# 8-tuple identity for kernel-equivalent layers. Two layers share a
# transfer-kernel launch iff their identities match. The first five fields
# are the legacy kernel-shape signature; the other three came from the
# DeepSeek-V4 hybrid-KV-cache work and collapse to defaults when the engine
# supplies no hints, recovering the legacy 5-tuple behavior:
#
# - ``logical_block_size`` (default = physical ``bs``): scheduler-side
#   tokens per block. ``LoadStoreOp.block_ids`` is keyed at this stride.
# - ``kv_cache_group_id`` (default 0): engine-side block-ID namespace.
#   Disjoint namespaces cannot share a launch (V4 vLLM gids 1 vs 2).
# - ``sliding_window`` (default 0): SWA window in tokens; non-zero
#   activates SWA-suffix-only stores/retrieves.
LayerGroupIdentity = tuple[int, int, int, int, int, int, int, torch.dtype]


@dataclass
class KVLayerGroupInfo:
    """One transfer-kernel dispatch unit: a set of KV layers that share a
    :data:`LayerGroupIdentity` and ride one kernel launch with one
    ``PageBufferShapeDesc``. Treat as immutable after construction.

    ``dtype`` is carried separately from ``shape_desc.element_size``
    because the latter is byte width — bfloat16 and float16 both report 2
    — and kernel template instantiation keys on the torch dtype.
    """

    layer_indices: list[int]
    """0-based layer indices in this group, in kernel-iteration order."""
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    """Kernel-facing shape descriptor. ``shape_desc.bs`` is the physical
    block size; the scheduler-side block size lives on
    :attr:`logical_block_size` and the two coincide when ``compress_ratio == 1``."""
    dtype: torch.dtype
    """Torch dtype for kernel template instantiation."""
    compress_ratio: int = 1
    """``logical_block_size // shape_desc.bs``. ``1`` for non-compressed
    groups; greater for V4 compressor / indexer caches."""
    physical_chunk_size: int = 0
    """Physical slots per LMCache chunk
    (``lmcache_logical_chunk_size // compress_ratio``). Set by
    ``GPUCacheContext`` after construction."""
    logical_block_size: int = 0
    """Scheduler-side tokens per block. Equals ``shape_desc.bs`` when
    no per-layer hint is supplied; otherwise carries the engine's
    ``KVCacheSpec.block_size`` for layers in this group, which is what
    ``LoadStoreOp.block_ids`` is stride-compatible with."""
    kv_cache_group_id: int = 0
    """Engine-side block-ID namespace handle. Layers in different
    namespaces cannot share a kernel launch even when every other field
    matches (e.g. V4's two SWA gids share spec but pull from disjoint
    ``BlockPool``-allocated IDs)."""
    sliding_window: int = 0
    """SWA window size in tokens, ``0`` for full attention. Non-zero
    activates the SWA-suffix-only optimization for this group: store/retrieve
    only the last ``ceil(sliding_window / logical_block_size)`` blocks per
    chunk."""

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


def _validate_uniform_layer_format(kv_caches: "DiscoverableKVCache") -> None:
    """Verify every layer tensor in ``kv_caches`` shares the same ``.dim()``.

    LMCache currently assumes a single
    :class:`~lmcache.c_ops.GPUKVFormat` per engine — format is detected
    once from the first tensor and reused for every kernel call. A
    mixed-format engine would silently miscompile every layer past the
    first, so we fail loudly at registration time if any leaf tensor's
    dim count diverges. The realistic mixed case is MLA (no TWO axis)
    vs MHA (TWO axis), which differ in dim count.

    Raises:
        ValueError: if two leaf tensors disagree on ``.dim()``.
    """
    leaf_dims: list[int] = []

    def _walk(node: "DiscoverableKVCache") -> None:
        if isinstance(node, torch.Tensor):
            leaf_dims.append(node.dim())
            return
        for sub in node:
            _walk(sub)

    _walk(kv_caches)
    if len(leaf_dims) <= 1:
        return
    first = leaf_dims[0]
    for i, d in enumerate(leaf_dims[1:], start=1):
        if d != first:
            raise ValueError(
                f"All layer tensors must share the same gpu_kv_format; "
                f"layer 0 has tensor dim {first} but layer {i} has dim "
                f"{d}. Mixed-format engines are not currently supported "
                f"by LMCache."
            )


class KVLayerGroupsManager:
    """Partition a model's KV layers into transfer-kernel dispatch units.

    Each layer in ``kv_caches`` is bucketed by its :data:`LayerGroupIdentity`;
    each bucket becomes one :class:`KVLayerGroupInfo`. Downstream consumers
    (``VLLMPagedMemGPUConnectorV3``, ``GPUCacheContext``, the multiprocess
    server) iterate ``self.kv_layer_groups`` and issue one transfer-kernel
    launch per group. The manager is a pure metadata object — no GPU buffers,
    no transfers. Layout parsing lives in
    :mod:`lmcache.v1.gpu_connector.utils`.
    """

    def __init__(
        self,
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_blocks: int,
        layout_hints: "LayoutHints | None" = None,
        lmcache_logical_chunk_size: int = 256,
    ) -> None:
        """Partition layers into groups keyed by :data:`LayerGroupIdentity`.

        Groups are emitted in the order of their first-appearing layer, so
        group indices are deterministic across runs.

        Args:
            kv_caches: KV cache structure (see
                :func:`normalize_kv_and_discover_format`).
            gpu_kv_format: Format returned by
                :func:`normalize_kv_and_discover_format`.
            num_blocks: Number of paged blocks. Stamped into every
                ``shape_desc.nb``. ``shape_desc.bs`` is discovered per-layer
                so compressed and non-compressed groups can coexist.
            layout_hints: Engine-provided hints. The manager reads:

                * ``inference_engine_logical_block_size`` — global fallback
                  used when the per-layer hint is absent.
                * ``per_layer_logical_block_size`` — list of length
                  ``num_layers``. Required when the engine emits mixed
                  scheduler block sizes (vLLM hybrid manager on V4).
                * ``per_layer_kv_cache_group_id`` — list of length
                  ``num_layers``. Splits layers that share spec but pull
                  block IDs from disjoint namespaces (V4 gids 1 and 2).
                * ``per_layer_sliding_window`` — list of length
                  ``num_layers``. Non-zero entries activate the
                  SWA-suffix-only optimization for that group.

                Absent hints collapse the identity to the legacy 5-tuple:
                ``compress_ratio=1``, ``logical_block_size=shape_desc.bs``,
                ``kv_cache_group_id=0``, ``sliding_window=0``.
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

        # Pull the global fallback hint. ``None`` is replaced by the first
        # group's physical ``bs`` after the grouping loop so the public
        # ``int`` contract holds.
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

        _validate_uniform_layer_format(kv_caches)

        # Resolve per-layer hints. Absent hints leave the fields at None
        # and the grouping loop falls back to the legacy 5-tuple identity
        # (one group per physical shape, namespace 0, full attention).
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

        # Emit groups in order of first-appearing layer for deterministic
        # group indices across runs.
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

        ``compress_ratio = logical_block_size // bs`` per group. When the
        engine provides neither the per-layer nor global hint (typically
        a non-vLLM engine or a tool/test setup), this falls back to
        ``compress_ratio = 1`` even if ``logical_block_size > bs``, to
        preserve the legacy non-compressed default.

        ``physical_chunk_size = lmcache_logical_chunk_size // compress_ratio``
        is the per-chunk physical slot count fed to the transfer kernel.

        Raises:
            ValueError: if ``logical_block_size`` is not a multiple of
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

        Single pass over ``kv_caches``. Hints default safely: absent
        ``per_layer_logical_bs`` falls back to physical ``bs``, absent
        ``per_layer_namespace`` to 0, absent ``per_layer_sliding_window``
        to 0 (full attention) — all three defaults collapse the 8-tuple
        to the legacy 5-tuple. The returned dict's value lists are passed
        by reference into ``KVLayerGroupInfo`` instances.
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

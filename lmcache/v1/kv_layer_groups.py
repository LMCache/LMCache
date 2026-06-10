# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.python_ops_fallback import set_shape_desc_dtype
from lmcache.utils import lmcache_deprecate
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache, LayoutHints
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

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


# The tuple that uniquely identifies a set of kernel-equivalent layers; one
# distinct identity becomes one LMCache KV group:
# ``(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype)``.
# The ``engine_group_idx`` slot is the engine group id (one paged-
# block address space). Block IDs are only meaningful within one such group, so
# layers from different groups must not share one LMCache group (and thus one
# transfer-kernel launch) even if their tensor shape and dtype match.
class KernelGroupIdentity(NamedTuple):
    kv_size: int
    num_heads: int
    head_size: int
    block_size: int
    engine_group_idx: int
    dtype: torch.dtype


LayerGroupIdentity = KernelGroupIdentity  # Alias for compatibility


# Sentinel ``per_layer_engine_group_idx`` value: a KV tensor tagged with it is
# excluded from every LMCache group (used for cross-layer KV-sharing layers; see
# ``create_engine_group_infos_from_vllm``).
EXCLUDED_ENGINE_GROUP = -1


def group_layers_by_identity(
    kv_caches: "DiscoverableKVCache",
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    num_layers: int,
    per_layer_engine_group_idx: Sequence[int] | None = None,
) -> list[tuple[LayerGroupIdentity, list[int]]]:
    """Partition layer indices by :data:`LayerGroupIdentity`.

    This helper is shared by vLLM-side LMCache group inflation and server-side
    ``KernelGroupInfo`` construction so both sides agree on group order.

    Args:
        kv_caches: Registered KV cache structure inspected for per-layer shape
            and dtype.
        gpu_kv_format: Format descriptor returned by
            :func:`normalize_kv_and_discover_format`, used to read heads/sizes.
        num_layers: Number of registered KV tensors to partition.
        per_layer_engine_group_idx: Optional per-registered-index engine
            block group id. When ``None`` every layer is treated as block group
            0 (non-hybrid); when present, layers from different engine block
            groups never share an identity even if their tensor shapes match.
            Layers whose value is ``EXCLUDED_ENGINE_GROUP`` are left out of all
            groups (e.g. cross-layer KV-sharing layers whose KV lives in their
            target owner's blocks).

    Returns:
        A list of ``(identity, layer_indices)`` pairs sorted by each group's
        first layer index, so the group order is deterministic and identical on
        both the vLLM and server sides.
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
        engine_group_idx = (
            per_layer_engine_group_idx[idx]
            if per_layer_engine_group_idx is not None
            else 0
        )
        # Skip layers explicitly excluded from grouping (e.g. cross-layer
        # KV-sharing layers, whose KV lives in their target owner's blocks).
        if engine_group_idx == EXCLUDED_ENGINE_GROUP:
            continue
        nh = 1 if mla else get_num_heads(kv_caches, gpu_kv_format, idx)
        hs = get_head_size(kv_caches, gpu_kv_format, idx)
        dt = get_dtype(kv_caches, gpu_kv_format, idx)
        bs = get_block_size(kv_caches, gpu_kv_format, idx)

        identity = LayerGroupIdentity(
            kv_size=kv_size,
            num_heads=nh,
            head_size=hs,
            block_size=bs,
            engine_group_idx=engine_group_idx,
            dtype=dt,
        )
        groups_dict[identity].append(idx)
    return sorted(groups_dict.items(), key=lambda kv: kv[1][0])


@dataclass
class KernelGroupInfo:
    """A single transfer-kernel dispatch unit: a set of KV layers that can
    ride one kernel launch with one ``PageBufferShapeDesc``.

    Membership is decided by :class:`KVLayerGroupsManager` according to
    :data:`LayerGroupIdentity`; every layer referenced by
    ``layer_indices`` shares the same
    ``(kv_size, num_heads, head_size, block_size, engine_group_idx,
    dtype)`` signature.
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
    block_stride_elems``) are stamped once at construction."""
    dtype: torch.dtype
    """Torch dtype of the KV cache tensors for this group. Used for
    kernel template instantiation; see class docstring for why we keep
    this alongside ``shape_desc.element_size``."""
    compress_ratio: int = 1
    """Logical-tokens-per-physical-slot for this group. ``1`` for
    non-compressed groups (one logical token per physical slot);
    greater than ``1`` for compressed groups where each physical slot
    packs ``compress_ratio`` logical tokens (e.g. DeepSeek V4
    compressor / indexer caches). Derived from
    ``inference_engine_logical_block_size`` carried in ``layout_hints``
    at :class:`KVLayerGroupsManager` construction time."""
    physical_chunk_size: int = 0
    """Number of *physical* slots in one LMCache chunk for this group
    (= ``lmcache_logical_chunk_size // compress_ratio``). This is what
    the block-level transfer kernel must be told, not the logical
    ``lmcache_logical_chunk_size`` which counts vLLM tokens. ``0``
    means the field has not been populated yet; ``GPUCacheContext``
    fills it in after construction once ``lmcache_logical_chunk_size``
    is known."""
    engine_group_idx: int = 0
    """Engine group index (paged-block address space). 0 for non-hybrid."""

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        sd = self.shape_desc
        return (
            f"KernelGroupInfo(layers={len(self.layer_indices)}, "
            f"indices={indices_repr}, "
            f"shape_desc=(kv={sd.kv_size}, nl={sd.nl}, nb={sd.nb}, "
            f"bs={sd.bs}, nh={sd.nh}, hs={sd.hs}, "
            f"element_size={sd.element_size}, "
            f"block_stride_elems={sd.block_stride_elems}), "
            f"dtype={self.dtype}, "
            f"compress_ratio={self.compress_ratio}, "
            f"physical_chunk_size={self.physical_chunk_size}, "
            f"engine_group_idx={self.engine_group_idx})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in this group."""
        return len(self.layer_indices)

    @property
    def hidden_dim_size(self) -> int:
        """Hidden dimension size (``num_heads * head_size``)."""
        return self.shape_desc.nh * self.shape_desc.hs


KVLayerGroupInfo = KernelGroupInfo  # Alias for compatibility


@dataclass
class ObjectGroupInfo:
    """Metadata for an 'object group'.

    An object group contains one or more kernel groups whose
    KV caches will be stored in the same memory object.

    This will be useful for dealing with sliding window or mamba
    KV caches that needs a different prefix matching logic from
    the full attention KV caches.
    """

    kernel_group_indices: list[int]
    """Indices of the kernel groups belonging to this object group, in the
    order they should be laid out in memory."""

    # NOTE: will add fields to indicate the "kv cache type" of this
    # object group in the follow-up PRs


class KVLayerGroupsManager:
    """Partition a model's KV layers into transfer-kernel dispatch units.

    At construction time, every layer in ``kv_caches`` is bucketed by its
    :data:`LayerGroupIdentity` (``(kv_size, num_heads, head_size,
    block_size, engine_group_idx, dtype)``). Each bucket becomes one
    :class:`KernelGroupInfo` holding the layer indices, a shared
    :class:`PageBufferShapeDesc`, and the group's torch dtype.

    Downstream consumers (``VLLMPagedMemGPUConnectorV3``,
    ``GPUCacheContext``, the multiprocess server) iterate
    ``self._kernel_groups`` and issue one transfer-kernel launch per
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
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        lmcache_logical_chunk_size: int = 256,
    ) -> None:
        """Partition layers into groups keyed by
        :data:`LayerGroupIdentity`.

        For each layer ``i`` in ``kv_caches``, read
        ``(kv_size, num_heads, head_size, dtype)`` via the format-aware
        accessors in ``utils.py``. Layers with identical identities are
        bucketed together; each bucket becomes one
        :class:`KernelGroupInfo`.

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
            layout_hints: Engine-provided hints. The manager only reads
                ``inference_engine_logical_block_size`` (logical tokens
                per inference-engine block) from it to derive each
                group's ``compress_ratio`` and ``physical_chunk_size``.
                ``None`` means every group is treated as non-compressed
                (``compress_ratio == 1``).
            engine_group_infos: LMCache-owned engine KV cache group
                metadata. When present, it is used to keep layers from
                different engine block-ID spaces in separate LMCache
                transfer groups.
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
        from lmcache.v1.multiprocess.group_view import get_engine_group_indices

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
        self._kernel_groups: list[KernelGroupInfo] = []
        self._object_groups: list[ObjectGroupInfo] = []

        num_layers = get_num_layers(kv_caches, gpu_kv_format)
        if num_layers == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        per_layer_engine_group_idx = get_engine_group_indices(
            engine_group_infos, num_layers
        )

        groups_by_identity = group_layers_by_identity(
            kv_caches, gpu_kv_format, num_layers, per_layer_engine_group_idx
        )

        # Emit groups in order of their first-appearing layer so that group
        # indices remain deterministic across runs.
        for group_idx, ((_, _, _, bs, engine_group_idx, dt), indices) in enumerate(
            groups_by_identity
        ):
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

            # Per-group logical block size: a group's own block_size can exceed
            # the global GCD hint (e.g. gemma-4 sliding=32, hint=16).
            # ``max(hint, bs)`` gives compress_ratio=1 for uncompressed groups
            # and the engine block size for compressed ones (bs < hint, DeepSeek).
            global_logical = self.inference_engine_logical_block_size_
            group_logical_block_size = (
                max(global_logical, bs) if global_logical is not None else None
            )

            # TODO (ApostaC): the code here is not very good.
            # Conceptually, KV Layer Group should not be aware of lmcache logical
            # chunk size at all.
            compress_ratio, physical_chunk_size = self._derive_compression_metadata(
                group_idx=group_idx,
                bs=bs,
                ie_logical_block_size=group_logical_block_size,
                lmcache_logical_chunk_size=lmcache_logical_chunk_size,
            )

            self._kernel_groups.append(
                KernelGroupInfo(
                    layer_indices=indices,
                    shape_desc=shape_desc,
                    dtype=dt,
                    compress_ratio=compress_ratio,
                    physical_chunk_size=physical_chunk_size,
                    engine_group_idx=engine_group_idx,
                )
            )

        self.inference_engine_logical_block_size_ = (
            self.inference_engine_logical_block_size_
            or self._kernel_groups[0].shape_desc.bs
        )

        logger.info(
            "KV layer groups: ---\n%s\n---",
            "\n".join(repr(g) for g in self._kernel_groups),
        )

        # Detect the object groups
        self._object_groups = self._detect_object_groups(engine_group_infos)

    @property
    def kernel_groups(self) -> list[KernelGroupInfo]:
        """List of :class:`KernelGroupInfo`, one per kernel group."""
        return self._kernel_groups

    @property
    @lmcache_deprecate("`kv_layer_groups` is an outdated alias for `kernel_groups`")
    def kv_layer_groups(self) -> list[KernelGroupInfo]:
        """List of :class:`KernelGroupInfo`, one per kernel group."""
        return self._kernel_groups

    @property
    def num_kernel_groups(self) -> int:
        """Number of :class:`KernelGroupInfo` entries.

        Zero if ``kv_caches`` had no layers at construction time.
        """
        return len(self._kernel_groups)

    @property
    def object_groups(self) -> list[ObjectGroupInfo]:
        """List of :class:`ObjectGroupInfo`, one per object group."""
        return self._object_groups

    @property
    def num_object_groups(self) -> int:
        """Number of :class:`ObjectGroupInfo` entries."""
        return len(self._object_groups)

    @property
    @lmcache_deprecate("`num_groups` is an outdated alias for `num_kernel_groups`")
    def num_groups(self) -> int:
        """Number of :class:`KernelGroupInfo` entries.

        Zero if ``kv_caches`` had no layers at construction time.
        """
        return len(self._kernel_groups)

    @property
    def inference_engine_logical_block_size(self) -> int:
        """Inference-engine-side logical block size.

        Taken from ``layout_hints`` at construction time, or falls back
        to the first group's physical ``bs`` when no hint is provided
        (non-vLLM engines, or vLLM without mixed-compression KV groups),
        in which case every group is treated as non-compressed.
        """
        return (
            self.inference_engine_logical_block_size_
            or self._kernel_groups[0].shape_desc.bs
        )

    def get_shape_desc(self, kernel_group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Return the :class:`PageBufferShapeDesc` for *kernel_group_idx*.

        Equivalent to ``self._kernel_groups[kernel_group_idx].shape_desc``.

        Args:
            kernel_group_idx: 0-based kernel group index.

        Raises:
            IndexError: If *kernel_group_idx* is out of range.
        """
        return self._kernel_groups[kernel_group_idx].shape_desc

    def get_physical_chunk_size(self, kernel_group_idx: int) -> int:
        """Return the per-chunk *physical* slot count for *kernel_group_idx*.

        Equivalent to
        ``self._kernel_groups[kernel_group_idx].physical_chunk_size``.
        For non-compressed groups this equals
        ``lmcache_logical_chunk_size``; for compressed groups it equals
        ``lmcache_logical_chunk_size // compress_ratio`` and is what the
        block-level transfer kernel must be told (the logical chunk size
        in *vLLM tokens* is not what the kernel addresses).

        Args:
            kernel_group_idx: 0-based kernel group index.

        Raises:
            IndexError: If *kernel_group_idx* is out of range.
        """
        return self._kernel_groups[kernel_group_idx].physical_chunk_size

    def calculate_num_blocks(self, kernel_group_idx: int, num_tokens: int) -> int:
        """Calculate the number of blocks for a given number of tokens in a
        specified kernel group.

        Args:
            kernel_group_idx: 0-based index of the kernel group.
            num_tokens: The total number of tokens to be processed for the group.

        Returns:
            The number of blocks.

        Raises:
            IndexError: If *kernel_group_idx* is out of range.
        """
        group = self._kernel_groups[kernel_group_idx]
        num_physical_slots = num_tokens // group.compress_ratio
        return num_physical_slots // group.shape_desc.bs

    ### Helper methods
    def _detect_object_groups(
        self, engine_group_infos: "Sequence[EngineGroupInfo]"
    ) -> list[ObjectGroupInfo]:
        """Detect object groups based on the provided engine group infos.

        Args:
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Returns:
            A list of ObjectGroupInfo instances representing the detected object groups.
        """
        # TODO: add the real object group detection logic based on
        # the attention type metadata in the engine group infos once it's
        # available.
        # Now, we are using a single object group, which means
        # all kernel groups' KV caches will be stored in the same memory object.
        return [
            ObjectGroupInfo(kernel_group_indices=list(range(len(self._kernel_groups))))
        ]

    @staticmethod
    def _derive_compression_metadata(
        group_idx: int,
        bs: int,
        ie_logical_block_size: "int | None",
        lmcache_logical_chunk_size: int,
    ) -> tuple[int, int]:
        """Resolve ``(compress_ratio, physical_chunk_size)`` for one group.

        ``compress_ratio`` falls back to ``1`` when
        ``ie_logical_block_size`` is absent (no compression info
        available); otherwise it equals
        ``ie_logical_block_size // bs`` and the divisibility invariants
        are enforced loudly. ``physical_chunk_size`` is then
        ``lmcache_logical_chunk_size // compress_ratio``, the per-chunk
        physical slot count fed to the block-level transfer kernel.
        """
        if ie_logical_block_size is None:
            compress_ratio = 1
        else:
            if ie_logical_block_size % bs != 0:
                raise ValueError(
                    f"inference engine logical block size "
                    f"{ie_logical_block_size} must be a multiple of "
                    f"group {group_idx} physical slot count {bs}"
                )
            compress_ratio = ie_logical_block_size // bs
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
                "(inference_engine_logical_block_size=%d -> "
                "slots=%d, compress_ratio=%d, physical_chunk_size=%d)",
                group_idx,
                ie_logical_block_size,
                bs,
                compress_ratio,
                physical_chunk_size,
            )
        return compress_ratio, physical_chunk_size


# ------------------------------------------------------------------ #
#  CLI shape-spec parser                                               #
# ------------------------------------------------------------------ #


def parse_kvcache_shape_spec(
    spec_str: str,
) -> list[KernelGroupInfo]:
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

    When consumed by the ``lmcache bench server`` CLI, ``NB``/``BS``
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
        A list of :class:`KernelGroupInfo`, one per group.

    Raises:
        ValueError: Malformed spec, unknown dtype, or a shape with a
            wrong number of dimensions.
    """
    if not spec_str:
        raise ValueError("KV shape specification cannot be empty")

    groups: list[KernelGroupInfo] = []
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
        set_shape_desc_dtype(shape_desc, dtype)

        indices = list(range(layer_offset, layer_offset + layer_count))
        groups.append(
            KernelGroupInfo(
                layer_indices=indices,
                shape_desc=shape_desc,
                dtype=dtype,
            )
        )
        layer_offset += layer_count

    if not groups:
        raise ValueError("No valid layer groups found in spec")

    return groups


def format_kvcache_shape_spec(groups: list[KernelGroupInfo]) -> str:
    """Format layer groups back into a ``--kvcache-shape-spec`` string.

    This is the inverse of :func:`parse_kvcache_shape_spec`; the
    result is round-trip safe (i.e. ``parse(format(x)) == x`` for any
    ``x`` that ``parse`` would produce).

    The returned string is also human-readable and is used by the
    ``lmcache bench server`` CLI to echo the resolved KV cache
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

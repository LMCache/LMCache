# SPDX-License-Identifier: Apache-2.0
"""Declarative KV-cache layout descriptor (RFC #3560, step 1).

A :class:`KVLayoutDescriptor` states a physical KV-cache layout as data: an
ordered partition of the logical axes (``KV``, ``L``, ``B``, ``N``, ``H``,
``C``) into physical tensor dimensions, plus the list grouping, the K/V packing
mode, dtypes, sparse per-dim stride overrides, and an optional quantization
spec. Everything the classification predicates answer today (``is_mla``,
``is_hnd``, the structural shape, ...) is derived from that structure instead
of being declared per format.

This module is **additive**: ``EngineKVFormat`` stays the authoritative
currency everywhere in LMCache, and nothing existing changes behavior. The
compat shim is the bijection at the bottom -- ``from_engine_kv_format`` /
``to_engine_kv_format`` map every current enum member to a canonical
descriptor and back, pinned by
``tests/v1/gpu_connector/test_kv_layout_descriptor.py`` in the style of
``test_kv_format_classification.py``. Later steps can emit descriptors from
detection and derive the enum as a view without changing this contract.

The module deliberately imports neither ``torch`` nor the compiled
``lmcache.lmcache_native`` extension at module level (the one use is a lazy
import inside :func:`to_engine_kv_format`), so the descriptor vocabulary
stays usable in contexts where the native extension is absent.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    import lmcache.lmcache_native as lmcache_native


class Axis(Enum):
    """Logical KV-cache axes (RFC #3560 coordinates plus ``KV``).

    ``N`` is the addressing unit inside a block: tokens per block for paged
    attention caches, number of recurrent states (1) for state-space caches.
    ``C`` is the per-head state content: ``head_size`` for split K/V,
    ``2 * head_size`` when K/V are fused into the content dim.
    """

    KV = "KV"  # the K/V pair
    L = "L"  # layers
    B = "B"  # blocks / pages
    N = "N"  # tokens per block / number of states
    H = "H"  # KV heads
    C = "C"  # per-head state content


class Grouping(Enum):
    """How the layout groups tensors outside the per-tensor dims.

    The list levels carry logical axes that therefore must not appear in
    ``dims``: ``PER_LAYER`` carries ``L``; ``KV_LISTS`` carries ``KV`` and
    ``L``.
    """

    SINGLE_TENSOR = "single_tensor"  # everything in one tensor
    PER_LAYER = "per_layer"  # list[NL] of per-layer tensors
    KV_LISTS = "kv_lists"  # [key_layers, value_layers] two-list form


class KVPacking(Enum):
    """Where the K/V pair lives relative to the per-token content."""

    # K and V are separately addressable: KV is a real axis, either its own
    # dim outside the token content or the KV_LISTS list level. kv_size == 2.
    SPLIT = "split"
    # K and V are packed inside the per-token content region (at or after
    # both N's and H's dims), so a token's KV is one contiguous run and
    # kv_size == 1.
    FUSED = "fused"
    # One shared plane, no K/V distinction (MLA latent cache). KV does not
    # appear in dims. kv_size == 1.
    SHARED = "shared"


class ScalePlacement(Enum):
    """Where a quantized cache stores its scales relative to the values."""

    # Per block: all tokens' values first, then all tokens' scales
    # (the DSA indexer k-cache's [BS x values][BS x scales] page regions).
    BLOCK_REGION = "block_region"
    # In-band inside each head's widened row (vLLM fp8 per-token-head).
    INLINE_PER_HEAD = "inline_per_head"


@dataclass(frozen=True)
class QuantSpec:
    """Quantization facts for a cache whose bytes are not plain elements.

    Attributes:
        scale_dtype: Dtype of the stored scales, e.g. ``"float32"``.
        values_per_scale: Number of quantized values covered by one scale.
        placement: Where the scales sit relative to the values.
    """

    scale_dtype: str
    values_per_scale: int
    placement: ScalePlacement


# ``storage_dtype`` value for canonical (enum-derived) descriptors: the enum
# does not constrain the element type, a concrete registration fills it.
DTYPE_UNSPECIFIED = ""

_EMPTY_STRIDES: Mapping[int, int] = MappingProxyType({})


@dataclass(frozen=True)
class KVLayoutDescriptor:
    """A physical KV-cache layout stated as data.

    Attributes:
        extents: Sizes of logical axes, where known. Canonical descriptors
            carry only the structurally forced entries (``KV``, ``H`` for MLA,
            ``C`` for the fixed-width indexer cache); a descriptor built from
            a concrete registration binds every axis it materializes.
        dims: Physical dims of one tensor, outermost first. Each entry is a
            tuple of logical axes folded row-major into that dim: ``(B, N)``
            is SGLang's fused page-buffer dim, ``(KV, C)`` is the unified
            content dim. A one-axis tuple is a plain dim; the empty tuple is
            a materialized size-1 dim that carries no logical axis (the
            RBLN singleton). Axes carried by the ``grouping`` list levels
            (``L`` for PER_LAYER, ``KV`` and ``L`` for KV_LISTS) must not
            appear.
        grouping: List structure around the tensor(s), see :class:`Grouping`.
        kv_packing: Where the K/V pair lives, see :class:`KVPacking`.
        storage_dtype: What ``tensor.dtype`` reports, e.g. ``"uint8"``.
            :data:`DTYPE_UNSPECIFIED` in canonical descriptors.
        logical_dtype: What the bytes mean when it differs from
            ``storage_dtype`` (quantized caches), else ``None``.
        dim_strides: Sparse per-dim physical strides in storage elements,
            keyed by index into ``dims``. A missing key means tight. This
            generalizes the transfer path's ``block_stride_elems`` (dim-0
            padding for pool-sharing layouts) to any padded dim.
        quant: Quantization facts, or ``None`` for plain caches.

    Raises:
        ValueError: If the structure is inconsistent (duplicate axes, axes
            that the grouping or packing forbids in ``dims``, out-of-range
            ``dim_strides`` keys, non-positive extents).
    """

    extents: Mapping[Axis, int]
    dims: tuple[tuple[Axis, ...], ...]
    grouping: Grouping
    kv_packing: KVPacking
    storage_dtype: str
    logical_dtype: str | None = None
    dim_strides: Mapping[int, int] = field(default_factory=lambda: _EMPTY_STRIDES)
    quant: QuantSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "extents", MappingProxyType(dict(self.extents)))
        object.__setattr__(
            self, "dim_strides", MappingProxyType(dict(self.dim_strides))
        )
        self._validate()

    def _validate(self) -> None:
        seen: set[Axis] = set()
        for fold in self.dims:
            if not isinstance(fold, tuple):
                raise ValueError(f"dims entries must be tuples, got {fold!r}")
            for axis in fold:
                if not isinstance(axis, Axis):
                    raise ValueError(f"dims may only contain Axis, got {axis!r}")
                if axis in seen:
                    raise ValueError(f"axis {axis} appears in more than one dim")
                seen.add(axis)

        if Axis.C not in seen:
            raise ValueError("the content axis C must be materialized in dims")

        carried = {
            Grouping.SINGLE_TENSOR: frozenset[Axis](),
            Grouping.PER_LAYER: frozenset({Axis.L}),
            Grouping.KV_LISTS: frozenset({Axis.KV, Axis.L}),
        }[self.grouping]
        overlap = carried & seen
        if overlap:
            raise ValueError(
                f"{self.grouping} carries {sorted(a.name for a in overlap)} at "
                "the list level; the axis must not also appear in dims"
            )

        if self.kv_packing is KVPacking.SHARED and Axis.KV in seen:
            raise ValueError("SHARED packing must not materialize a KV axis")
        if self.kv_packing is KVPacking.SPLIT:
            if self.grouping is not Grouping.KV_LISTS and Axis.KV not in seen:
                raise ValueError(
                    "SPLIT packing needs a KV axis in dims (or KV_LISTS grouping)"
                )
        if self.kv_packing is KVPacking.FUSED:
            kv_dim = self.axis_dim(Axis.KV)
            if kv_dim is None:
                raise ValueError("FUSED packing needs the KV axis in dims")
            for axis in (Axis.N, Axis.H):
                axis_dim = self.axis_dim(axis)
                if axis_dim is not None and axis_dim > kv_dim:
                    raise ValueError(
                        f"FUSED packing puts KV inside the content region, but "
                        f"{axis.name}'s dim {axis_dim} comes after KV's {kv_dim}"
                    )

        for dim_idx in self.dim_strides:
            if not 0 <= dim_idx < len(self.dims):
                raise ValueError(
                    f"dim_strides key {dim_idx} out of range for {len(self.dims)} dims"
                )
        for axis, extent in self.extents.items():
            if not isinstance(axis, Axis):
                raise ValueError(f"extents keys must be Axis, got {axis!r}")
            if extent <= 0:
                raise ValueError(f"extent of {axis.name} must be positive: {extent}")

    # ── Structure accessors ────────────────────────────────────────────

    def axis_dim(self, axis: Axis) -> int | None:
        """Return the index of the dim whose fold contains *axis*, or None.

        Args:
            axis: The logical axis to locate.

        Returns:
            The 0-based index into ``dims``, or ``None`` when the axis is not
            materialized in this tensor (carried by a list level, dropped, or
            packed away).
        """
        for dim_idx, fold in enumerate(self.dims):
            if axis in fold:
                return dim_idx
        return None

    def dim_extent(self, dim_idx: int) -> int:
        """Return the size of physical dim *dim_idx* (product of its fold).

        Args:
            dim_idx: Index into ``dims``.

        Returns:
            The dim's extent; the empty fold has extent 1.

        Raises:
            ValueError: If any axis in the fold has no bound extent.
        """
        size = 1
        for axis in self.dims[dim_idx]:
            if axis not in self.extents:
                raise ValueError(f"axis {axis.name} has no bound extent")
            size *= self.extents[axis]
        return size

    def resolved_strides(self) -> tuple[int, ...]:
        """Return per-dim strides in storage elements, innermost tight.

        Strides are resolved outermost-last: each dim's stride is its inner
        neighbor's stride times that neighbor's extent, unless ``dim_strides``
        overrides it. An override therefore also shifts every dim outside it,
        which is how one ``B``-stride entry describes a per-layer view into a
        padded multi-layer pool.

        Returns:
            One stride per entry of ``dims``, outermost first.

        Raises:
            ValueError: If any materialized axis has no bound extent.
        """
        strides = [0] * len(self.dims)
        inner = 1
        for dim_idx in range(len(self.dims) - 1, -1, -1):
            strides[dim_idx] = self.dim_strides.get(dim_idx, inner)
            inner = strides[dim_idx] * self.dim_extent(dim_idx)
        return tuple(strides)

    # ── Derived classification facts ───────────────────────────────────
    # One definition each, derived from structure. The canonical descriptors
    # reproduce the per-format facts pinned in csrc/engine_kv_format.h and on
    # the KVFormatSpec classes (see the round-trip test).

    @property
    def is_cross_layer(self) -> bool:
        """All layers in one fused tensor."""
        return self.grouping is Grouping.SINGLE_TENSOR

    @property
    def is_kv_list(self) -> bool:
        """Keys and values in two top-level lists."""
        return self.grouping is Grouping.KV_LISTS

    @property
    def is_layer_list(self) -> bool:
        """One list entry per layer."""
        return self.grouping is Grouping.PER_LAYER

    @property
    def is_mla(self) -> bool:
        """Single shared latent plane, no K/V split."""
        return self.kv_packing is KVPacking.SHARED

    @property
    def is_hnd(self) -> bool:
        """Heads stored before block tokens within a tensor."""
        h_dim = self.axis_dim(Axis.H)
        n_dim = self.axis_dim(Axis.N)
        return h_dim is not None and n_dim is not None and h_dim < n_dim

    @property
    def is_fused_packed(self) -> bool:
        """K/V packed inside the per-token content region."""
        return self.kv_packing is KVPacking.FUSED

    @property
    def is_two_major(self) -> bool:
        """The size-2 K/V axis precedes the block axis within the tensor."""
        if self.kv_packing is not KVPacking.SPLIT:
            return False
        kv_dim = self.axis_dim(Axis.KV)
        b_dim = self.axis_dim(Axis.B)
        return kv_dim is not None and b_dim is not None and kv_dim < b_dim

    @property
    def is_pbs_fused(self) -> bool:
        """Blocks and tokens folded into one page-buffer axis."""
        return any(Axis.B in fold and Axis.N in fold for fold in self.dims)

    @property
    def kv_size(self) -> int:
        """Number of separately addressed K/V planes: 2 for SPLIT, else 1."""
        return 2 if self.kv_packing is KVPacking.SPLIT else 1


# ── Bijection with EngineKVFormat ──────────────────────────────────────
# One canonical descriptor per current enum member, keyed by member name (the
# native and pure-Python enums are distinct types with the same names, cf.
# specs/registry.py). Canonical descriptors bind only structurally forced
# extents; dtype stays DTYPE_UNSPECIFIED except where the format itself pins
# it (the blocked-scale indexer cache).

_KV = Axis.KV
_L = Axis.L
_B = Axis.B
_N = Axis.N
_H = Axis.H
_C = Axis.C

_SPLIT_EXTENTS: Mapping[Axis, int] = MappingProxyType({_KV: 2})
_MLA_EXTENTS: Mapping[Axis, int] = MappingProxyType({_KV: 1, _H: 1})


def _split(
    dims: tuple[tuple[Axis, ...], ...],
    grouping: Grouping,
) -> KVLayoutDescriptor:
    """Build a canonical split-K/V descriptor.

    Args:
        dims: Physical dims, outermost first.
        grouping: List structure around the tensor(s).

    Returns:
        The canonical descriptor with ``KV`` extent 2 and unbound dtype.
    """
    return KVLayoutDescriptor(
        extents=_SPLIT_EXTENTS,
        dims=dims,
        grouping=grouping,
        kv_packing=KVPacking.SPLIT,
        storage_dtype=DTYPE_UNSPECIFIED,
    )


def _fused(dims: tuple[tuple[Axis, ...], ...]) -> KVLayoutDescriptor:
    """Build a canonical fused-K/V per-layer descriptor.

    Args:
        dims: Physical dims, outermost first, with ``KV`` in the content
            region.

    Returns:
        The canonical descriptor with ``KV`` extent 2 and unbound dtype.
    """
    return KVLayoutDescriptor(
        extents=_SPLIT_EXTENTS,
        dims=dims,
        grouping=Grouping.PER_LAYER,
        kv_packing=KVPacking.FUSED,
        storage_dtype=DTYPE_UNSPECIFIED,
    )


ENGINE_KV_FORMAT_DESCRIPTORS: Mapping[str, KVLayoutDescriptor] = MappingProxyType(
    {
        # vLLM cross-layer pool: [NB, NL, 2, BS, NH, HS] in one tensor.
        "NB_NL_TWO_BS_NH_HS": _split(
            ((_B,), (_L,), (_KV,), (_N,), (_H,), (_C,)), Grouping.SINGLE_TENSOR
        ),
        # vLLM flash attention: NL x [2, NB, BS, NH, HS].
        "NL_X_TWO_NB_BS_NH_HS": _split(
            ((_KV,), (_B,), (_N,), (_H,), (_C,)), Grouping.PER_LAYER
        ),
        # vLLM flash infer: NL x [NB, 2, BS, NH, HS].
        "NL_X_NB_TWO_BS_NH_HS": _split(
            ((_B,), (_KV,), (_N,), (_H,), (_C,)), Grouping.PER_LAYER
        ),
        # vLLM MLA: NL x [NB, BS, HS], head axis dropped.
        "NL_X_NB_BS_HS": KVLayoutDescriptor(
            extents=_MLA_EXTENTS,
            dims=((_B,), (_N,), (_C,)),
            grouping=Grouping.PER_LAYER,
            kv_packing=KVPacking.SHARED,
            storage_dtype=DTYPE_UNSPECIFIED,
        ),
        # SGLang MHA: 2 x NL x [PBS, NH, HS], page buffer fused.
        "TWO_X_NL_X_NBBS_NH_HS": _split(((_B, _N), (_H,), (_C,)), Grouping.KV_LISTS),
        # SGLang MLA: NL x [PBS, 1, HS], latent head materialized as 1.
        "NL_X_NBBS_ONE_HS": KVLayoutDescriptor(
            extents=_MLA_EXTENTS,
            dims=((_B, _N), (_H,), (_C,)),
            grouping=Grouping.PER_LAYER,
            kv_packing=KVPacking.SHARED,
            storage_dtype=DTYPE_UNSPECIFIED,
        ),
        # vLLM flash attention HND: NL x [2, NB, NH, BS, HS].
        "NL_X_TWO_NB_NH_BS_HS": _split(
            ((_KV,), (_B,), (_H,), (_N,), (_C,)), Grouping.PER_LAYER
        ),
        # vLLM flash infer HND: NL x [NB, 2, NH, BS, HS].
        "NL_X_NB_TWO_NH_BS_HS": _split(
            ((_B,), (_KV,), (_H,), (_N,), (_C,)), Grouping.PER_LAYER
        ),
        # TRT-LLM cross-layer HND: [NB, NL, 2, NH, BS, HS] in one tensor.
        "NB_NL_TWO_NH_BS_HS": _split(
            ((_B,), (_L,), (_KV,), (_H,), (_N,), (_C,)), Grouping.SINGLE_TENSOR
        ),
        # SGLang MHA via the MP daemon: 2 x NL x [NB, BS, NH, HS].
        "TWO_X_NL_X_NB_BS_NH_HS": _split(
            ((_B,), (_N,), (_H,), (_C,)), Grouping.KV_LISTS
        ),
        # DEPRECATED fused HND: NL x [NB, NH, BS, 2, HS], K/V axis materialized.
        "NL_X_NB_NH_BS_TWO_HS": _fused(((_B,), (_H,), (_N,), (_KV,), (_C,))),
        # DEPRECATED fused NHD: NL x [NB, BS, NH, 2, HS].
        "NL_X_NB_BS_NH_TWO_HS": _fused(((_B,), (_N,), (_H,), (_KV,), (_C,))),
        # vLLM unified KV cache HND: NL x [NB, NH, BS, CS], CS = (KV, C) folded.
        "NL_X_NB_NH_BS_CS": _fused(((_B,), (_H,), (_N,), (_KV, _C))),
        # vLLM unified KV cache NHD: NL x [NB, BS, NH, CS].
        "NL_X_NB_BS_NH_CS": _fused(((_B,), (_N,), (_H,), (_KV, _C))),
        # vLLM DSA indexer k-cache: NL x [NB, BS, 132] uint8; per block all
        # tokens' 128 fp8 values then all tokens' fp32 scales, so per-token
        # addressing does not exist and C's 132 is a format constant.
        "NL_X_NB_BSV_BSS": KVLayoutDescriptor(
            extents=MappingProxyType({_KV: 1, _H: 1, _C: 132}),
            dims=((_B,), (_N,), (_C,)),
            grouping=Grouping.PER_LAYER,
            kv_packing=KVPacking.SHARED,
            storage_dtype="uint8",
            logical_dtype="float8_e4m3fn",
            quant=QuantSpec(
                scale_dtype="float32",
                values_per_scale=128,
                placement=ScalePlacement.BLOCK_REGION,
            ),
        ),
        # vLLM-RBLN HND: NL x [2, NB, NH, 1, BS, HS]; the empty fold is the
        # backend-required singleton between heads and tokens.
        "NL_X_TWO_NB_NH_ONE_BS_HS": _split(
            ((_KV,), (_B,), (_H,), (), (_N,), (_C,)), Grouping.PER_LAYER
        ),
    }
)


def _structural_key(
    desc: KVLayoutDescriptor,
) -> tuple[Grouping, KVPacking, tuple[tuple[Axis, ...], ...], ScalePlacement | None]:
    """Return the part of a descriptor the enum can represent.

    Args:
        desc: The descriptor to project.

    Returns:
        ``(grouping, kv_packing, dims, scale placement)``. Extents, dtypes
        and stride overrides are dropped: the enum is a lossy view, so any
        concrete descriptor sharing a canonical structure projects to that
        member.
    """
    return (
        desc.grouping,
        desc.kv_packing,
        desc.dims,
        desc.quant.placement if desc.quant is not None else None,
    )


_NAME_BY_STRUCTURE: Mapping[
    tuple[Grouping, KVPacking, tuple[tuple[Axis, ...], ...], ScalePlacement | None],
    str,
] = MappingProxyType(
    {_structural_key(desc): name for name, desc in ENGINE_KV_FORMAT_DESCRIPTORS.items()}
)


def from_engine_kv_format(
    fmt: "lmcache_native.EngineKVFormat",
) -> KVLayoutDescriptor:
    """Return the canonical descriptor for an ``EngineKVFormat`` member.

    Args:
        fmt: An ``EngineKVFormat`` member (native or pure-Python fallback;
            only its ``name`` is read).

    Returns:
        The canonical :class:`KVLayoutDescriptor` for the member.

    Raises:
        ValueError: If the member has no descriptor (a new format must be
            added to ``ENGINE_KV_FORMAT_DESCRIPTORS`` deliberately).
    """
    desc = ENGINE_KV_FORMAT_DESCRIPTORS.get(fmt.name)
    if desc is None:
        raise ValueError(f"EngineKVFormat member without a descriptor: {fmt}")
    return desc


def to_engine_kv_format_name(desc: KVLayoutDescriptor) -> str:
    """Return the ``EngineKVFormat`` member name matching a descriptor.

    Matching is structural (grouping, packing, dims, scale placement), so a
    concrete descriptor with bound extents, dtypes or stride overrides maps
    to the same member as its canonical template.

    Args:
        desc: The descriptor to classify.

    Returns:
        The matching member name.

    Raises:
        ValueError: If no current member has this structure.
    """
    name = _NAME_BY_STRUCTURE.get(_structural_key(desc))
    if name is None:
        raise ValueError(
            f"no EngineKVFormat member for layout {desc.grouping.value} / "
            f"{desc.kv_packing.value} with dims {desc.dims}"
        )
    return name


def to_engine_kv_format(desc: KVLayoutDescriptor) -> "lmcache_native.EngineKVFormat":
    """Return the ``EngineKVFormat`` member matching a descriptor.

    The native extension is imported lazily so that this module stays
    importable without it; use :func:`to_engine_kv_format_name` where the
    member name suffices.

    Args:
        desc: The descriptor to classify.

    Returns:
        The matching native ``EngineKVFormat`` member.

    Raises:
        ValueError: If no current member has this structure.
    """
    # First Party
    import lmcache.lmcache_native as lmcache_native

    return getattr(lmcache_native.EngineKVFormat, to_engine_kv_format_name(desc))

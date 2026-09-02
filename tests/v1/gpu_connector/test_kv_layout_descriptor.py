# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for the KV layout descriptor bijection (RFC #3560 step 1).

Companion to ``test_kv_format_classification.py``: that file pins the
per-format facts declared on the specs and in ``csrc/engine_kv_format.h``;
this file pins that every ``EngineKVFormat`` member has exactly one canonical
:class:`KVLayoutDescriptor`, that the mapping round-trips both ways for every
member (checked exhaustively, not sampled), and that the facts *derived* from
descriptor structure reproduce the facts *declared* on the specs. A new enum
member fails ``test_every_format_has_descriptor`` until it gets a descriptor,
mirroring how ``test_every_format_is_pinned`` guards the golden tables.

The last two tests tie the descriptor to the ``KVLayoutName`` vocabulary
(``test_blocks_first_detection.py``): a standardized name is an axis order,
and a blocks-first per-layer view is the unified-cache entry plus one block
stride override, projecting onto the member ``detect_format`` returns.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import (
    describe_shape,
    detect_format,
    get_spec_class,
)
from lmcache.v1.gpu_connector.kv_format.descriptor import (
    ENGINE_KV_FORMAT_DESCRIPTORS,
    Axis,
    Grouping,
    KVLayoutDescriptor,
    KVPacking,
    from_engine_kv_format,
    kv_layout_axes,
    to_engine_kv_format,
    to_engine_kv_format_name,
    with_block_stride,
)
from lmcache.v1.gpu_connector.kv_format.types import KV_LAYOUT_NAMES
import lmcache.lmcache_native as lmcache_native

F = lmcache_native.EngineKVFormat

# The member vLLM's unified (fused K/V) per-layer cache classifies as under
# each kv_layout hint LMCache accepts (kv_format/detectors/vllm.py).
UNIFIED_MEMBER_BY_KV_LAYOUT = {
    "NHD": "NL_X_NB_BS_NH_CS",
    "HND": "NL_X_NB_NH_BS_CS",
    "BLNHC": "NL_X_NB_BS_NH_CS",
    "BLHNC": "NL_X_NB_NH_BS_CS",
}


def _all_formats():
    return [v for v in vars(F).values() if isinstance(v, F)]


def test_every_format_has_descriptor():
    # A new EngineKVFormat must get a canonical descriptor deliberately.
    assert {fmt.name for fmt in _all_formats()} == set(ENGINE_KV_FORMAT_DESCRIPTORS)


def test_round_trip_holds_for_every_format():
    # Exhaustive in both directions: enum -> descriptor -> the same enum
    # member, and canonical descriptor -> name -> the same descriptor.
    for fmt in _all_formats():
        desc = from_engine_kv_format(fmt)
        assert to_engine_kv_format(desc) == fmt, fmt
        assert to_engine_kv_format_name(desc) == fmt.name, fmt
    for name, desc in ENGINE_KV_FORMAT_DESCRIPTORS.items():
        assert from_engine_kv_format(getattr(F, name)) is desc


def test_descriptors_are_structurally_distinct():
    # The mapping is a bijection: no two members may share a structure.
    def structure(desc: KVLayoutDescriptor):
        return (desc.grouping, desc.kv_packing, desc.dims, desc.quant)

    names = list(ENGINE_KV_FORMAT_DESCRIPTORS)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            da = ENGINE_KV_FORMAT_DESCRIPTORS[a]
            db = ENGINE_KV_FORMAT_DESCRIPTORS[b]
            assert structure(da) != structure(db), f"{a} and {b} share a structure"


def test_derived_facts_match_spec_facts():
    # The spec classes declare each format's facts (mirrored in
    # csrc/engine_kv_format.h); the descriptor derives the same facts from
    # structure. Both routes must agree for every member.
    for fmt in _all_formats():
        spec = get_spec_class(fmt)
        desc = from_engine_kv_format(fmt)
        declared = (
            spec.is_cross_layer,
            spec.is_kv_list,
            spec.is_layer_list,
            spec.is_mla,
            spec.is_hnd,
            spec.is_fused_packed,
            spec.is_two_major,
            spec.is_pbs_fused,
            spec.is_kv_second_tuple,
        )
        derived = (
            desc.is_cross_layer,
            desc.is_kv_list,
            desc.is_layer_list,
            desc.is_mla,
            desc.is_hnd,
            desc.is_fused_packed,
            desc.is_two_major,
            desc.is_pbs_fused,
            desc.is_kv_second_tuple,
        )
        assert derived == declared, f"{fmt}: derived {derived}, declared {declared}"


def test_kv_size_matches_packing():
    # kv_size == 1 exactly for shared (MLA) and fused-packed formats.
    for fmt in _all_formats():
        spec = get_spec_class(fmt)
        desc = from_engine_kv_format(fmt)
        expected = 1 if (spec.is_mla or spec.is_fused_packed) else 2
        assert desc.kv_size == expected, fmt


def test_dims_rank_matches_shape_grammar():
    # describe_shape renders the per-entry tensor rank from the enum name;
    # the descriptor's dims must have the same rank (folds included: NBBS is
    # one dim, ONE is one dim, CS is one dim).
    for fmt in _all_formats():
        desc = from_engine_kv_format(fmt)
        inner = describe_shape(fmt).rsplit("[", 1)[1].rstrip("]")
        rank = len(inner.split(", "))
        assert len(desc.dims) == rank, f"{fmt}: {describe_shape(fmt)} vs {desc.dims}"


def test_stride_override_describes_padded_pool_view():
    # The kvcached per-layer view: one [NB, 2, BS, NH, HS] view per layer
    # into a contiguous [NB, NL, 2, BS, NH, HS] pool, so the block axis
    # strides over all NL layers. One dim_strides entry states that.
    nl, nb, bs, nh, hs = 4, 8, 16, 2, 64
    desc = KVLayoutDescriptor(
        extents={Axis.KV: 2, Axis.B: nb, Axis.N: bs, Axis.H: nh, Axis.C: hs},
        dims=((Axis.B,), (Axis.KV,), (Axis.N,), (Axis.H,), (Axis.C,)),
        grouping=Grouping.PER_LAYER,
        kv_packing=KVPacking.SPLIT,
        storage_dtype="bfloat16",
        dim_strides={0: nl * 2 * bs * nh * hs},
    )
    assert desc.resolved_strides() == (
        nl * 2 * bs * nh * hs,  # B: overridden, spans all layers
        bs * nh * hs,  # KV: tight
        nh * hs,  # N: tight
        hs,  # H: tight
        1,  # C: tight
    )
    # Structural matching ignores the override: it is still the flash-infer
    # per-layer member.
    assert to_engine_kv_format_name(desc) == "NL_X_NB_TWO_BS_NH_HS"
    # Tight variant of the same structure resolves without the override.
    tight = KVLayoutDescriptor(
        extents={Axis.KV: 2, Axis.B: nb, Axis.N: bs, Axis.H: nh, Axis.C: hs},
        dims=((Axis.B,), (Axis.KV,), (Axis.N,), (Axis.H,), (Axis.C,)),
        grouping=Grouping.PER_LAYER,
        kv_packing=KVPacking.SPLIT,
        storage_dtype="bfloat16",
    )
    assert tight.resolved_strides() == (
        2 * bs * nh * hs,
        bs * nh * hs,
        nh * hs,
        hs,
        1,
    )


def test_with_block_stride_needs_a_plain_block_dim():
    # SGLang folds B with N into the page-buffer dim: no single B stride.
    with pytest.raises(ValueError, match="plain dim"):
        with_block_stride(ENGINE_KV_FORMAT_DESCRIPTORS["TWO_X_NL_X_NBBS_NH_HS"], 64)
    with pytest.raises(ValueError, match="non-negative"):
        with_block_stride(ENGINE_KV_FORMAT_DESCRIPTORS["NL_X_NB_BS_HS"], -1)


def test_validation_rejects_inconsistent_structures():
    def build(**overrides):
        kwargs = dict(
            extents={},
            dims=((Axis.B,), (Axis.KV,), (Axis.N,), (Axis.H,), (Axis.C,)),
            grouping=Grouping.PER_LAYER,
            kv_packing=KVPacking.SPLIT,
            storage_dtype="",
        )
        kwargs.update(overrides)
        return KVLayoutDescriptor(**kwargs)

    # Duplicate axis across dims.
    with pytest.raises(ValueError, match="more than one dim"):
        build(dims=((Axis.B,), (Axis.B,), (Axis.C,)))
    # Missing content axis.
    with pytest.raises(ValueError, match="content axis C"):
        build(dims=((Axis.KV,), (Axis.B,), (Axis.N,), (Axis.H,)))
    # The grouping already carries the axis at a list level.
    with pytest.raises(ValueError, match="list level"):
        build(dims=((Axis.L,), (Axis.KV,), (Axis.B,), (Axis.C,)))
    # SHARED must not carry KV, in dims or at a list level.
    with pytest.raises(ValueError, match="SHARED"):
        build(kv_packing=KVPacking.SHARED)
    with pytest.raises(ValueError, match="SHARED"):
        build(
            kv_packing=KVPacking.SHARED,
            grouping=Grouping.KV_LISTS,
            dims=((Axis.B,), (Axis.N,), (Axis.H,), (Axis.C,)),
        )
    # SPLIT needs a KV axis somewhere; a (K, V) pair list level suffices.
    with pytest.raises(ValueError, match="SPLIT"):
        build(dims=((Axis.B,), (Axis.N,), (Axis.H,), (Axis.C,)))
    build(
        grouping=Grouping.PER_LAYER_KV_PAIRS,
        dims=((Axis.B,), (Axis.N,), (Axis.H,), (Axis.C,)),
    )
    # FUSED requires KV inside the content region (after N and H).
    with pytest.raises(ValueError, match="FUSED"):
        build(
            kv_packing=KVPacking.FUSED,
            dims=((Axis.KV,), (Axis.B,), (Axis.N,), (Axis.H,), (Axis.C,)),
        )
    # Stride override for a dim that does not exist.
    with pytest.raises(ValueError, match="out of range"):
        build(dim_strides={7: 128})
    # Extents must be positive.
    with pytest.raises(ValueError, match="positive"):
        build(extents={Axis.B: 0})


def test_unknown_structure_raises():
    # The LMCache-side 2LTD chunk layout has no EngineKVFormat member.
    desc = KVLayoutDescriptor(
        extents={Axis.KV: 2},
        dims=((Axis.KV,), (Axis.L,), (Axis.B, Axis.N), (Axis.H, Axis.C)),
        grouping=Grouping.SINGLE_TENSOR,
        kv_packing=KVPacking.SPLIT,
        storage_dtype="",
    )
    with pytest.raises(ValueError, match="no EngineKVFormat member"):
        to_engine_kv_format_name(desc)


def test_kv_layout_names_spell_axis_orders():
    # KVLayoutName and Axis are the same letters: every accepted name is an
    # ordering of L, B, H, N, C, and the facts the detector derives from name
    # compares are axis-order facts on the descriptor side.
    assert set(KV_LAYOUT_NAMES) == set(UNIFIED_MEMBER_BY_KV_LAYOUT)
    for name in KV_LAYOUT_NAMES:
        axes = kv_layout_axes(name)
        assert set(axes) == {Axis.L, Axis.B, Axis.H, Axis.N, Axis.C}, name
        desc = ENGINE_KV_FORMAT_DESCRIPTORS[UNIFIED_MEMBER_BY_KV_LAYOUT[name]]
        assert (axes.index(Axis.H) < axes.index(Axis.N)) == desc.is_hnd, name
        # Blocks-first is B outside L: a per-layer view's block step then
        # spans every layer, so exactly these names need a B stride override.
        blocks_first = axes.index(Axis.B) < axes.index(Axis.L)
        assert blocks_first == (name in ("BLHNC", "BLNHC")), name
    # vLLM's spellings of LMCache's legacy names.
    assert kv_layout_axes("LBNHC") == kv_layout_axes("NHD")
    assert kv_layout_axes("LBHNC") == kv_layout_axes("HND")
    # A heads-outermost name is vocabulary too; its order shows why
    # translate_vllm_kv_cache_layout rejects it: H outside B fragments each
    # block's content per head.
    axes = kv_layout_axes("LHBNC")
    assert axes.index(Axis.H) < axes.index(Axis.B)
    for bad in ("LBHN", "LBHNCX", "LBHNN", "NDH", ""):
        with pytest.raises(ValueError, match="ordering"):
            kv_layout_axes(bad)


@pytest.mark.parametrize("kv_layout", ["BLHNC", "BLNHC"])
def test_blocks_first_view_is_unified_entry_plus_block_stride(kv_layout):
    # vLLM's blocks-first layouts classify as the unified-cache members with
    # the block step in stride(0), which resolve_block_stride_and_log_layout
    # hands the kernels as block_stride_elems. In descriptor terms that is the
    # canonical entry plus one B stride override: it resolves to the view's
    # torch strides and projects onto the member detect_format returns.
    nb, nl, nh, bs, cs = 6, 3, 2, 4, 8
    inner = (nh, bs, cs) if kv_layout == "BLHNC" else (bs, nh, cs)
    pool = torch.zeros(nb, nl, *inner)
    views = [pool[:, layer] for layer in range(nl)]
    fmt, _ = detect_format(views, EngineType.VLLM, {"kv_layout": kv_layout})
    assert fmt.name == UNIFIED_MEMBER_BY_KV_LAYOUT[kv_layout]

    canonical = ENGINE_KV_FORMAT_DESCRIPTORS[fmt.name]
    compact = KVLayoutDescriptor(
        extents={Axis.KV: 2, Axis.B: nb, Axis.N: bs, Axis.H: nh, Axis.C: cs // 2},
        dims=canonical.dims,
        grouping=canonical.grouping,
        kv_packing=canonical.kv_packing,
        storage_dtype="float32",
    )
    view = with_block_stride(compact, views[0].stride(0))
    assert view.resolved_strides() == views[0].stride()
    assert dict(view.dim_strides) == {0: nl * nh * bs * cs}
    assert to_engine_kv_format(view) == fmt
    # A layer-compact cache under the same declaration is the tight variant,
    # and 0 means tight as block_stride_elems does on the transfer path.
    assert compact.resolved_strides() == torch.zeros(nb, *inner).stride()
    assert to_engine_kv_format(compact) == fmt
    assert with_block_stride(view, 0) == compact

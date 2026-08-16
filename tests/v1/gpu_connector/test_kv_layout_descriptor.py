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
"""

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector.kv_format import describe_shape, get_spec_class
from lmcache.v1.gpu_connector.kv_format.descriptor import (
    ENGINE_KV_FORMAT_DESCRIPTORS,
    Axis,
    Grouping,
    KVLayoutDescriptor,
    KVPacking,
    from_engine_kv_format,
    to_engine_kv_format,
    to_engine_kv_format_name,
)
import lmcache.lmcache_native as lmcache_native

F = lmcache_native.EngineKVFormat


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
    # SHARED must not materialize KV.
    with pytest.raises(ValueError, match="SHARED"):
        build(kv_packing=KVPacking.SHARED)
    # SPLIT needs a KV axis somewhere.
    with pytest.raises(ValueError, match="SPLIT"):
        build(dims=((Axis.B,), (Axis.N,), (Axis.H,), (Axis.C,)))
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

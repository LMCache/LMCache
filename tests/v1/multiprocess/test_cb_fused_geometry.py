# SPDX-License-Identifier: Apache-2.0
"""CB re-RoPE geometry must follow each format's ``is_fused_packed`` spec fact.

Fused blocks-first K/V packs K and V into one ``2 * head_size`` row per head
(``kv_size == 1``). If ``_cb_group_rope_geometry`` misclassifies such a group
as split K/V it halves ``per_head``, doubles ``n_heads``, and the rope kernel
then rotates the V half of every head. The helper used to decide this from a
hardcoded format list, so any fused format added later silently took the
split path; these tests pin the decision to the spec fact for every
``EngineKVFormat`` so a new format cannot regress it.

The spec fact alone is not sufficient, though: it is recorded once per
registration, and a registration may MIX a fused main K/V group with a
key-only side cache. MiniMax-M3 does exactly that (60 fused ``2 * head_size``
layers plus 57 key-only ``head_size`` lightning-indexer layers, all rank 4 and
all ``kv_size == 1``), so the shared format labelled the index group fused,
``per_head`` became ``2 * head_size``, and it did not divide that group's row --
every CB retrieve raised and the blend degraded on every rank. The mixed-
registration tests below pin the per-group rule. CPU only.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector.kv_format import get_spec_class
from lmcache.v1.multiprocess.modules.blend import _cb_group_rope_geometry
import lmcache.lmcache_native as lmcache_native

F = lmcache_native.EngineKVFormat
NH, HS = 8, 64


def _group(engine_kv_format, tokens_per_block: int = 4) -> SimpleNamespace:
    """Minimal kernel-group stand-in carrying only what the helper reads."""
    return SimpleNamespace(
        tokens_per_block=tokens_per_block,
        slots_per_block=tokens_per_block,
        engine_kv_format=engine_kv_format,
    )


def _group_with_shape(
    engine_kv_format, nh: int, hs: int, tokens_per_block: int = 128
) -> SimpleNamespace:
    """Kernel-group stand-in that also declares its allocated shape.

    ``shape_desc.hs`` is the per-head width vLLM allocated for THIS group --
    ``2 * head_size`` when K/V are packed together, ``head_size`` when the group
    is key-only or split.
    """
    return SimpleNamespace(
        tokens_per_block=tokens_per_block,
        slots_per_block=tokens_per_block,
        engine_kv_format=engine_kv_format,
        shape_desc=SimpleNamespace(nh=nh, hs=hs),
    )


def _all_formats() -> list:
    return [v for v in vars(F).values() if isinstance(v, F)]


@pytest.mark.parametrize("fmt", _all_formats(), ids=lambda f: f.name)
def test_fused_packed_follows_spec_fact(fmt):
    """For every format, the rope geometry agrees with the spec's fused fact:
    fused rows are ``2 * head_size`` wide and the head count is unchanged."""
    spec = get_spec_class(fmt)
    kv_size = 1 if (spec.is_fused_packed or spec.is_mla) else 2
    per_head_expected = 2 * HS if spec.is_fused_packed else HS
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group(fmt), kv_size, NH * per_head_expected, HS, 0
    )
    assert fused == spec.is_fused_packed, fmt.name
    assert (per_head, n_heads, rot_offset) == (per_head_expected, NH, 0), fmt.name


def test_fused_format_keeps_v_half_out_of_the_rope_window():
    """Regression: a fused group must rotate NH heads of width 2*HS (K half
    only), not 2*NH heads of width HS (which re-RoPEs the V half)."""
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group(F.NL_X_NB_NH_BS_CS), 1, NH * 2 * HS, HS, 0
    )
    assert (fused, per_head, n_heads, rot_offset) == (True, 2 * HS, NH, 0)


def test_untagged_group_infers_split_kv():
    """A group with no recorded format (legacy stand-ins) keeps the plain
    split-K/V inference."""
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group(None), 2, NH * HS, HS, 0
    )
    assert (fused, per_head, n_heads, rot_offset) == (False, HS, NH, 0)


# MiniMax-M3 as measured on an 8-way H200 run (head_size 128): kernel group 0 is
# 60 fused layers of one 256-wide row, kernel group 1 is 57 key-only
# lightning-indexer layers of one 128-wide row. Both are rank 4 and both report
# kv_size == 1, so both carry the same fused engine_kv_format.
M3_HEAD_SIZE = 128
M3_FUSED_FORMAT = F.NL_X_NB_BS_NH_CS


def test_mixed_registration_key_only_group_is_not_fused():
    """Regression: the index side cache must not inherit the fused claim.

    Before the per-group rule this raised
    "hidden_dim (128) not a multiple of per-head width (256; fused=True)",
    which the blend server answered as scatter_ran=False -- so every scatter
    degraded TP-collectively and blend_rate was 0.0 with correct output.
    """
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group_with_shape(M3_FUSED_FORMAT, nh=1, hs=M3_HEAD_SIZE),
        1,
        M3_HEAD_SIZE,
        M3_HEAD_SIZE,
        1,
    )
    # One 128-wide K plane, rotated whole: the indexer's keys ARE RoPE'd (the
    # M3 write kernel ropes index-K at the token's absolute position), so this
    # group must be rotated, just not as a fused pair.
    assert (fused, per_head, n_heads, rot_offset) == (False, M3_HEAD_SIZE, 1, 0)


def test_mixed_registration_main_group_stays_fused():
    """The fused main K/V group in the same registration is unaffected."""
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group_with_shape(M3_FUSED_FORMAT, nh=1, hs=2 * M3_HEAD_SIZE),
        1,
        2 * M3_HEAD_SIZE,
        M3_HEAD_SIZE,
        0,
    )
    assert (fused, per_head, n_heads, rot_offset) == (
        True,
        2 * M3_HEAD_SIZE,
        1,
        0,
    )


def test_multi_head_fused_group_with_shape_desc_stays_fused():
    """The ordinary fused case keeps working when a shape is declared."""
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group_with_shape(F.NL_X_NB_NH_BS_CS, nh=NH, hs=2 * HS),
        1,
        NH * 2 * HS,
        HS,
        0,
    )
    assert (fused, per_head, n_heads, rot_offset) == (True, 2 * HS, NH, 0)


def test_declared_shape_never_promotes_a_non_fused_format():
    """The per-group width only ever NARROWS a fused claim, never widens one.

    A split-K/V registration whose row happens to be ``2 * head_size`` wide is
    two heads, not one fused head; the format is authoritative for that.
    """
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group_with_shape(F.NL_X_TWO_NB_NH_BS_HS, nh=2, hs=HS),
        2,
        2 * HS,
        HS,
        0,
    )
    assert (fused, per_head, n_heads, rot_offset) == (False, HS, 2, 0)

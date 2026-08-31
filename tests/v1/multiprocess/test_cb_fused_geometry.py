# SPDX-License-Identifier: Apache-2.0
"""CB re-RoPE geometry must follow each format's ``is_fused_packed`` spec fact.

Fused blocks-first K/V packs K and V into one ``2 * head_size`` row per head
(``kv_size == 1``). If ``_cb_group_rope_geometry`` misclassifies such a group
as split K/V it halves ``per_head``, doubles ``n_heads``, and the rope kernel
then rotates the V half of every head. The helper used to decide this from a
hardcoded format list, so any fused format added later silently took the
split path; these tests pin the decision to the spec fact for every
``EngineKVFormat`` so a new format cannot regress it. CPU only.
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

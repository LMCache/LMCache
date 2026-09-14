# SPDX-License-Identifier: Apache-2.0
"""Golden tests for the ``EngineKVFormat`` classification predicates.

The structural shape (``is_cross_layer`` / ``is_kv_list`` / ``is_layer_list``)
and the ``is_mla`` / ``is_kv_second_tuple`` modifiers are declared once per
format on its :class:`KVFormatSpec` and mirrored for the device kernels in
``csrc/engine_kv_format.h`` (reached via ``device_ops``). This file pins each
format's classification, checks the two sides agree, and enforces that the
three structural flags partition every format (exactly one is true), so a new
format or an edit cannot silently break the contract the per-layer detection
relies on.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format import get_spec_class
import lmcache.lmcache_native as lmcache_native

F = lmcache_native.EngineKVFormat

# (is_cross_layer, is_kv_list, is_layer_list, is_mla, is_kv_second_tuple) per
# format.
EXPECTED = {
    F.NB_NL_TWO_BS_NH_HS: (True, False, False, False, False),
    F.NB_NL_TWO_NH_BS_HS: (True, False, False, False, False),
    F.TWO_X_NL_X_NBBS_NH_HS: (False, True, False, False, False),
    F.TWO_X_NL_X_NB_BS_NH_HS: (False, True, False, False, False),
    F.NL_X_TWO_NB_BS_NH_HS: (False, False, True, False, False),
    F.NL_X_NB_TWO_BS_NH_HS: (False, False, True, False, False),
    F.NL_X_TWO_NB_NH_BS_HS: (False, False, True, False, False),
    F.NL_X_NB_TWO_NH_BS_HS: (False, False, True, False, False),
    F.NL_X_TWO_NB_NH_ONE_BS_HS: (False, False, True, False, False),
    F.NL_X_NB_NH_BS_TWO_HS: (False, False, True, False, False),
    F.NL_X_NB_BS_NH_TWO_HS: (False, False, True, False, False),
    F.NL_X_NB_NH_BS_CS: (False, False, True, False, False),
    F.NL_X_NB_BS_NH_CS: (False, False, True, False, False),
    F.NL_X_NB_BS_HS: (False, False, True, True, False),
    F.NL_X_NBBS_ONE_HS: (False, False, True, True, False),
    F.NL_X_NB_BSV_BSS: (False, False, True, True, False),
    F.NL_X_TWO_X_NB_BS_NH_HS: (False, False, True, False, True),
    F.NB_NL_TWO_NH_BS_HS: (True, False, False, False, False),
    F.TWO_X_NL_X_NBBS_NH_HS: (False, True, False, False, False),
    F.TWO_X_NL_X_NB_BS_NH_HS: (False, True, False, False, False),
    F.NL_X_TWO_NB_BS_NH_HS: (False, False, True, False, False),
    F.NL_X_NB_TWO_BS_NH_HS: (False, False, True, False, False),
    F.NL_X_TWO_NB_NH_BS_HS: (False, False, True, False, False),
    F.NL_X_NB_TWO_NH_BS_HS: (False, False, True, False, False),
    F.NL_X_TWO_NB_NH_ONE_BS_HS: (False, False, True, False, False),
    F.NL_X_NB_NH_BS_TWO_HS: (False, False, True, False, False),
    F.NL_X_NB_BS_NH_TWO_HS: (False, False, True, False, False),
    F.NL_X_NB_NH_BS_CS: (False, False, True, False, False),
    F.NL_X_NB_BS_NH_CS: (False, False, True, False, False),
    F.NL_X_NB_BS_HS: (False, False, True, True, False),
    F.NL_X_NBBS_ONE_HS: (False, False, True, True, False),
    F.NL_X_NB_BSV_BSS: (False, False, True, True, False),
    F.NL_X_TWO_X_NB_BS_HS: (False, False, True, True, True),
}

# Facts that only the spec carries (no native predicate mirrors them):
# (is_hnd, is_fused_packed, is_two_major, is_pbs_fused) per format.
EXPECTED_SPEC_FACTS = {
    F.NB_NL_TWO_BS_NH_HS: (False, False, False, False),
    F.NB_NL_TWO_NH_BS_HS: (True, False, False, False),
    F.TWO_X_NL_X_NBBS_NH_HS: (False, False, False, True),
    F.TWO_X_NL_X_NB_BS_NH_HS: (False, False, False, False),
    F.NL_X_TWO_NB_BS_NH_HS: (False, False, True, False),
    F.NL_X_NB_TWO_BS_NH_HS: (False, False, False, False),
    F.NL_X_TWO_NB_NH_BS_HS: (True, False, True, False),
    F.NL_X_TWO_NB_NH_ONE_BS_HS: (True, False, True, False),
    F.NL_X_NB_TWO_NH_BS_HS: (True, False, False, False),
    F.NL_X_NB_NH_BS_TWO_HS: (True, True, False, False),
    F.NL_X_NB_BS_NH_TWO_HS: (False, True, False, False),
    F.NL_X_NB_NH_BS_CS: (True, True, False, False),
    F.NL_X_NB_BS_NH_CS: (False, True, False, False),
    F.NL_X_NB_BS_HS: (False, False, False, False),
    F.NL_X_NBBS_ONE_HS: (False, False, False, True),
    F.NL_X_NB_BSV_BSS: (False, False, False, False),
    F.NL_X_TWO_X_NB_BS_NH_HS: (False, False, False, False),
    F.NL_X_TWO_X_NB_BS_HS: (False, False, False, False),
}


def _all_formats():
    return [v for v in vars(F).values() if isinstance(v, F)]


def test_classification_matches_golden():
    for fmt, expected in EXPECTED.items():
        got = (
            lmcache_native.is_cross_layer(fmt),
            lmcache_native.is_kv_list(fmt),
            lmcache_native.is_layer_list(fmt),
            lmcache_native.is_mla(fmt),
            lmcache_native.is_kv_second_tuple(fmt),
        )
        assert got == expected, f"{fmt}: got {got}, expected {expected}"


def test_spec_facts_match_predicates():
    # The spec is the Python-side owner of the facts; on a native build the
    # predicates come from csrc, so this pins the two copies together.
    for fmt in _all_formats():
        spec = get_spec_class(fmt)
        got = (
            spec.is_cross_layer,
            spec.is_kv_list,
            spec.is_layer_list,
            spec.is_mla,
            spec.is_kv_second_tuple,
        )
        expected = (
            lmcache_native.is_cross_layer(fmt),
            lmcache_native.is_kv_list(fmt),
            lmcache_native.is_layer_list(fmt),
            lmcache_native.is_mla(fmt),
            lmcache_native.is_kv_second_tuple(fmt),
        )
        assert got == expected, f"{fmt}: spec {got}, native {expected}"


def test_spec_only_facts_match_golden():
    for fmt, expected in EXPECTED_SPEC_FACTS.items():
        spec = get_spec_class(fmt)
        got = (spec.is_hnd, spec.is_fused_packed, spec.is_two_major, spec.is_pbs_fused)
        assert got == expected, f"{fmt}: got {got}, expected {expected}"


def test_every_format_is_pinned():
    # A new EngineKVFormat must be added to EXPECTED (and classified) deliberately.
    assert set(_all_formats()) == set(EXPECTED)
    assert set(_all_formats()) == set(EXPECTED_SPEC_FACTS)


def test_structural_flags_partition_every_format():
    # Exactly one structural shape is true for every format.
    for fmt in _all_formats():
        structural = (
            lmcache_native.is_cross_layer(fmt),
            lmcache_native.is_kv_list(fmt),
            lmcache_native.is_layer_list(fmt),
        )
        assert sum(structural) == 1, f"{fmt}: structural flags {structural}"

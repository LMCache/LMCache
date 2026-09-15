# SPDX-License-Identifier: Apache-2.0
"""Golden tests for the ``EngineKVFormat`` classification facts.

The structural shape (``is_cross_layer`` / ``is_kv_list`` / ``is_layer_list``)
and the ``is_mla`` / ``is_kv_second_tuple`` modifiers are declared once per
format on its :class:`KVFormatSpec`, which is the single owner every Python call
site reads. The device kernels need the same facts at compile time, so
``csrc/engine_kv_format.h`` carries a C++ copy of the table. This file pins each
format's classification, parses the csrc table to check the two sides agree, and
enforces that the three structural flags partition every format (exactly one is
true), so a new format or an edit cannot silently break the contract the
per-layer detection relies on.
"""

# Standard
from pathlib import Path
import re

# Third Party
import pytest

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
}

# Facts that no device kernel branches on, so csrc declares them without a
# predicate: (is_hnd, is_fused_packed, is_two_major, is_pbs_fused) per format.
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
}


# Every fact the two sides declare, in FormatFacts / KVFormatSpec order.
ALL_FACT_NAMES = (
    "is_cross_layer",
    "is_kv_list",
    "is_layer_list",
    "is_mla",
    "is_hnd",
    "is_fused_packed",
    "is_two_major",
    "is_pbs_fused",
    "is_kv_second_tuple",
)

CSRC_HEADER = Path(__file__).resolve().parents[3] / "csrc" / "engine_kv_format.h"

_CASE_RE = re.compile(r"case EngineKVFormat::(\w+):")
_SETTER_RE = re.compile(r"facts\.(\w+) = true;")


def _all_formats():
    return [v for v in vars(F).values() if isinstance(v, F)]


def _csrc_format_facts() -> dict[str, set[str]]:
    """Parse the ``format_facts`` switch in ``csrc/engine_kv_format.h``.

    Returns:
        Format enum name -> the set of fact names csrc sets to true for it.
        A format whose case body sets nothing maps to an empty set.
    """
    switch_body = CSRC_HEADER.read_text().split(
        "constexpr FormatFacts format_facts(", 1
    )[1]
    facts: dict[str, set[str]] = {}
    # One case body may be shared by several fall-through labels.
    labels: list[str] = []
    for raw_line in switch_body.splitlines():
        line = raw_line.strip()
        case_match = _CASE_RE.fullmatch(line)
        if case_match:
            labels.append(case_match.group(1))
            facts.setdefault(case_match.group(1), set())
            continue
        setter_match = _SETTER_RE.fullmatch(line)
        if setter_match:
            for label in labels:
                facts[label].add(setter_match.group(1))
        elif line == "break;":
            labels = []
        elif line.startswith("default:"):
            break
    return facts


def test_classification_matches_golden():
    for fmt, expected in EXPECTED.items():
        spec = get_spec_class(fmt)
        got = (
            spec.is_cross_layer,
            spec.is_kv_list,
            spec.is_layer_list,
            spec.is_mla,
            spec.is_kv_second_tuple,
        )
        assert got == expected, f"{fmt}: got {got}, expected {expected}"


@pytest.mark.skipif(
    not CSRC_HEADER.is_file(), reason="csrc sources absent (installed package)"
)
def test_spec_facts_match_csrc_table():
    # The spec owns the facts for Python; csrc keeps a C++ copy the device
    # kernels compile against. Nothing at runtime crosses the two, so pin them
    # here: a fact set on one side only would silently split the transfer path
    # (Python) from the kernel (csrc).
    csrc_facts = _csrc_format_facts()
    assert csrc_facts.keys() == {fmt.name for fmt in _all_formats()}, (
        "csrc format_facts and the EngineKVFormat enum list different formats"
    )
    drifted = {
        fmt.name: (
            sorted({name for name in ALL_FACT_NAMES if getattr(spec, name)}),
            sorted(csrc_facts[fmt.name]),
        )
        for fmt, spec in ((fmt, get_spec_class(fmt)) for fmt in _all_formats())
        if {name for name in ALL_FACT_NAMES if getattr(spec, name)}
        != csrc_facts[fmt.name]
    }
    assert not drifted, f"spec vs csrc facts drift (spec, csrc): {drifted}"


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
        spec = get_spec_class(fmt)
        structural = (spec.is_cross_layer, spec.is_kv_list, spec.is_layer_list)
        assert sum(structural) == 1, f"{fmt}: structural flags {structural}"


def test_removed_predicates_are_not_rebound():
    # The facts are spec-owned on the Python side; these bindings were dropped
    # from csrc/lmcache_native/pybind.cpp and must not come back.
    removed = (
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
        "is_kv_second_tuple",
    )
    for name in removed:
        assert not hasattr(lmcache_native, name), (
            f"lmcache_native.{name} is back; read the fact from the spec instead"
        )

    # The .pyi stub is read by type checkers, not by Python at import time, so
    # the module-level check above can't catch a declaration left behind
    # there: mypy would keep accepting a call that raises AttributeError at
    # runtime.
    pyi_path = Path(lmcache_native.__file__).with_name("lmcache_native.pyi")
    if pyi_path.is_file():
        pyi_source = pyi_path.read_text()
        for name in removed:
            assert not re.search(rf"^def {name}\(", pyi_source, re.MULTILINE), (
                f"lmcache_native.pyi still declares {name}; remove its stub too"
            )

# SPDX-License-Identifier: Apache-2.0
"""Tests for the multiprocess protocol registry (``RequestType`` + definitions)."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.protocols import initialize_protocols
from lmcache.v1.multiprocess.protocols.base import RequestType

CB_REQUESTS = [
    "CB_REGISTER_ROPE",
    "CB_UNREGISTER_ROPE",
    "CB_RETRIEVE_PRE_COMPUTED",
    "CB_UNIFIED_LOOKUP",
]

DEPRECATED_CB_ALIASES = {
    "CB_REGISTER_ROPE_V3": "CB_REGISTER_ROPE",
    "CB_UNREGISTER_ROPE_V3": "CB_UNREGISTER_ROPE",
    "CB_RETRIEVE_PRE_COMPUTED_V3": "CB_RETRIEVE_PRE_COMPUTED",
}


def test_every_request_type_has_a_definition() -> None:
    definitions = initialize_protocols()
    assert set(definitions) == set(RequestType)


def test_blend_requests_are_registered() -> None:
    definitions = initialize_protocols()
    for name in CB_REQUESTS:
        assert RequestType[name] in definitions


@pytest.mark.parametrize(("alias", "canonical"), sorted(DEPRECATED_CB_ALIASES.items()))
def test_deprecated_cb_aliases_resolve_to_the_canonical_member(
    alias: str, canonical: str
) -> None:
    # Same member => same wire value, so an older blend plugin that still sends
    # the ``_V3`` name dispatches to the same handler.
    assert getattr(RequestType, alias) is RequestType[canonical]
    assert RequestType[alias].name == canonical


def test_deprecated_cb_aliases_are_not_separate_members() -> None:
    # Enum aliases are excluded from iteration, so the registry validation
    # (one definition per member) is unaffected by them.
    names = {member.name for member in RequestType}
    assert not names & set(DEPRECATED_CB_ALIASES)

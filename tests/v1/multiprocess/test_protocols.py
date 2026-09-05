# SPDX-License-Identifier: Apache-2.0
"""Tests for the multiprocess protocol registry (``RequestType`` + definitions)."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.mq import msgspec_decode, msgspec_encode
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


def test_registration_aware_ping_protocol_is_registered() -> None:
    definitions = initialize_protocols()
    request_type = RequestType.PING_REGISTERED

    definition = definitions[request_type]
    assert definition.payload_classes == [int, list[RequestType]]
    assert definition.response_class == list[RequestType]


def test_registration_type_list_round_trips_on_the_wire() -> None:
    registration_types = [
        RequestType.REGISTER_KV_CACHE,
        RequestType.REGISTER_Q_CACHE,
    ]

    encoded = msgspec_encode(registration_types, cls=list[RequestType])

    assert msgspec_decode(encoded, cls=list[RequestType]) == registration_types


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


@pytest.mark.parametrize(
    "name",
    [
        "P2P_LOOKUP_AND_LOCK",
        "P2P_QUERY_LOOKUP_RESULTS",
        "P2P_UNLOCK_OBJECTS",
        "GET_EXPERIMENTAL",
        "PING_REGISTERED",
    ],
)
def test_compatibility_aliases_do_not_reuse_later_auto_values(name: str) -> None:
    assert RequestType[name].name == name

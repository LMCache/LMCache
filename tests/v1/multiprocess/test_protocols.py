# SPDX-License-Identifier: Apache-2.0
"""Tests for the multiprocess protocol registry (``RequestType`` + definitions)."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.protocols import initialize_protocols
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.blend import (
    BLEND_PROTOCOL_VERSION,
    handshake_response,
)

# Frozen wire ids. RequestType values are exchanged between builds, so this
# table is APPEND-ONLY: a new request type adds a row with a never-used value;
# no row is ever renumbered, reused, or removed (deprecated members keep their
# row). If this test fails on an existing row, fix the enum -- not the table.
FROZEN_WIRE_IDS = {
    "REGISTER_KV_CACHE": 1,
    "UNREGISTER_KV_CACHE": 2,
    "REGISTER_Q_CACHE": 3,
    "UNREGISTER_Q_CACHE": 4,
    "STORE_Q": 5,
    "STORE": 6,
    "RETRIEVE": 7,
    "LOOKUP": 8,
    "QUERY_PREFETCH_STATUS": 9,
    "WAIT_PREFETCH_STATUS": 10,
    "QUERY_PREFETCH_LOOKUP_HITS": 11,
    "FREE_LOOKUP_LOCKS": 12,
    "END_SESSION": 13,
    "REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": 14,
    "UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": 15,
    "PREPARE_STORE": 16,
    "COMMIT_STORE": 17,
    "PREPARE_RETRIEVE": 18,
    "COMMIT_RETRIEVE": 19,
    "CLEAR": 20,
    "GET_CHUNK_SIZE": 21,
    "PING": 22,
    "REPORT_BLOCK_ALLOCATION": 23,
    "NOOP": 24,
    "CB_REGISTER_ROPE": 25,
    "CB_UNREGISTER_ROPE": 26,
    "CB_RETRIEVE_PRE_COMPUTED": 27,
    "CB_UNIFIED_LOOKUP": 28,
    "P2P_LOOKUP_AND_LOCK": 29,
    "P2P_QUERY_LOOKUP_RESULTS": 30,
    "P2P_UNLOCK_OBJECTS": 31,
    "GET_EXPERIMENTAL": 32,
    "CB_PROTOCOL_HANDSHAKE": 33,
}

CB_REQUESTS = [
    "CB_REGISTER_ROPE",
    "CB_UNREGISTER_ROPE",
    "CB_RETRIEVE_PRE_COMPUTED",
    "CB_UNIFIED_LOOKUP",
    "CB_PROTOCOL_HANDSHAKE",
]

DEPRECATED_CB_ALIASES = {
    "CB_REGISTER_ROPE_V3": "CB_REGISTER_ROPE",
    "CB_UNREGISTER_ROPE_V3": "CB_UNREGISTER_ROPE",
    "CB_RETRIEVE_PRE_COMPUTED_V3": "CB_RETRIEVE_PRE_COMPUTED",
}


def test_wire_ids_are_frozen() -> None:
    # Every canonical member has exactly its pinned wire value. Failing here
    # means an existing id was renumbered/reused -- that breaks every deployed
    # client/server pair (see #4758, #4897). Fix the enum, never this table.
    actual = {member.name: member.value for member in RequestType}
    assert actual == FROZEN_WIRE_IDS


def test_wire_ids_are_unique() -> None:
    values = list(FROZEN_WIRE_IDS.values())
    assert len(values) == len(set(values))


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


def test_handshake_response() -> None:
    server_version, compatible = handshake_response(BLEND_PROTOCOL_VERSION)
    assert server_version == BLEND_PROTOCOL_VERSION
    assert compatible

    server_version, compatible = handshake_response(BLEND_PROTOCOL_VERSION + 1)
    assert server_version == BLEND_PROTOCOL_VERSION
    assert not compatible


@pytest.mark.parametrize(
    "name",
    [
        "P2P_LOOKUP_AND_LOCK",
        "P2P_QUERY_LOOKUP_RESULTS",
        "P2P_UNLOCK_OBJECTS",
        "GET_EXPERIMENTAL",
    ],
)
def test_compatibility_aliases_do_not_reuse_later_auto_values(name: str) -> None:
    assert RequestType[name].name == name

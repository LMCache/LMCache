# SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral RPC routes and message contracts."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.protocols import base as protocol_base
from lmcache.v1.multiprocess.protocols.blend import (
    BLEND_PROTOCOL_VERSION,
    handshake_response,
)
from lmcache.v1.multiprocess.rpc_messages import RPC_MESSAGE_TYPES
from lmcache.v1.multiprocess.rpc_messages.registry import iter_rpc_message_types

CB_OPERATIONS = [
    "cb_register_rope",
    "cb_unregister_rope",
    "cb_retrieve_pre_computed",
    "cb_unified_lookup",
    "cb_protocol_handshake",
]


def test_legacy_request_type_enum_is_not_part_of_the_protocol() -> None:
    assert not hasattr(protocol_base, "RequestType")


def test_every_rpc_operation_has_a_message_pair() -> None:
    assert set(RPC_MESSAGE_TYPES) == {
        operation for operation, _ in iter_rpc_message_types()
    }
    assert all(
        operation.isidentifier() and operation == operation.lower()
        for operation in RPC_MESSAGE_TYPES
    )


def test_message_pairs_are_declared_in_domain_modules() -> None:
    """Prevent new RPC contracts from returning to one shared definition file."""
    for _, (request_class, response_class) in iter_rpc_message_types():
        for message_class in (request_class, response_class):
            assert message_class.__module__.startswith(
                "lmcache.v1.multiprocess.rpc_messages."
            )
            assert not message_class.__module__.endswith(".__init__")


@pytest.mark.parametrize("operation", CB_OPERATIONS)
def test_blend_operations_have_message_pairs(operation: str) -> None:
    assert operation in RPC_MESSAGE_TYPES


def test_handshake_response() -> None:
    server_version, compatible = handshake_response(BLEND_PROTOCOL_VERSION)
    assert server_version == BLEND_PROTOCOL_VERSION
    assert compatible

    server_version, compatible = handshake_response(BLEND_PROTOCOL_VERSION + 1)
    assert server_version == BLEND_PROTOCOL_VERSION
    assert not compatible

# SPDX-License-Identifier: Apache-2.0
"""Tests for descriptor-derived multiprocess gRPC method metadata."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    get_request_message_class,
    get_response_message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.protocol import (
    RPC,
    RPC_METHODS,
    coerce_rpc_method,
)

CB_METHODS = [
    "CbRegisterRope",
    "CbUnregisterRope",
    "CbRetrievePreComputed",
    "CbUnifiedLookup",
]

DEPRECATED_CB_NAMES = [
    "CB_REGISTER_ROPE_V3",
    "CB_UNREGISTER_ROPE_V3",
    "CB_RETRIEVE_PRE_COMPUTED_V3",
    "CbRegisterRopeV3",
    "CbUnregisterRopeV3",
    "CbRetrievePreComputedV3",
]


def test_every_descriptor_rpc_has_generated_messages() -> None:
    """Every protobuf RPC resolves to generated request/response messages."""
    for rpc_method in RPC_METHODS:
        assert get_request_message_class(rpc_method).DESCRIPTOR.full_name.startswith(
            "lmcache.mp."
        )
        assert get_response_message_class(rpc_method).DESCRIPTOR.full_name.startswith(
            "lmcache.mp."
        )


def test_blend_methods_are_registered_on_blend_service() -> None:
    """CacheBlend exposes current method names without V2/V3 suffixes."""
    for name in CB_METHODS:
        rpc_method = getattr(RPC, name)
        assert coerce_rpc_method(name) is rpc_method
        assert rpc_method.service_name == "BlendService"


@pytest.mark.parametrize("name", DEPRECATED_CB_NAMES)
def test_deprecated_cb_names_are_not_accepted(name: str) -> None:
    """Deprecated CacheBlend V3 aliases are not part of the gRPC protocol."""
    with pytest.raises(ValueError, match="Invalid RPC method"):
        coerce_rpc_method(name)

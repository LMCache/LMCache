# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mp-mode gRPC wire contract.

These tests are IO-free; they cover the two invariants that both the
LMCache mp-mode server and the vLLM connector rely on:

1. Bare ``host:port`` URLs are prefixed with the correct default scheme
   before reaching the transport layer.
2. Every ``RpcMethod`` token maps to a real gRPC rpc method on
   the generated ``MessageQueueServicer``, i.e. the ``.proto`` file
   never drifts away from the Python protocol layer.
"""

# Third Party
import pytest

pytest.importorskip(
    "vllm",
    reason="mp-mode connector imports vLLM at module top",
)

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    _ensure_transport_scheme,
)
from lmcache.v1.multiprocess.mq import (  # noqa: E402
    request_type_to_method_name,
)
from lmcache.v1.multiprocess.protocol import RPC, RpcMethod  # noqa: E402
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (  # noqa: E402
    lmcache_mq_pb2_grpc,
)


def test_ensure_transport_scheme_defaults_to_grpc():
    # Bare host:port must be prefixed with grpc:// because gRPC is now
    # the only supported mp-mode transport.
    assert _ensure_transport_scheme("127.0.0.1:5555") == "grpc://127.0.0.1:5555"
    assert _ensure_transport_scheme("localhost:5555") == "grpc://localhost:5555"


@pytest.mark.parametrize(
    "url",
    [
        "grpc://127.0.0.1:5555",
        "grpc://lmcache-mp.svc:5555",
    ],
)
def test_ensure_transport_scheme_preserves_existing_scheme(url):
    assert _ensure_transport_scheme(url) == url


@pytest.mark.parametrize(
    "request_type,expected",
    [
        (RPC.Store, "Store"),
        (RPC.Retrieve, "Retrieve"),
        (RPC.PrepareStore, "PrepareStore"),
        (RPC.CbLookupPreComputedV2, "CbLookupPreComputedV2"),
        (RPC.P2PLookupAndLock, "P2PLookupAndLock"),
    ],
)
def test_request_type_to_method_name(request_type, expected):
    assert request_type_to_method_name(request_type) == expected


def test_every_request_type_maps_to_a_real_grpc_method():
    # The .proto file must define a rpc method for every RpcMethod,
    # otherwise the servicer dispatch layer would 404 that rpc.
    servicer_methods = {
        m
        for m in dir(lmcache_mq_pb2_grpc.MessageQueueServicer)
        if not m.startswith("_")
    }
    missing = [
        (rt.name, request_type_to_method_name(rt))
        for rt in RpcMethod
        if request_type_to_method_name(rt) not in servicer_methods
    ]
    assert not missing, (
        f"RpcMethod members without a matching proto rpc method: {missing}"
    )

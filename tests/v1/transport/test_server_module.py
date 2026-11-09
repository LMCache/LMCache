# SPDX-License-Identifier: Apache-2.0
"""Wire contract of the server-module call envelopes."""

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.protocols import server_module as documented_path
from lmcache.v1.transport.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)

_REQUEST = ServerModuleCallRequest(method="my_package.echo", payload=b"hi")
_REQUEST_BYTES = bytes.fromhex(
    "82a66d6574686f64af6d795f7061636b6167652e6563686fa77061796c6f6164c4026869"
)
_RESPONSE = ServerModuleCallResponse(success=False, payload=b"", error="boom")
_RESPONSE_BYTES = bytes.fromhex(
    "83a773756363657373c2a77061796c6f6164c400a56572726f72a4626f6f6d"
)


def test_documented_import_path_exports_the_same_classes() -> None:
    assert documented_path.ServerModuleCallRequest is ServerModuleCallRequest
    assert documented_path.ServerModuleCallResponse is ServerModuleCallResponse


def test_msgpack_bytes() -> None:
    assert msgspec.msgpack.encode(_REQUEST) == _REQUEST_BYTES
    assert msgspec.msgpack.encode(_RESPONSE) == _RESPONSE_BYTES
    request = msgspec.msgpack.decode(_REQUEST_BYTES, type=ServerModuleCallRequest)
    response = msgspec.msgpack.decode(_RESPONSE_BYTES, type=ServerModuleCallResponse)
    assert (request, response) == (_REQUEST, _RESPONSE)

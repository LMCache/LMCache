# SPDX-License-Identifier: Apache-2.0
"""Compatibility for shared request errors and gRPC authentication."""

# Standard
import pickle  # noqa: TID251 -- trusted fixtures verify legacy error serialization

# Third Party
import pytest

# First Party
from lmcache.v1.client.transport import errors as client_errors
from lmcache.v1.client.transport.grpc_impl.auth import (
    GrpcClientAuthenticationInterceptor as ClientAuthenticationInterceptor,
)
from lmcache.v1.transport import errors
from lmcache.v1.transport.grpc.auth import GrpcClientAuthenticationInterceptor


@pytest.mark.parametrize("name", ["MPRequestError", "MPServerUnavailableError"])
def test_request_error_compatibility(name: str) -> None:
    """Old imports catch shared errors and old serialized errors still load."""
    error_type = getattr(errors, name)
    assert error_type is getattr(client_errors, name)
    error = error_type("INTERNAL", "peer failed")
    with pytest.raises(client_errors.REQUEST_FAILURES):
        raise error
    legacy_pickle = (
        f"clmcache.v1.client.transport.errors\n{name}\n(VINTERNAL\nVpeer failed\ntR."
    ).encode()
    for restored in (pickle.loads(pickle.dumps(error)), pickle.loads(legacy_pickle)):
        assert type(restored) is error_type
        assert restored.code == error.code
        assert restored.details == error.details
        assert str(restored) == str(error)


def test_authentication_interceptor_compatibility() -> None:
    """The old client import exposes the shared interceptor class."""
    assert ClientAuthenticationInterceptor is GrpcClientAuthenticationInterceptor

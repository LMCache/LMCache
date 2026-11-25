# SPDX-License-Identifier: Apache-2.0
"""Bearer authentication interceptor for the gRPC request client."""

# Standard
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# Third Party
import grpc

# First Party
from lmcache.security.bearer import bearer_authorization_header
from lmcache.v1.transport.grpc.endpoint import AUTHORIZATION_METADATA_KEY


@dataclass(frozen=True)
class _AuthenticatedClientCallDetails(grpc.ClientCallDetails):
    method: str
    timeout: float | None
    metadata: Sequence[tuple[str, str | bytes]] | None
    credentials: grpc.CallCredentials | None
    wait_for_ready: bool | None
    compression: grpc.Compression | None


class GrpcClientAuthenticationInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
):
    """Inject one Bearer credential into unary and streaming RPC metadata."""

    def __init__(self, api_key: str) -> None:
        self._authorization = bearer_authorization_header(api_key)

    def _authenticated_details(
        self, details: grpc.ClientCallDetails
    ) -> _AuthenticatedClientCallDetails:
        metadata = [
            item
            for item in (details.metadata or ())
            if item[0].lower() != AUTHORIZATION_METADATA_KEY
        ]
        metadata.append((AUTHORIZATION_METADATA_KEY, self._authorization))
        return _AuthenticatedClientCallDetails(
            method=details.method,
            timeout=details.timeout,
            metadata=metadata,
            credentials=details.credentials,
            wait_for_ready=details.wait_for_ready,
            compression=details.compression,
        )

    def intercept_unary_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        """Inject authentication into one unary request."""
        return continuation(self._authenticated_details(client_call_details), request)

    def intercept_unary_stream(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        """Inject authentication into one server-streaming request."""
        return continuation(self._authenticated_details(client_call_details), request)

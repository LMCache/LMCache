# SPDX-License-Identifier: Apache-2.0
"""Bearer authentication interceptors for the gRPC request transport."""

# Standard
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

# Third Party
import grpc

# First Party
from lmcache.security.api_key import APIKeyVerifier
from lmcache.security.http import bearer_authorization_header, bearer_token

_AUTHORIZATION_METADATA_KEY = "authorization"
_AUTHENTICATION_ERROR = "Invalid or missing API key"


def _authorization_metadata(
    metadata: Iterable[tuple[str, str | bytes]] | None,
) -> str | None:
    """Return the authorization metadata value, if it is textual."""
    for key, value in metadata or ():
        if key.lower() == _AUTHORIZATION_METADATA_KEY and isinstance(value, str):
            return value
    return None


def _abort_unary(request: Any, context: grpc.ServicerContext) -> Any:
    """Reject an unauthenticated unary request."""
    del request
    context.abort(grpc.StatusCode.UNAUTHENTICATED, _AUTHENTICATION_ERROR)
    raise RuntimeError("gRPC context abort unexpectedly returned")


def _abort_unary_stream(request: Any, context: grpc.ServicerContext) -> Iterator[Any]:
    """Reject an unauthenticated server-streaming request."""
    del request
    context.abort(grpc.StatusCode.UNAUTHENTICATED, _AUTHENTICATION_ERROR)
    return
    yield  # pragma: no cover - marks this function as an iterator


def _abort_stream_unary(
    request_iterator: Iterator[Any], context: grpc.ServicerContext
) -> Any:
    """Reject an unauthenticated client-streaming request."""
    del request_iterator
    context.abort(grpc.StatusCode.UNAUTHENTICATED, _AUTHENTICATION_ERROR)
    raise RuntimeError("gRPC context abort unexpectedly returned")


def _abort_stream_stream(
    request_iterator: Iterator[Any], context: grpc.ServicerContext
) -> Iterator[Any]:
    """Reject an unauthenticated bidirectional-streaming request."""
    del request_iterator
    context.abort(grpc.StatusCode.UNAUTHENTICATED, _AUTHENTICATION_ERROR)
    return
    yield  # pragma: no cover - marks this function as an iterator


class GrpcAuthenticationInterceptor(grpc.ServerInterceptor):
    """Reject invalid metadata before protobuf request deserialization."""

    def __init__(self, verifier: APIKeyVerifier) -> None:
        self._verifier = verifier

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler | None],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        """Authenticate one RPC and preserve its streaming cardinality."""
        handler = continuation(handler_call_details)
        if handler is None or not self._verifier.enabled:
            return handler
        authorization = _authorization_metadata(
            handler_call_details.invocation_metadata
        )
        if self._verifier.verify_token(bearer_token(authorization)):
            return handler

        # Deliberately omit the request deserializer. gRPC therefore invokes
        # the rejecting handler with raw bytes instead of parsing protobuf.
        if handler.request_streaming and handler.response_streaming:
            return grpc.stream_stream_rpc_method_handler(_abort_stream_stream)
        if handler.request_streaming:
            return grpc.stream_unary_rpc_method_handler(_abort_stream_unary)
        if handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(_abort_unary_stream)
        return grpc.unary_unary_rpc_method_handler(_abort_unary)


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
            if item[0].lower() != _AUTHORIZATION_METADATA_KEY
        ]
        metadata.append((_AUTHORIZATION_METADATA_KEY, self._authorization))
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

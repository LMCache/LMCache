# SPDX-License-Identifier: Apache-2.0
"""Method-oriented client built directly from generated gRPC descriptors."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse
import uuid

# Third Party
import grpc

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
    message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    decode_response_to_python,
    encode_request_from_call,
)

_GRPC_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)
_CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"


def parse_grpc_target(server_url: str) -> str:
    """Convert an LMCache multiprocess endpoint into a gRPC target.

    Args:
        server_url: gRPC endpoint.

    Returns:
        A target accepted by ``grpc.insecure_channel``.

    Raises:
        ValueError: If the URL scheme or target is invalid.
    """
    if "://" not in server_url:
        if not server_url:
            raise ValueError("gRPC server target must not be empty")
        return server_url

    parsed = urlparse(server_url)
    if parsed.scheme == "grpc":
        if not parsed.netloc:
            raise ValueError(f"Missing host in gRPC URL: {server_url!r}")
        return parsed.netloc
    if parsed.scheme == "grpc+unix":
        path = f"/{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        if not path:
            raise ValueError(f"Missing socket path in gRPC URL: {server_url!r}")
        return f"unix://{path}"
    raise ValueError(f"Unsupported gRPC URL scheme: {parsed.scheme!r}")


@dataclass(frozen=True)
class _ClientRpc:
    stub_method: Any
    request_class: type[Any]


ClientRpcCallable = Callable[..., MessagingFuture[Any]]


class GrpcMultiprocessClient(RequestClient):
    """Expose every generated unary RPC as a snake-case client method."""

    def __init__(self, server_url: str) -> None:
        self._channel = grpc.insecure_channel(
            parse_grpc_target(server_url), options=_GRPC_OPTIONS
        )
        stubs: dict[str, Any] = {}
        self._rpc_methods: dict[str, _ClientRpc] = {}
        for binding, method in iter_methods():
            service_name = binding.descriptor.name
            stub = stubs.get(service_name)
            if stub is None:
                stub_class = getattr(binding.grpc_module, f"{service_name}Stub")
                stub = stub_class(self._channel)
                stubs[service_name] = stub
            name = client_method_name(method.name)
            if name in self._rpc_methods:
                raise RuntimeError(f"Duplicate gRPC client method: {name}")
            self._rpc_methods[name] = _ClientRpc(
                stub_method=getattr(stub, method.name),
                request_class=message_class(method.input_type),
            )
        self._metadata = ((_CLIENT_ID_METADATA_KEY, uuid.uuid4().bytes),)

    def __getattr__(self, name: str) -> ClientRpcCallable:
        """Resolve a generated RPC as a method-oriented client call."""
        rpc = self._rpc_methods.get(name)
        if rpc is None:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute {name!r}"
            )

        def invoke(*args: Any, **kwargs: Any) -> MessagingFuture[Any]:
            return self._call(rpc, args, kwargs)

        return invoke

    def __dir__(self) -> list[str]:
        """Include descriptor-derived RPC methods in introspection output."""
        return sorted(set(super().__dir__()) | set(self._rpc_methods))

    def cb_register_rope_v3(self, *args: Any, **kwargs: Any) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``CbRegisterRope``."""
        return self.cb_register_rope(*args, **kwargs)

    def cb_unregister_rope_v3(self, *args: Any, **kwargs: Any) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``CbUnregisterRope``."""
        return self.cb_unregister_rope(*args, **kwargs)

    def cb_retrieve_pre_computed_v3(
        self, *args: Any, **kwargs: Any
    ) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``CbRetrievePreComputed``."""
        return self.cb_retrieve_pre_computed(*args, **kwargs)

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._channel.close()

    def _call(
        self,
        rpc: _ClientRpc,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> MessagingFuture[Any]:
        request = encode_request_from_call(rpc.request_class, args, kwargs)
        future: MessagingFuture[Any] = MessagingFuture()
        call = rpc.stub_method.future(
            request,
            metadata=self._metadata,
            wait_for_ready=True,
        )

        def on_done(grpc_future: grpc.Future[Any]) -> None:
            try:
                result = decode_response_to_python(grpc_future.result())
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        call.add_done_callback(on_done)
        return future


def _make_client_rpc_method(name: str) -> ClientRpcCallable:
    def rpc_method(
        self: GrpcMultiprocessClient,
        *args: Any,
        **kwargs: Any,
    ) -> MessagingFuture[Any]:
        return self._call(self._rpc_methods[name], args, kwargs)

    rpc_method.__name__ = name
    rpc_method.__qualname__ = f"GrpcMultiprocessClient.{name}"
    return rpc_method


def _install_client_rpc_methods() -> None:
    """Install descriptor-derived RPC methods on the concrete client class."""
    for _, method in iter_methods():
        name = client_method_name(method.name)
        if name in GrpcMultiprocessClient.__dict__:
            raise RuntimeError(f"gRPC client method conflicts with {name!r}")
        setattr(GrpcMultiprocessClient, name, _make_client_rpc_method(name))


_install_client_rpc_methods()

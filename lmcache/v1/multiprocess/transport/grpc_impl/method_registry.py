# SPDX-License-Identifier: Apache-2.0
"""Compile and register codecs for every generated gRPC method."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable

# First Party
from lmcache.v1.multiprocess.rpc import RpcOperation, get_rpc_spec, get_rpc_specs
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
    message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    RequestDecoder,
    RequestEncoder,
    ResponseDecoder,
    ResponseEncoder,
    compile_request_codec_for_types,
    compile_request_decoder,
    compile_response_decoder_for_type,
    compile_response_encoder,
    compile_response_encoder_for_type,
)


def _normalize_none_type(value: Any) -> Any:
    return None if value is type(None) else value


@dataclass(frozen=True)
class GrpcMethodCodec:
    """Compiled protobuf converters for one generated gRPC method."""

    full_name: str
    operation: RpcOperation
    request_message_class: type[Any]
    response_message_class: type[Any]
    payload_types: tuple[Any, ...]
    response_type: Any
    request_encoder: RequestEncoder
    request_decoder: RequestDecoder
    response_encoder: ResponseEncoder
    response_decoder: ResponseDecoder

    def validate_handler(self, handler: Callable[..., Any]) -> None:
        """Validate that a service handler implements the gRPC contract.

        Args:
            handler: Bound service implementation method.

        Raises:
            TypeError: If request or response annotations differ from the
                annotated gRPC service contract.
        """
        _decoder, handler_payload_types = compile_request_decoder(
            self.request_message_class, handler
        )
        _encoder, handler_response_type = compile_response_encoder(
            self.response_message_class, handler
        )
        if handler_payload_types != self.payload_types:
            raise TypeError(
                f"{self.full_name} handler payload annotations "
                f"{handler_payload_types!r} do not match gRPC contract types "
                f"{self.payload_types!r}"
            )
        if _normalize_none_type(handler_response_type) != _normalize_none_type(
            self.response_type
        ):
            raise TypeError(
                f"{self.full_name} handler return annotation "
                f"{handler_response_type!r} does not match gRPC contract type "
                f"{self.response_type!r}"
            )


@dataclass(frozen=True)
class GrpcMethodCodecRegistry:
    """Read-only lookup table for all generated gRPC method codecs."""

    by_full_name: Mapping[str, GrpcMethodCodec]


@lru_cache(maxsize=1)
def get_method_codec_registry() -> GrpcMethodCodecRegistry:
    """Build and validate codecs for all generated gRPC methods.

    Returns:
        Read-only codec lookup table keyed by full protobuf method name.

    Raises:
        RuntimeError: If a generated method has no matching RPC contract, or
            a protobuf method or operation is duplicated.
        TypeError: If a protobuf message cannot represent its annotated types.
    """
    by_full_name: dict[str, GrpcMethodCodec] = {}
    operations: set[RpcOperation] = set()
    for _binding, method in iter_methods():
        operation = client_method_name(method.name)
        try:
            rpc_spec = get_rpc_spec(operation)
        except KeyError as exc:
            raise RuntimeError(
                f"Generated gRPC method {method.full_name} has no matching "
                f"RequestClient method {operation!r}"
            ) from exc
        if operation in operations:
            raise RuntimeError(f"Duplicate generated gRPC operation: {operation}")

        request_message_class = message_class(method.input_type)
        response_message_class = message_class(method.output_type)
        payload_types = rpc_spec.payload_types
        request_encoder, request_decoder = compile_request_codec_for_types(
            request_message_class, payload_types
        )
        response_type = rpc_spec.response_type
        response_encoder = compile_response_encoder_for_type(
            response_message_class, response_type
        )
        response_decoder = compile_response_decoder_for_type(
            response_message_class, response_type
        )
        codec = GrpcMethodCodec(
            full_name=method.full_name,
            operation=operation,
            request_message_class=request_message_class,
            response_message_class=response_message_class,
            payload_types=payload_types,
            response_type=response_type,
            request_encoder=request_encoder,
            request_decoder=request_decoder,
            response_encoder=response_encoder,
            response_decoder=response_decoder,
        )
        if method.full_name in by_full_name:
            raise RuntimeError(f"Duplicate generated gRPC method: {method.full_name}")
        by_full_name[method.full_name] = codec
        operations.add(operation)

    missing_methods = set(get_rpc_specs()) - operations
    if missing_methods:
        raise RuntimeError(
            "RequestClient RPCs have no generated gRPC method: "
            f"{sorted(missing_methods)}"
        )

    return GrpcMethodCodecRegistry(
        by_full_name=MappingProxyType(by_full_name),
    )

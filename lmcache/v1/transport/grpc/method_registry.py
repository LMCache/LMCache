# SPDX-License-Identifier: Apache-2.0
"""Compile and register codecs for every generated gRPC method."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

# First Party
from lmcache.v1.transport.grpc.descriptors import (
    client_method_name,
    iter_methods,
    message_class,
)
from lmcache.v1.transport.grpc.proto_codec import (
    RequestDecoder,
    RequestEncoder,
    ResponseDecoder,
    ResponseEncoder,
    compile_request_codec_for_types,
    compile_response_decoder_for_type,
    compile_response_encoder_for_type,
)
from lmcache.v1.transport.rpc import RpcOperation, get_rpc_spec, get_rpc_specs


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


@dataclass(frozen=True)
class GrpcMethodCodecRegistry:
    """Read-only lookup table for all generated gRPC method codecs."""

    by_full_name: Mapping[str, GrpcMethodCodec]


@lru_cache(maxsize=None)
def get_method_codec_registry(
    rpc_client: type | None = None,
) -> GrpcMethodCodecRegistry:
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
    for _binding, method in iter_methods(rpc_client):
        operation = client_method_name(method.name)
        try:
            rpc_spec = get_rpc_spec(operation, rpc_client)
        except KeyError as exc:
            raise RuntimeError(
                f"Generated gRPC method {method.full_name} has no matching "
                f"RequestClient method {operation!r}"
            ) from exc
        if operation in operations:
            raise RuntimeError(f"Duplicate generated gRPC operation: {operation}")
        if method.client_streaming or method.server_streaming != rpc_spec.streaming:
            raise TypeError(f"Streaming contract mismatch for {method.full_name}")

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

    missing_methods = set(get_rpc_specs(rpc_client)) - operations
    if missing_methods:
        raise RuntimeError(
            "RequestClient RPCs have no generated gRPC method: "
            f"{sorted(missing_methods)}"
        )

    return GrpcMethodCodecRegistry(
        by_full_name=MappingProxyType(by_full_name),
    )

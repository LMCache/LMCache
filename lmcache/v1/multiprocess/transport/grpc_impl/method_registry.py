# SPDX-License-Identifier: Apache-2.0
"""Compile and register codecs for every generated gRPC method."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable

# First Party
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
)
from lmcache.v1.multiprocess.transport.grpc_impl.services import (
    get_service_implementation_class,
)


def _normalize_none_type(value: Any) -> Any:
    return None if value is type(None) else value


@dataclass(frozen=True)
class GrpcMethodCodec:
    """Compiled protobuf converters for one generated gRPC method."""

    client_name: str
    full_name: str
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
    """Read-only lookup tables for all generated gRPC method codecs."""

    by_client_name: Mapping[str, GrpcMethodCodec]
    by_full_name: Mapping[str, GrpcMethodCodec]


@lru_cache(maxsize=1)
def get_method_codec_registry() -> GrpcMethodCodecRegistry:
    """Build and validate codecs for all generated gRPC methods.

    Returns:
        Read-only codec lookup tables keyed by client and protobuf names.

    Raises:
        RuntimeError: If a generated method has no service implementation or a
            duplicate method name is discovered.
        TypeError: If a protobuf message cannot represent its annotated types.
    """
    by_client_name: dict[str, GrpcMethodCodec] = {}
    by_full_name: dict[str, GrpcMethodCodec] = {}
    for binding, method in iter_methods():
        client_name = client_method_name(method.name)
        request_message_class = message_class(method.input_type)
        response_message_class = message_class(method.output_type)
        implementation_class = get_service_implementation_class(binding.descriptor.name)
        contract_method = getattr(implementation_class, method.name, None)
        if not callable(contract_method):
            raise RuntimeError(
                f"{implementation_class.__name__} has no gRPC method {method.full_name}"
            )
        _request_decoder, payload_types = compile_request_decoder(
            request_message_class, contract_method
        )
        request_encoder, request_decoder = compile_request_codec_for_types(
            request_message_class, payload_types
        )
        response_encoder, response_type = compile_response_encoder(
            response_message_class, contract_method
        )
        response_decoder = compile_response_decoder_for_type(
            response_message_class, response_type
        )
        codec = GrpcMethodCodec(
            client_name=client_name,
            full_name=method.full_name,
            request_message_class=request_message_class,
            response_message_class=response_message_class,
            payload_types=payload_types,
            response_type=response_type,
            request_encoder=request_encoder,
            request_decoder=request_decoder,
            response_encoder=response_encoder,
            response_decoder=response_decoder,
        )
        if client_name in by_client_name:
            raise RuntimeError(f"Duplicate gRPC client method: {client_name}")
        if method.full_name in by_full_name:
            raise RuntimeError(f"Duplicate generated gRPC method: {method.full_name}")
        by_client_name[client_name] = codec
        by_full_name[method.full_name] = codec

    return GrpcMethodCodecRegistry(
        by_client_name=MappingProxyType(by_client_name),
        by_full_name=MappingProxyType(by_full_name),
    )


__all__ = [
    "GrpcMethodCodec",
    "GrpcMethodCodecRegistry",
    "get_method_codec_registry",
]

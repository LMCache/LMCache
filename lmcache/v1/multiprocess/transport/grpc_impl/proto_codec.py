# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for shared protobuf structural codecs."""

# Standard
from typing import Any, Callable

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.codecs import (
    get_message_codec_registry,
)
from lmcache.v1.transport.grpc import proto_codec as _shared
from lmcache.v1.transport.grpc.proto_codec import FieldReader as FieldReader
from lmcache.v1.transport.grpc.proto_codec import FieldWriter as FieldWriter
from lmcache.v1.transport.grpc.proto_codec import RequestDecoder as RequestDecoder
from lmcache.v1.transport.grpc.proto_codec import RequestEncoder as RequestEncoder
from lmcache.v1.transport.grpc.proto_codec import ResponseDecoder as ResponseDecoder
from lmcache.v1.transport.grpc.proto_codec import ResponseEncoder as ResponseEncoder
from lmcache.v1.transport.grpc.proto_codec import ValueDecoder as ValueDecoder
from lmcache.v1.transport.grpc.proto_codec import ValueEncoder as ValueEncoder


def _compile_message_codec(
    descriptor: Any, py_type: Any
) -> tuple[FieldWriter, FieldReader]:
    return _shared._compile_message_codec(
        descriptor, py_type, get_message_codec_registry()
    )


def compile_request_codec_for_types(
    message_cls: Any,
    payload_types: tuple[Any, ...],
) -> tuple[RequestEncoder, RequestDecoder]:
    return _shared.compile_request_codec_for_types(
        message_cls, payload_types, get_message_codec_registry()
    )


def compile_request_decoder(
    message_cls: Any, handler: Callable[..., Any]
) -> tuple[RequestDecoder, tuple[Any, ...]]:
    return _shared.compile_request_decoder(
        message_cls, handler, get_message_codec_registry()
    )


def compile_response_encoder(
    message_cls: Any,
    handler: Callable[..., Any],
) -> tuple[ResponseEncoder, Any]:
    return _shared.compile_response_encoder(
        message_cls, handler, get_message_codec_registry()
    )


def compile_response_encoder_for_type(
    message_cls: Any,
    response_type: Any,
) -> ResponseEncoder:
    return _shared.compile_response_encoder_for_type(
        message_cls, response_type, get_message_codec_registry()
    )


def compile_response_decoder_for_type(
    message_cls: Any, response_type: Any
) -> ResponseDecoder:
    return _shared.compile_response_decoder_for_type(
        message_cls, response_type, get_message_codec_registry()
    )


def decode_response_to_type(response: Any, response_type: Any) -> Any:
    return _shared.decode_response_to_type(
        response, response_type, get_message_codec_registry()
    )


__all__ = [
    "FieldReader",
    "FieldWriter",
    "RequestDecoder",
    "RequestEncoder",
    "ResponseDecoder",
    "ResponseEncoder",
    "ValueDecoder",
    "ValueEncoder",
    "compile_request_codec_for_types",
    "compile_request_decoder",
    "compile_response_decoder_for_type",
    "compile_response_encoder",
    "compile_response_encoder_for_type",
    "decode_response_to_type",
]

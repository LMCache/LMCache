# SPDX-License-Identifier: Apache-2.0
"""Compile and register adapters for every generated gRPC method."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, get_type_hints
import inspect

# First Party
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_request_message_class,
    get_response_message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
    message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.message_conversion import (
    MessageToProto,
    ProtoToMessage,
    build_message_conversion,
)


@dataclass(frozen=True)
class GrpcMethodBinding:
    """Compiled protobuf converters for one generated gRPC method."""

    full_name: str
    request_type: RequestType
    request_message_class: type[Any]
    response_message_class: type[Any]
    python_request_class: type[Any]
    python_response_class: type[Any]
    request_to_proto: MessageToProto
    proto_to_request: ProtoToMessage
    response_to_proto: MessageToProto
    proto_to_response: ProtoToMessage

    def validate_handler(self, handler: Callable[..., Any]) -> None:
        """Validate that a service handler implements the gRPC contract.

        Args:
            handler: Bound service implementation method.

        Raises:
            TypeError: If request or response annotations differ from the
                annotated gRPC service contract.
        """
        signature = inspect.signature(handler)
        hints = get_type_hints(handler)
        parameters = tuple(
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if parameters and parameters[0].name in ("self", "cls"):
            parameters = parameters[1:]
        if len(parameters) != 1 or hints.get(parameters[0].name) is not (
            self.python_request_class
        ):
            raise TypeError(
                f"{self.full_name} handler must accept exactly one "
                f"{self.python_request_class.__name__}"
            )
        if hints.get("return") is not self.python_response_class:
            raise TypeError(
                f"{self.full_name} handler return annotation "
                f"must be {self.python_response_class.__name__}"
            )


@dataclass(frozen=True)
class GrpcMethodRegistry:
    """Read-only lookup table for all generated gRPC method adapters."""

    by_full_name: Mapping[str, GrpcMethodBinding]


@lru_cache(maxsize=1)
def get_method_registry() -> GrpcMethodRegistry:
    """Build and validate adapters for all generated gRPC methods.

    Returns:
        Read-only adapter lookup table keyed by full protobuf method name.

    Raises:
        RuntimeError: If a generated method has no matching request type, or
            a protobuf method or request type is duplicated.
        TypeError: If a protobuf message cannot represent its annotated types.
    """
    by_full_name: dict[str, GrpcMethodBinding] = {}
    request_types: set[RequestType] = set()
    for _binding, method in iter_methods():
        request_name = client_method_name(method.name).upper()
        try:
            request_type = RequestType[request_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Generated gRPC method {method.full_name} has no matching "
                f"RequestType.{request_name}"
            ) from exc
        if request_type in request_types:
            raise RuntimeError(
                f"Duplicate generated gRPC request type: {request_type.name}"
            )

        request_message_class = message_class(method.input_type)
        response_message_class = message_class(method.output_type)
        python_request_class = get_request_message_class(request_type)
        python_response_class = get_response_message_class(request_type)
        request_to_proto, proto_to_request = build_message_conversion(
            request_message_class,
            python_request_class,
        )
        response_to_proto, proto_to_response = build_message_conversion(
            response_message_class, python_response_class
        )
        adapter = GrpcMethodBinding(
            full_name=method.full_name,
            request_type=request_type,
            request_message_class=request_message_class,
            response_message_class=response_message_class,
            python_request_class=python_request_class,
            python_response_class=python_response_class,
            request_to_proto=request_to_proto,
            proto_to_request=proto_to_request,
            response_to_proto=response_to_proto,
            proto_to_response=proto_to_response,
        )
        if method.full_name in by_full_name:
            raise RuntimeError(f"Duplicate generated gRPC method: {method.full_name}")
        by_full_name[method.full_name] = adapter
        request_types.add(request_type)

    return GrpcMethodRegistry(
        by_full_name=MappingProxyType(by_full_name),
    )


__all__ = [
    "GrpcMethodBinding",
    "GrpcMethodRegistry",
    "get_method_registry",
]

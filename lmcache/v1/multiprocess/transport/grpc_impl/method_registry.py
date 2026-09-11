# SPDX-License-Identifier: Apache-2.0
"""Bind generated gRPC method names to transport-neutral Python messages."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, get_type_hints
import inspect

# First Party
from lmcache.v1.multiprocess.protocol import (
    RpcOperation,
    get_request_message_class,
    get_response_message_class,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    client_method_name,
    iter_methods,
)


@dataclass(frozen=True)
class GrpcMethodBinding:
    """Python request/response contract for one generated gRPC method."""

    full_name: str
    method_path: str
    operation: RpcOperation
    python_request_class: type[Any]
    python_response_class: type[Any]

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
    """Read-only lookup table for all generated gRPC method contracts."""

    by_full_name: Mapping[str, GrpcMethodBinding]


@lru_cache(maxsize=1)
def get_method_registry() -> GrpcMethodRegistry:
    """Build and validate Python contracts for all generated gRPC methods.

    Returns:
        Read-only contract lookup table keyed by full protobuf method name.

    Raises:
        RuntimeError: If a generated method has no locally declared Python
            message contract, or a protobuf method or operation is duplicated.
    """
    by_full_name: dict[str, GrpcMethodBinding] = {}
    operations: set[RpcOperation] = set()
    for _binding, method in iter_methods():
        operation = client_method_name(method.name)
        try:
            python_request_class = get_request_message_class(operation)
            python_response_class = get_response_message_class(operation)
        except ValueError as exc:
            raise RuntimeError(
                f"Generated gRPC method {method.full_name} has no matching Python "
                f"RPC contract for {operation!r}"
            ) from exc
        if operation in operations:
            raise RuntimeError(f"Duplicate generated gRPC operation: {operation}")
        adapter = GrpcMethodBinding(
            full_name=method.full_name,
            method_path=(f"/{method.containing_service.full_name}/{method.name}"),
            operation=operation,
            python_request_class=python_request_class,
            python_response_class=python_response_class,
        )
        if method.full_name in by_full_name:
            raise RuntimeError(f"Duplicate generated gRPC method: {method.full_name}")
        by_full_name[method.full_name] = adapter
        operations.add(operation)

    return GrpcMethodRegistry(
        by_full_name=MappingProxyType(by_full_name),
    )


__all__ = [
    "GrpcMethodBinding",
    "GrpcMethodRegistry",
    "get_method_registry",
]

# SPDX-License-Identifier: Apache-2.0
"""Scheduling metadata for concrete gRPC service methods."""

# Standard
from enum import Enum, auto
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

_HANDLER_TYPE_ATTR = "__lmcache_grpc_handler_type__"
_REQUIRES_AFFINITY_ATTR = "__lmcache_grpc_requires_affinity__"


class GrpcHandlerType(Enum):
    """Select how a concrete gRPC service method is executed."""

    SYNC = auto()
    BLOCKING = auto()


def grpc_method(
    handler_type: GrpcHandlerType = GrpcHandlerType.SYNC,
    *,
    requires_client_affinity: bool = False,
) -> Callable[[F], F]:
    """Attach server scheduling metadata to a service implementation method."""

    def decorate(func: F) -> F:
        setattr(func, _HANDLER_TYPE_ATTR, handler_type)
        setattr(func, _REQUIRES_AFFINITY_ATTR, requires_client_affinity)
        return func

    return decorate


def get_grpc_method_options(
    handler: Callable[..., Any],
) -> tuple[GrpcHandlerType, bool]:
    """Return scheduling metadata attached by :func:`grpc_method`."""
    source = getattr(handler, "__func__", handler)
    return (
        getattr(source, _HANDLER_TYPE_ATTR, GrpcHandlerType.SYNC),
        getattr(source, _REQUIRES_AFFINITY_ATTR, False),
    )


def require_service(service: T | None, feature: str) -> T:
    """Return an enabled service or report the RPC as unimplemented."""
    if service is None:
        raise NotImplementedError(f"{feature} is not enabled on this server")
    return service

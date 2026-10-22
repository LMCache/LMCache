# SPDX-License-Identifier: Apache-2.0
"""Handler binding shared by the sync and aio gRPC request servers."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

# Third Party
import grpc

# First Party
from lmcache.v1.multiprocess.request_handler import (
    BoundRequestHandler,
    HandlerType,
    iter_request_handlers,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import ServiceBinding
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    RequestDecoder,
    ResponseEncoder,
)

GRPC_SERVER_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)
CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"


@dataclass
class GrpcRequestHandler:
    """A module handler bound to one generated gRPC method."""

    operation: str
    handler: Callable[..., Any] | None
    handler_type: HandlerType
    requires_client_affinity: bool
    request_decoder: RequestDecoder
    response_encoder: ResponseEncoder


def bind_module_handlers(
    modules: Sequence[object],
) -> dict[str, BoundRequestHandler]:
    """Collect the decorated handlers of server modules by operation.

    Args:
        modules: Ordered business modules. A later module overrides an
            earlier handler for the same operation.

    Returns:
        The bound handlers, keyed by operation name.
    """
    handlers_by_operation: dict[str, BoundRequestHandler] = {}
    for module in modules:
        for registered in iter_request_handlers(module):
            handlers_by_operation[registered.operation] = registered
    return handlers_by_operation


def build_service_handlers(
    binding: ServiceBinding,
    handlers_by_operation: dict[str, BoundRequestHandler],
) -> dict[str, GrpcRequestHandler]:
    """Match every method of a generated service with its module handler.

    Methods without a module handler are kept with ``handler=None`` so the
    server can answer them with ``UNIMPLEMENTED``.

    Args:
        binding: The generated service to serve.
        handlers_by_operation: Handlers from ``bind_module_handlers``.

    Returns:
        The handler for every method, keyed by the method's full name.

    Raises:
        TypeError: If a module handler does not match its protocol types.
    """
    service_handlers: dict[str, GrpcRequestHandler] = {}
    codec_registry = get_method_codec_registry()
    for method in binding.descriptor.methods:
        method_codec = codec_registry.by_full_name[method.full_name]
        bound_handler = handlers_by_operation.get(method_codec.operation)
        if bound_handler is not None:
            method_codec.validate_handler(bound_handler.handler)
        service_handlers[method.full_name] = GrpcRequestHandler(
            operation=method_codec.operation,
            handler=(bound_handler.handler if bound_handler is not None else None),
            handler_type=(
                bound_handler.options.handler_type
                if bound_handler is not None
                else HandlerType.SERIALIZED
            ),
            requires_client_affinity=(
                bound_handler.options.requires_client_affinity
                if bound_handler is not None
                else False
            ),
            request_decoder=method_codec.request_decoder,
            response_encoder=method_codec.response_encoder,
        )
    return service_handlers


def client_affinity_key(
    context: grpc.ServicerContext | grpc.aio.ServicerContext,
) -> int:
    """Return the key that keeps one client's affinity handlers on one worker.

    Clients send a random ID in their metadata; callers without one fall back
    to their peer address.

    Args:
        context: The call's servicer context, sync or aio.

    Returns:
        A hash identifying the calling client.
    """
    for key, value in context.invocation_metadata() or ():
        if key == CLIENT_ID_METADATA_KEY:
            return hash(value)
    return hash(context.peer())

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
    validate_handler,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import ServiceBinding
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    GrpcMethodCodec,
    get_method_codec_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    RequestDecoder,
    ResponseEncoder,
    compile_request_decoder,
    compile_response_encoder,
)
from lmcache.v1.transport.grpc.endpoint import CLIENT_ID_METADATA_KEY
from lmcache.v1.transport.rpc import get_rpc_spec

GRPC_SERVER_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    # gRPC binds with SO_REUSEPORT by default, so a second server on the same
    # port would start and split the clients instead of failing like ZMQ does.
    ("grpc.so_reuseport", 0),
)


def _normalize_none_type(value: Any) -> Any:
    return None if value is type(None) else value


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


def validate_method_handler(
    codec: GrpcMethodCodec, handler: Callable[..., Any]
) -> None:
    """Validate that a service handler implements a gRPC method's contract.

    Args:
        codec: Compiled codec of the gRPC method the handler serves.
        handler: Bound service implementation method.

    Raises:
        TypeError: If request or response annotations differ from the
            annotated gRPC service contract.
    """
    if get_rpc_spec(codec.operation).streaming:
        validate_handler(codec.operation, handler)
        return
    _decoder, handler_payload_types = compile_request_decoder(
        codec.request_message_class, handler
    )
    _encoder, handler_response_type = compile_response_encoder(
        codec.response_message_class, handler
    )
    if handler_payload_types != codec.payload_types:
        raise TypeError(
            f"{codec.full_name} handler payload annotations "
            f"{handler_payload_types!r} do not match gRPC contract types "
            f"{codec.payload_types!r}"
        )
    if _normalize_none_type(handler_response_type) != _normalize_none_type(
        codec.response_type
    ):
        raise TypeError(
            f"{codec.full_name} handler return annotation "
            f"{handler_response_type!r} does not match gRPC contract type "
            f"{codec.response_type!r}"
        )


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
            validate_method_handler(method_codec, bound_handler.handler)
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

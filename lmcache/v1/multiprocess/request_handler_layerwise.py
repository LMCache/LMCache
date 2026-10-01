# SPDX-License-Identifier: Apache-2.0
"""Handler metadata and discovery for the layer-wise operations.

Mirrors :mod:`lmcache.v1.multiprocess.request_handler`, but checks handlers
against the layer-wise RPC specs and keeps its metadata on a separate
attribute. The shared :func:`iter_request_handlers` therefore never sees
these handlers, so transports that cannot serve them (gRPC) start unchanged.
"""

# Standard
from typing import Any, Callable, TypeVar, get_type_hints
import inspect

# First Party
from lmcache.v1.multiprocess.request_handler import (
    BoundRequestHandler,
    HandlerType,
    RequestHandlerOptions,
)
from lmcache.v1.multiprocess.rpc import RpcOperation
from lmcache.v1.multiprocess.transport.base_layerwise import get_layerwise_rpc_spec

F = TypeVar("F", bound=Callable[..., Any])

_LAYERWISE_HANDLER_OPTIONS_ATTR = "__lmcache_layerwise_request_handler_options__"


def layerwise_request_handler(
    handler_type: HandlerType = HandlerType.SYNC,
    *,
    requires_client_affinity: bool = False,
) -> Callable[[F], F]:
    """Attach scheduling metadata; the operation is the method name.

    A streaming operation is declared ``HandlerType.BLOCKING``: it runs on the
    same worker pools as any blocking handler, and the transport decides from
    the operation whether to answer with one frame or several.
    """
    if requires_client_affinity and handler_type is not HandlerType.BLOCKING:
        raise ValueError("Client affinity requires HandlerType.BLOCKING")

    options = RequestHandlerOptions(
        operation=None,
        handler_type=handler_type,
        requires_client_affinity=requires_client_affinity,
    )

    def decorate(func: F) -> F:
        setattr(func, _LAYERWISE_HANDLER_OPTIONS_ATTR, options)
        return func

    return decorate


def _validate_layerwise_handler(
    operation: RpcOperation, handler: Callable[..., Any]
) -> None:
    """Check payload and return annotations against the layer-wise spec.

    Keyword-only parameters are not payloads (a streaming handler takes its
    ``response_channel`` that way), so only positional ones are compared.
    """
    try:
        spec = get_layerwise_rpc_spec(operation)
    except KeyError:
        raise ValueError(
            f"{handler.__qualname__} is not a layer-wise operation"
        ) from None
    hints = get_type_hints(handler)
    parameters = [
        parameter
        for parameter in inspect.signature(handler).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in {"self", "cls"}
    ]
    payload_types = tuple(hints.get(p.name, p.annotation) for p in parameters)
    if payload_types != spec.payload_types:
        raise TypeError(
            f"Handler {handler.__qualname__} payload annotations "
            f"{payload_types!r} do not match RPC {operation!r} types "
            f"{spec.payload_types!r}"
        )
    if hints.get("return") != spec.response_type:
        raise TypeError(
            f"Handler {handler.__qualname__} return annotation "
            f"{hints.get('return')!r} does not match RPC {operation!r} type "
            f"{spec.response_type!r}"
        )


def iter_layerwise_request_handlers(
    module: object,
) -> tuple[BoundRequestHandler, ...]:
    """Discover a module's layer-wise handlers and validate them."""
    handlers: list[BoundRequestHandler] = []
    module_type = module if inspect.isclass(module) else type(module)
    for name, source in inspect.getmembers(module_type, predicate=callable):
        func = getattr(source, "__func__", source)
        options = getattr(func, _LAYERWISE_HANDLER_OPTIONS_ATTR, None)
        if options is None:
            continue
        handler = getattr(module, name)
        _validate_layerwise_handler(name, handler)
        handlers.append(
            BoundRequestHandler(operation=name, handler=handler, options=options)
        )
    return tuple(handlers)

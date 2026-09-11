# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request handler metadata and discovery."""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, get_type_hints
import inspect

# First Party
from lmcache.v1.multiprocess.protocol import (
    RpcOperation,
    get_request_message_class,
    get_response_message_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType

F = TypeVar("F", bound=Callable[..., Any])

_HANDLER_OPTIONS_ATTR = "__lmcache_request_handler_options__"


@dataclass(frozen=True)
class RequestHandlerOptions:
    """Describe how a transport-neutral request handler is executed.

    Args:
        operation: RPC route handled by the method, inferred from its
            ``handle_<operation>`` name.
        handler_type: Whether to run the method inline or on a worker.
        requires_client_affinity: Whether requests from one client must use
            the same worker.
    """

    operation: RpcOperation
    handler_type: HandlerType
    requires_client_affinity: bool


@dataclass(frozen=True)
class BoundRequestHandler:
    """Pair a bound module method with its request handler metadata.

    Args:
        handler: Bound module method that handles the request.
        options: Transport-neutral scheduling metadata for the method.
    """

    handler: Callable[..., Any]
    options: RequestHandlerOptions


def request_handler(
    handler_type: HandlerType = HandlerType.SYNC,
    *,
    requires_client_affinity: bool = False,
) -> Callable[[F], F]:
    """Mark a module method as a transport-neutral request handler.

    Args:
        handler_type: Whether to execute inline or on a worker.
        requires_client_affinity: Whether requests from one client must use
            the same worker.

    Returns:
        A decorator that attaches immutable handler metadata to a method.

    Raises:
        ValueError: If client affinity is requested for a handler that is not
            ``HandlerType.BLOCKING``.
    """
    if requires_client_affinity and handler_type is not HandlerType.BLOCKING:
        raise ValueError("Client affinity requires HandlerType.BLOCKING")

    def decorate(func: F) -> F:
        operation = func.__name__.removeprefix("handle_")
        if operation == func.__name__:
            raise ValueError(
                "Request handlers must use the name handle_<operation>; "
                f"got {func.__qualname__}"
            )
        options = RequestHandlerOptions(
            operation=operation,
            handler_type=handler_type,
            requires_client_affinity=requires_client_affinity,
        )
        setattr(func, _HANDLER_OPTIONS_ATTR, options)
        return func

    return decorate


def get_request_handler_options(
    handler: Callable[..., Any],
) -> RequestHandlerOptions | None:
    """Return transport-neutral metadata attached to a handler.

    Args:
        handler: Bound or unbound callable to inspect.

    Returns:
        Handler metadata, or ``None`` if the callable is not a request
        handler.
    """
    source = getattr(handler, "__func__", handler)
    return getattr(source, _HANDLER_OPTIONS_ATTR, None)


def iter_request_handlers(module: object) -> tuple[BoundRequestHandler, ...]:
    """Discover and validate request handlers exposed by a module.

    Args:
        module: Business module whose decorated methods should be discovered.

    Returns:
        Bound request handlers ordered by method name.

    Raises:
        ValueError: If one module registers an operation more than once, or its
            annotations disagree with the Python RPC message contract.
    """
    handlers: list[BoundRequestHandler] = []
    seen: set[RpcOperation] = set()
    module_type = module if inspect.isclass(module) else type(module)
    for name, source in inspect.getmembers(module_type, predicate=callable):
        options = get_request_handler_options(source)
        if options is None:
            continue
        handler = getattr(module, name)
        if options.operation in seen:
            raise ValueError(
                f"{module_type.__name__} has multiple handlers for {options.operation}"
            )
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
        expected_request_class = get_request_message_class(options.operation)
        if len(parameters) != 1 or hints.get(parameters[0].name) is not (
            expected_request_class
        ):
            raise ValueError(
                f"{module_type.__name__}.{handler.__name__} must accept exactly "
                f"one {expected_request_class.__name__}"
            )
        expected_response_class = get_response_message_class(options.operation)
        if hints.get("return") is not expected_response_class:
            raise ValueError(
                f"{module_type.__name__}.{handler.__name__} must return "
                f"{expected_response_class.__name__}"
            )
        seen.add(options.operation)
        handlers.append(BoundRequestHandler(handler=handler, options=options))
    return tuple(handlers)


__all__ = [
    "BoundRequestHandler",
    "RequestHandlerOptions",
    "get_request_handler_options",
    "iter_request_handlers",
    "request_handler",
]

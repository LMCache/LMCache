# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request handler metadata and discovery."""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast, get_type_hints
import inspect

# First Party
from lmcache.v1.multiprocess.protocol import (
    get_handler_type,
    get_request_message_class,
    get_response_message_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.rpc_messages import (
    RpcRequest,
    unwrap_request_message,
    wrap_response_message,
)

F = TypeVar("F", bound=Callable[..., Any])

_HANDLER_OPTIONS_ATTR = "__lmcache_request_handler_options__"
_MISSING = object()


@dataclass(frozen=True)
class RequestHandlerOptions:
    """Describe how a transport-neutral request handler is executed.

    Args:
        request_type: Protocol request handled by the method.
        handler_type: Whether to run the method inline or on a worker.
        requires_client_affinity: Whether requests from one client must use
            the same worker.
    """

    request_type: RequestType
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
    request_type: RequestType,
    handler_type: HandlerType = HandlerType.SYNC,
    *,
    requires_client_affinity: bool = False,
) -> Callable[[F], F]:
    """Mark a module method as a transport-neutral request handler.

    Args:
        request_type: Protocol request handled by the method.
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

    options = RequestHandlerOptions(
        request_type=request_type,
        handler_type=handler_type,
        requires_client_affinity=requires_client_affinity,
    )

    request_class = get_request_message_class(request_type)
    response_class = get_response_message_class(request_type)

    def decorate(func: F) -> F:
        hints = get_type_hints(func)
        parameters = tuple(inspect.signature(func).parameters.values())[1:]
        if (
            len(parameters) == 1
            and hints.get(parameters[0].name) is request_class
            and hints.get("return") is response_class
        ):
            setattr(func, _HANDLER_OPTIONS_ATTR, options)
            return func

        def dispatch(
            self: object,
            request: object = _MISSING,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if request is _MISSING:
                return func(self, *args, **kwargs)
            if not isinstance(request, request_class) or args or kwargs:
                return func(self, request, *args, **kwargs)
            result = func(self, *unwrap_request_message(cast(RpcRequest, request)))
            return wrap_response_message(request_type.name, result)

        dispatch.__name__ = func.__name__
        dispatch.__qualname__ = func.__qualname__
        dispatch.__doc__ = func.__doc__
        dispatch.__module__ = func.__module__
        dispatch.__annotations__ = {
            "request": request_class,
            "return": response_class,
        }
        setattr(dispatch, _HANDLER_OPTIONS_ATTR, options)
        return dispatch  # type: ignore[return-value]

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
        ValueError: If one module registers a request more than once, or its
            annotation disagrees with the protocol definition.
    """
    handlers: list[BoundRequestHandler] = []
    seen: set[RequestType] = set()
    module_type = module if inspect.isclass(module) else type(module)
    for name, source in inspect.getmembers(module_type, predicate=callable):
        options = get_request_handler_options(source)
        if options is None:
            continue
        handler = getattr(module, name)
        if options.request_type in seen:
            raise ValueError(
                f"{module_type.__name__} has multiple handlers for "
                f"{options.request_type.name}"
            )
        expected_handler_type = get_handler_type(options.request_type)
        if options.handler_type is not expected_handler_type:
            raise ValueError(
                f"{module_type.__name__}.{handler.__name__} declares "
                f"{options.handler_type.name} for {options.request_type.name}, "
                f"but the protocol declares {expected_handler_type.name}"
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
        expected_request_type = get_request_message_class(options.request_type)
        if len(parameters) != 1 or hints.get(parameters[0].name) is not (
            expected_request_type
        ):
            raise ValueError(
                f"{module_type.__name__}.{handler.__name__} must accept exactly "
                f"one {expected_request_type.__name__}"
            )
        expected_response_type = get_response_message_class(options.request_type)
        if hints.get("return") is not expected_response_type:
            raise ValueError(
                f"{module_type.__name__}.{handler.__name__} must return "
                f"{expected_response_type.__name__}"
            )
        seen.add(options.request_type)
        handlers.append(BoundRequestHandler(handler=handler, options=options))
    return tuple(handlers)


__all__ = [
    "BoundRequestHandler",
    "RequestHandlerOptions",
    "get_request_handler_options",
    "iter_request_handlers",
    "request_handler",
]

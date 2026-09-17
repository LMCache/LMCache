# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral request handler metadata and discovery."""

# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, TypeVar, get_type_hints
import inspect

# First Party
from lmcache.v1.multiprocess.rpc import RpcOperation, get_rpc_spec

F = TypeVar("F", bound=Callable[..., Any])

_HANDLER_OPTIONS_ATTR = "__lmcache_request_handler_options__"


class HandlerType(Enum):
    """Select how a request handler is executed."""

    SYNC = auto()
    BLOCKING = auto()
    NON_BLOCKING = auto()


@dataclass(frozen=True)
class RequestHandlerOptions:
    """Describe how a transport-neutral request handler is executed.

    Args:
        operation: RPC operation handled by the method. ``None`` means infer
            the operation from the decorated method name.
        handler_type: Whether to run the method inline or on a worker.
        requires_client_affinity: Whether requests from one client must use
            the same worker.
    """

    operation: RpcOperation | None
    handler_type: HandlerType
    requires_client_affinity: bool


@dataclass(frozen=True)
class BoundRequestHandler:
    """Pair a bound module method with its resolved handler metadata.

    Args:
        operation: RPC operation implemented by the handler.
        handler: Bound module method that handles the request.
        options: Transport-neutral scheduling metadata for the method.
    """

    operation: RpcOperation
    handler: Callable[..., Any]
    options: RequestHandlerOptions


def request_handler(
    handler_type: HandlerType = HandlerType.SYNC,
    *,
    operation: RpcOperation | None = None,
    requires_client_affinity: bool = False,
) -> Callable[[F], F]:
    """Mark a module method as a transport-neutral request handler.

    The RPC operation defaults to the decorated method name. An explicit name
    is only needed when a legacy implementation method cannot be renamed.

    Args:
        handler_type: Whether to execute inline or on a worker.
        operation: Optional RPC operation override.
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
        operation=operation,
        handler_type=handler_type,
        requires_client_affinity=requires_client_affinity,
    )

    def decorate(func: F) -> F:
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


def _normalize_none_type(value: Any) -> Any:
    return type(None) if value is None else value


def _validate_handler(operation: RpcOperation, handler: Callable[..., Any]) -> None:
    spec = get_rpc_spec(operation)
    handler_signature = inspect.signature(handler)
    hints = get_type_hints(handler)
    parameters = tuple(handler_signature.parameters.values())
    if parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    payload_types = tuple(
        hints.get(parameter.name, parameter.annotation) for parameter in parameters
    )
    response_type = hints.get("return", handler_signature.return_annotation)
    if payload_types != spec.payload_types:
        raise TypeError(
            f"Handler {handler.__qualname__} payload annotations "
            f"{payload_types!r} do not match RPC {operation!r} types "
            f"{spec.payload_types!r}"
        )
    if _normalize_none_type(response_type) != _normalize_none_type(spec.response_type):
        raise TypeError(
            f"Handler {handler.__qualname__} return annotation "
            f"{response_type!r} does not match RPC {operation!r} type "
            f"{spec.response_type!r}"
        )


def iter_request_handlers(module: object) -> tuple[BoundRequestHandler, ...]:
    """Discover and validate request handlers exposed by a module.

    Args:
        module: Business module whose decorated methods should be discovered.

    Returns:
        Bound request handlers ordered by method name.

    Raises:
        KeyError: If a handler names an unknown RPC operation.
        TypeError: If handler annotations differ from the RPC contract.
        ValueError: If one module registers an operation more than once.
    """
    handlers: list[BoundRequestHandler] = []
    seen: set[RpcOperation] = set()
    module_type = module if inspect.isclass(module) else type(module)
    for name, source in inspect.getmembers(module_type, predicate=callable):
        options = get_request_handler_options(source)
        if options is None:
            continue
        operation = options.operation or name
        handler = getattr(module, name)
        if operation in seen:
            raise ValueError(
                f"{module_type.__name__} has multiple handlers for {operation!r}"
            )
        _validate_handler(operation, handler)
        seen.add(operation)
        handlers.append(
            BoundRequestHandler(
                operation=operation,
                handler=handler,
                options=options,
            )
        )
    return tuple(handlers)

# SPDX-License-Identifier: Apache-2.0
"""Dynamic loading for out-of-tree multiprocess server modules."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast
import inspect
import json

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.engine_module import DiscoverableModule, EngineModule
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.utils.subclass_discovery import discover_subclasses

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

logger = init_logger(__name__)

ExtServerModuleMethod = Callable[[bytes], bytes]
_ServerModuleDecorated = TypeVar(
    "_ServerModuleDecorated",
    bound=Callable[..., bytes],
)
_SERVER_MODULE_HANDLER_ATTR = "__lmcache_server_module_handler__"
_GRPC_SERVICE_REGISTRAR = "register_grpc_services"
_ZMQ_SERVICE_REGISTRAR = "register_zmq_services"


@dataclass(frozen=True)
class ExtServerModuleSpec:
    """Configuration for explicitly importing out-of-tree module classes.

    Args:
        module_path: Dotted Python import path containing one or more concrete
            :class:`DiscoverableModule` subclasses.
        config: Plugin-specific JSON-compatible configuration exposed through
            :meth:`ModuleBuildContext.config_for` to every class in the module.
    """

    module_path: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtServerModuleSpec":
        """Create a spec from a parsed ``--server-module`` JSON object.

        Args:
            data: JSON object containing ``module_path`` and an optional
                ``config`` field.

        Returns:
            A validated :class:`ExtServerModuleSpec`.

        Raises:
            ValueError: If the object has invalid field types.
        """
        unknown_fields = sorted(set(data) - {"module_path", "config"})
        if unknown_fields:
            raise ValueError(
                "unknown server module fields: %s" % ", ".join(unknown_fields)
            )

        module_path = data.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            raise ValueError("server module module_path must be a non-empty string")

        config = data.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("server module config must be a JSON object")

        return cls(module_path=module_path, config=config)


@dataclass(frozen=True)
class ExtServerModuleHandlerOptions:
    """Metadata for one namespaced server-module extension handler.

    Args:
        method: Namespaced method name inside the server-module envelope.
    """

    method: str


@dataclass(frozen=True)
class ExtServerModuleBoundHandler:
    """Pair a bound extension handler with its namespaced method."""

    method: str
    handler: ExtServerModuleMethod


def server_module_handler(
    method: str,
) -> Callable[[_ServerModuleDecorated], _ServerModuleDecorated]:
    """Mark a module method as a server-module extension handler.

    Extension handlers receive plugin-owned request bytes and return
    plugin-owned response bytes. LMCache transports only route the
    namespaced method and opaque bytes; the plugin package owns any higher
    level schema, encoding, and compatibility policy.

    Args:
        method: Namespaced method name, such as ``"my_package.echo"``.

    Returns:
        A decorator that attaches immutable handler metadata to a method.

    Raises:
        ValueError: If ``method`` is empty or not namespaced.
    """
    if not method or "." not in method:
        raise ValueError("server module handler method must be namespaced")

    options = ExtServerModuleHandlerOptions(method=method)

    def decorate(func: _ServerModuleDecorated) -> _ServerModuleDecorated:
        setattr(func, _SERVER_MODULE_HANDLER_ATTR, options)
        return func

    return decorate


class ExtServerModuleRouter:
    """Dispatch the stable server-module envelope to extension handlers."""

    def __init__(
        self,
        ctx: MPCacheServerContext,
        handlers: Sequence[ExtServerModuleBoundHandler],
    ) -> None:
        self._ctx = ctx
        self._handlers = {handler.method: handler.handler for handler in handlers}

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared server context."""
        return self._ctx

    def report_status(self) -> dict:
        """Return the registered server-module extension methods."""
        return {
            "server_module_plugins": {
                "methods": sorted(self._handlers),
            },
        }

    def close(self) -> None:
        """Release resources owned by the router."""
        return None

    @request_handler(HandlerType.BLOCKING)
    def server_module_call(
        self,
        request: ServerModuleCallRequest,
    ) -> ServerModuleCallResponse:
        """Dispatch one namespaced extension request.

        Args:
            request: Opaque extension request envelope.

        Returns:
            Opaque extension response envelope. Handler exceptions are reported
            as ``success=False`` so clients receive a response rather than a
            dropped request.
        """
        handler = self._handlers.get(request.method)
        if handler is None:
            return ServerModuleCallResponse(
                success=False,
                payload=b"",
                error="unknown server module method: %s" % request.method,
            )

        try:
            payload = handler(request.payload)
        except Exception as exc:
            logger.exception(
                "server module method %s failed",
                request.method,
            )
            return ServerModuleCallResponse(
                success=False,
                payload=b"",
                error=str(exc),
            )
        return ServerModuleCallResponse(success=True, payload=payload, error="")


def parse_server_module_specs(raw_specs: Sequence[str]) -> list[ExtServerModuleSpec]:
    """Parse ``--server-module`` JSON strings into validated specs.

    Args:
        raw_specs: JSON strings. Each string may be one object or a list of
            objects, which makes environment-specific launchers easier to
            compose.

    Returns:
        The parsed server-module specs in command-line order.

    Raises:
        ValueError: If JSON parsing fails or any entry has an invalid shape.
    """
    specs: list[ExtServerModuleSpec] = []
    for raw_spec in raw_specs:
        try:
            parsed = json.loads(raw_spec)
        except json.JSONDecodeError as exc:
            raise ValueError("--server-module must be valid JSON: %s" % exc) from exc

        entries = parsed if isinstance(parsed, list) else [parsed]
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("--server-module entries must be JSON objects")
            specs.append(ExtServerModuleSpec.from_dict(entry))
    return specs


def load_server_module_classes(
    specs: Sequence[ExtServerModuleSpec],
) -> list[tuple[type[DiscoverableModule], dict[str, Any]]]:
    """Import configured out-of-tree ``DiscoverableModule`` classes.

    Args:
        specs: Explicit Python modules to import, in command-line order.

    Returns:
        Pairs of module class and plugin-specific configuration. Classes within
        one Python module are ordered by qualified name.

    Raises:
        ImportError: If a plugin module cannot be imported.
        TypeError: If a configured Python module defines no concrete
            :class:`DiscoverableModule` subclass.
        ValueError: If the same module class is configured more than once.
    """
    loaded: list[tuple[type[DiscoverableModule], dict[str, Any]]] = []
    seen: set[type[DiscoverableModule]] = set()
    for spec in specs:
        module_classes = sorted(
            discover_subclasses(
                spec.module_path,
                DiscoverableModule,  # type: ignore[type-abstract]
                walk_submodules=False,
            ),
            key=lambda module_type: module_type.__qualname__,
        )
        if not module_classes:
            raise TypeError(
                f"server module {spec.module_path!r} defines no concrete "
                "DiscoverableModule subclass"
            )
        for module_type in module_classes:
            if module_type in seen:
                raise ValueError(
                    f"server module class {module_type.__module__}."
                    f"{module_type.__qualname__} is configured more than once"
                )
            seen.add(module_type)
            loaded.append((module_type, dict(spec.config)))
    return loaded


def build_server_module_router(
    ctx: MPCacheServerContext,
    modules: Sequence[EngineModule],
) -> ExtServerModuleRouter | None:
    """Build a dispatcher for extension handlers exposed by modules.

    Args:
        ctx: Shared server context for the router.
        modules: Plugin modules to inspect for ``@server_module_handler``.

    Returns:
        A router module when at least one extension handler exists; otherwise
        ``None``.

    Raises:
        ValueError: If two modules register the same extension method.
    """
    handlers_by_method: dict[str, ExtServerModuleBoundHandler] = {}
    for module in modules:
        for handler in iter_server_module_handlers(module):
            if handler.method in handlers_by_method:
                raise ValueError("duplicate server module method: %s" % handler.method)
            handlers_by_method[handler.method] = handler

    if not handlers_by_method:
        return None
    return ExtServerModuleRouter(ctx, tuple(handlers_by_method.values()))


def register_grpc_services(
    modules: Sequence[object],
    server: Any,
) -> None:
    """Register out-of-tree gRPC services exposed by server modules.

    Args:
        modules: Ordered server modules to inspect for a
            ``register_grpc_services(server)`` method.
        server: The concrete ``grpc.Server`` owned by the gRPC transport.

    Raises:
        TypeError: If a module exposes a non-callable registrar attribute.
    """
    _register_transport_services(modules, server, _GRPC_SERVICE_REGISTRAR)


def register_zmq_services(
    modules: Sequence[object],
    server: Any,
) -> None:
    """Register out-of-tree ZMQ services exposed by server modules.

    Args:
        modules: Ordered server modules to inspect for a
            ``register_zmq_services(server)`` method.
        server: The concrete ``MessageQueueServer`` owned by the ZMQ transport.

    Raises:
        TypeError: If a module exposes a non-callable registrar attribute.
    """
    _register_transport_services(modules, server, _ZMQ_SERVICE_REGISTRAR)


def iter_server_module_handlers(
    module: object,
) -> tuple[ExtServerModuleBoundHandler, ...]:
    """Discover namespaced extension handlers exposed by a module.

    Args:
        module: Business module whose decorated methods should be discovered.

    Returns:
        Bound extension handlers ordered by method name.
    """
    handlers: list[ExtServerModuleBoundHandler] = []
    module_type = module if inspect.isclass(module) else type(module)
    for name, source in inspect.getmembers(module_type, predicate=callable):
        options = getattr(source, _SERVER_MODULE_HANDLER_ATTR, None)
        if options is None:
            continue
        if not isinstance(options, ExtServerModuleHandlerOptions):
            raise TypeError(
                f"{module_type.__name__}.{name} has invalid server-module "
                "handler metadata"
            )
        handler = cast(ExtServerModuleMethod, getattr(module, name))
        handlers.append(
            ExtServerModuleBoundHandler(
                method=options.method,
                handler=handler,
            )
        )
    return tuple(sorted(handlers, key=lambda handler: handler.method))


def _register_transport_services(
    modules: Sequence[object],
    server: Any,
    registrar_name: str,
) -> None:
    """Call optional transport service registrars exposed by modules."""
    for module in modules:
        registrar = getattr(module, registrar_name, None)
        if registrar is None:
            continue
        if not callable(registrar):
            raise TypeError(
                f"{type(module).__name__}.{registrar_name} must be callable"
            )
        cast(Callable[[Any], None], registrar)(server)

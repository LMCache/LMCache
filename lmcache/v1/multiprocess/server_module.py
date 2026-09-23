# SPDX-License-Identifier: Apache-2.0
"""Dynamic loading for out-of-tree multiprocess server modules."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast
import importlib
import inspect
import json

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.engine_module import EngineModule
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

logger = init_logger(__name__)

ServerModuleMethod = Callable[[bytes], bytes]
TransportServiceRegistrar = Callable[[Any], None]
_ServerModuleDecorated = TypeVar(
    "_ServerModuleDecorated",
    bound=Callable[..., bytes],
)
_SERVER_MODULE_HANDLER_ATTR = "__lmcache_server_module_handler__"
_GRPC_SERVICE_REGISTRAR = "register_grpc_services"
_ZMQ_SERVICE_REGISTRAR = "register_zmq_services"


@dataclass(frozen=True)
class ServerModuleSpec:
    """Configuration for one dynamically loaded server-module factory.

    Args:
        module_path: Dotted Python import path containing the factory.
        factory_name: Name of the callable inside ``module_path``. Defaults to
            ``build_server_modules``.
        config: Plugin-specific JSON-compatible configuration passed to the
            factory in :class:`ServerModuleBuildContext`.
    """

    module_path: str
    factory_name: str = "build_server_modules"
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerModuleSpec":
        """Create a spec from a parsed ``--server-module`` JSON object.

        Args:
            data: JSON object containing ``module_path`` and optional
                ``factory_name`` / ``config`` fields.

        Returns:
            A validated :class:`ServerModuleSpec`.

        Raises:
            ValueError: If the object has invalid field types.
        """
        module_path = data.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            raise ValueError("server module module_path must be a non-empty string")

        factory_name = data.get("factory_name", "build_server_modules")
        if not isinstance(factory_name, str) or not factory_name:
            raise ValueError("server module factory_name must be a non-empty string")

        config = data.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("server module config must be a JSON object")

        return cls(
            module_path=module_path,
            factory_name=factory_name,
            config=config,
        )


@dataclass(frozen=True)
class ServerModuleBuildContext:
    """Context passed to an out-of-tree server-module factory.

    Args:
        server_context: Shared :class:`MPCacheServerContext` used by built-in
            modules.
        mp_config: Parsed multiprocess server configuration.
        coordinator_config: Parsed coordinator configuration.
        modules: Built-in modules already assembled before this plugin runs.
        config: Plugin-specific JSON-compatible configuration from the
            corresponding :class:`ServerModuleSpec`.
    """

    server_context: MPCacheServerContext
    mp_config: MPServerConfig
    coordinator_config: CoordinatorConfig
    modules: tuple[EngineModule, ...]
    config: dict[str, Any]


@dataclass(frozen=True)
class ServerModuleComponents:
    """Out-of-tree server extension components returned by a factory.

    Args:
        modules: Transport-neutral business modules to compose into
            :class:`MPCacheServer`.
        grpc_service_registrars: Callables that register package-owned gRPC
            services with the concrete ``grpc.Server``.
        zmq_service_registrars: Callables that register or start package-owned
            ZMQ services using the concrete ``MessageQueueServer``.
    """

    modules: Sequence[EngineModule] = ()
    grpc_service_registrars: Sequence[TransportServiceRegistrar] = ()
    zmq_service_registrars: Sequence[TransportServiceRegistrar] = ()


ServerModuleFactory = Callable[
    [ServerModuleBuildContext],
    EngineModule | Sequence[EngineModule] | ServerModuleComponents | None,
]


@dataclass(frozen=True)
class ServerModuleHandlerOptions:
    """Metadata for one namespaced server-module extension handler.

    Args:
        method: Namespaced method name inside the server-module envelope.
    """

    method: str


@dataclass(frozen=True)
class BoundServerModuleHandler:
    """Pair a bound extension handler with its namespaced method."""

    method: str
    handler: ServerModuleMethod


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

    options = ServerModuleHandlerOptions(method=method)

    def decorate(func: _ServerModuleDecorated) -> _ServerModuleDecorated:
        setattr(func, _SERVER_MODULE_HANDLER_ATTR, options)
        return func

    return decorate


class ServerModuleRouter:
    """Dispatch the stable server-module envelope to extension handlers."""

    def __init__(
        self,
        ctx: MPCacheServerContext,
        handlers: Sequence[BoundServerModuleHandler],
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


def parse_server_module_specs(raw_specs: Sequence[str]) -> list[ServerModuleSpec]:
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
    specs: list[ServerModuleSpec] = []
    for raw_spec in raw_specs:
        try:
            parsed = json.loads(raw_spec)
        except json.JSONDecodeError as exc:
            raise ValueError("--server-module must be valid JSON: %s" % exc) from exc

        entries = parsed if isinstance(parsed, list) else [parsed]
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("--server-module entries must be JSON objects")
            specs.append(ServerModuleSpec.from_dict(entry))
    return specs


def load_server_modules(
    specs: Sequence[ServerModuleSpec],
    *,
    server_context: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
    built_modules: Sequence[EngineModule],
) -> list[EngineModule]:
    """Load out-of-tree server modules from configured factories.

    Args:
        specs: Dynamic module factories to load.
        server_context: Shared server context passed to factories.
        mp_config: Parsed multiprocess server configuration.
        coordinator_config: Parsed coordinator configuration.
        built_modules: Built-in modules assembled before plugin loading.

    Returns:
        Dynamically loaded modules, ordered by ``specs``.

    Raises:
        ImportError: If a plugin module cannot be imported.
        AttributeError: If the configured factory is missing.
        TypeError: If a factory is not callable or returns a non-module value.
    """
    return list(
        load_server_module_components(
            specs,
            server_context=server_context,
            mp_config=mp_config,
            coordinator_config=coordinator_config,
            built_modules=built_modules,
        ).modules
    )


def load_server_module_components(
    specs: Sequence[ServerModuleSpec],
    *,
    server_context: MPCacheServerContext,
    mp_config: MPServerConfig,
    coordinator_config: CoordinatorConfig,
    built_modules: Sequence[EngineModule],
) -> ServerModuleComponents:
    """Load out-of-tree modules and service registrars from factories.

    Args:
        specs: Dynamic extension factories to load.
        server_context: Shared server context passed to factories.
        mp_config: Parsed multiprocess server configuration.
        coordinator_config: Parsed coordinator configuration.
        built_modules: Built-in modules assembled before plugin loading.

    Returns:
        Aggregated modules and transport-specific service registrars, ordered by
        ``specs``.

    Raises:
        ImportError: If a plugin module cannot be imported.
        AttributeError: If the configured factory is missing.
        TypeError: If a factory is not callable or returns an invalid value.
    """
    loaded: list[EngineModule] = []
    grpc_service_registrars: list[TransportServiceRegistrar] = []
    zmq_service_registrars: list[TransportServiceRegistrar] = []
    for spec in specs:
        module = importlib.import_module(spec.module_path)
        factory = getattr(module, spec.factory_name)
        if not callable(factory):
            raise TypeError(
                "server module factory %s.%s is not callable"
                % (spec.module_path, spec.factory_name)
            )

        build_context = ServerModuleBuildContext(
            server_context=server_context,
            mp_config=mp_config,
            coordinator_config=coordinator_config,
            modules=tuple([*built_modules, *loaded]),
            config=spec.config,
        )
        components = _coerce_components(factory(build_context), spec)
        loaded.extend(components.modules)
        grpc_service_registrars.extend(components.grpc_service_registrars)
        zmq_service_registrars.extend(components.zmq_service_registrars)
    return ServerModuleComponents(
        modules=tuple(loaded),
        grpc_service_registrars=tuple(grpc_service_registrars),
        zmq_service_registrars=tuple(zmq_service_registrars),
    )


def build_server_module_router(
    ctx: MPCacheServerContext,
    modules: Sequence[EngineModule],
) -> ServerModuleRouter | None:
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
    handlers_by_method: dict[str, BoundServerModuleHandler] = {}
    for module in modules:
        for handler in iter_server_module_handlers(module):
            if handler.method in handlers_by_method:
                raise ValueError("duplicate server module method: %s" % handler.method)
            handlers_by_method[handler.method] = handler

    if not handlers_by_method:
        return None
    return ServerModuleRouter(ctx, tuple(handlers_by_method.values()))


def register_grpc_services(
    modules: Sequence[object],
    server: Any,
    service_registrars: Sequence[TransportServiceRegistrar] = (),
) -> None:
    """Register out-of-tree gRPC services exposed by server modules.

    Args:
        modules: Ordered server modules to inspect for a
            ``register_grpc_services(server)`` method.
        server: The concrete ``grpc.Server`` owned by the gRPC transport.
        service_registrars: Explicit package-owned gRPC service registrars
            returned by server-module factories.

    Raises:
        TypeError: If a module exposes a non-callable registrar attribute.
    """
    _call_transport_service_registrars(service_registrars, server)
    _register_transport_services(modules, server, _GRPC_SERVICE_REGISTRAR)


def register_zmq_services(
    modules: Sequence[object],
    server: Any,
    service_registrars: Sequence[TransportServiceRegistrar] = (),
) -> None:
    """Register out-of-tree ZMQ services exposed by server modules.

    Args:
        modules: Ordered server modules to inspect for a
            ``register_zmq_services(server)`` method.
        server: The concrete ``MessageQueueServer`` owned by the ZMQ transport.
        service_registrars: Explicit package-owned ZMQ service registrars
            returned by server-module factories.

    Raises:
        TypeError: If a module exposes a non-callable registrar attribute.
    """
    _call_transport_service_registrars(service_registrars, server)
    _register_transport_services(modules, server, _ZMQ_SERVICE_REGISTRAR)


def iter_server_module_handlers(module: object) -> tuple[BoundServerModuleHandler, ...]:
    """Discover namespaced extension handlers exposed by a module.

    Args:
        module: Business module whose decorated methods should be discovered.

    Returns:
        Bound extension handlers ordered by method name.
    """
    handlers: list[BoundServerModuleHandler] = []
    module_type = module if inspect.isclass(module) else type(module)
    for name, source in inspect.getmembers(module_type, predicate=callable):
        options = getattr(source, _SERVER_MODULE_HANDLER_ATTR, None)
        if options is None:
            continue
        if not isinstance(options, ServerModuleHandlerOptions):
            raise TypeError(
                f"{module_type.__name__}.{name} has invalid server-module "
                "handler metadata"
            )
        handler = cast(ServerModuleMethod, getattr(module, name))
        handlers.append(
            BoundServerModuleHandler(
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
        cast(TransportServiceRegistrar, registrar)(server)


def _call_transport_service_registrars(
    service_registrars: Sequence[TransportServiceRegistrar],
    server: Any,
) -> None:
    """Call explicit transport service registrars returned by factories."""
    for registrar in service_registrars:
        if not callable(registrar):
            raise TypeError("transport service registrar must be callable")
        registrar(server)


def _coerce_components(
    value: EngineModule | Sequence[EngineModule] | ServerModuleComponents | None,
    spec: ServerModuleSpec,
) -> ServerModuleComponents:
    """Normalize one factory return value into extension components."""
    if isinstance(value, ServerModuleComponents):
        return ServerModuleComponents(
            modules=tuple(_coerce_modules(value.modules, spec)),
            grpc_service_registrars=tuple(
                _coerce_service_registrars(
                    value.grpc_service_registrars,
                    spec,
                    "grpc_service_registrars",
                )
            ),
            zmq_service_registrars=tuple(
                _coerce_service_registrars(
                    value.zmq_service_registrars,
                    spec,
                    "zmq_service_registrars",
                )
            ),
        )
    return ServerModuleComponents(modules=tuple(_coerce_modules(value, spec)))


def _coerce_modules(
    value: EngineModule | Sequence[EngineModule] | None,
    spec: ServerModuleSpec,
) -> list[EngineModule]:
    """Normalize one factory return value into a list of engine modules."""
    if value is None:
        return []
    if _is_engine_module(value):
        return [cast(EngineModule, value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        modules: list[EngineModule] = []
        for module in value:
            if not _is_engine_module(module):
                raise TypeError(
                    "server module factory %s.%s returned a sequence containing "
                    "a non-EngineModule value: %r"
                    % (spec.module_path, spec.factory_name, module)
                )
            modules.append(cast(EngineModule, module))
        return modules
    raise TypeError(
        "server module factory %s.%s must return an EngineModule, a sequence "
        "of EngineModule, or None" % (spec.module_path, spec.factory_name)
    )


def _coerce_service_registrars(
    values: Sequence[TransportServiceRegistrar],
    spec: ServerModuleSpec,
    field_name: str,
) -> list[TransportServiceRegistrar]:
    """Validate transport service registrars from one components object."""
    registrars: list[TransportServiceRegistrar] = []
    for registrar in values:
        if not callable(registrar):
            raise TypeError(
                "server module factory %s.%s returned %s containing a "
                "non-callable value: %r"
                % (spec.module_path, spec.factory_name, field_name, registrar)
            )
        registrars.append(registrar)
    return registrars


def _is_engine_module(value: object) -> bool:
    """Return whether *value* exposes the EngineModule structural contract."""
    return (
        hasattr(value, "context")
        and callable(getattr(value, "report_status", None))
        and callable(getattr(value, "close", None))
    )

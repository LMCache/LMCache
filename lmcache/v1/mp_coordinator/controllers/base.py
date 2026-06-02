# SPDX-License-Identifier: Apache-2.0
"""The pluggable controller seam.

A controller owns one functional domain (registration, and later quota,
routing, KV ops). It declares which message types it handles; the manager
merges those declarations into a single dispatch table at startup. Adding a
capability therefore means adding a controller and its messages and listing the
controller in the manager -- no change to dispatch logic.

Handlers are keyed by concrete message type but typed against the channel base
class (:class:`PushMsg` / :class:`ReqMsg`); a handler narrows to its concrete
type internally. This keeps the dispatch table soundly typed without a
``Any``-typed registry.
"""

# Standard
from typing import Awaitable, Callable

# First Party
from lmcache.v1.mp_coordinator.lifecycle import LifecycleHooks
from lmcache.v1.mp_coordinator.message import (
    PushMsg,
    ReqMsg,
    ReqRetMsg,
)
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.mp_coordinator.transport import CoordinatorTransport

# A push handler consumes a fire-and-forget message and returns nothing.
PushHandler = Callable[[PushMsg], Awaitable[None]]
# A request handler consumes a request and returns a reply message.
ReqHandler = Callable[[ReqMsg], Awaitable[ReqRetMsg]]


class ControllerContext:
    """Shared collaborators handed to every controller at ``post_init``.

    Controllers reach the registry, the transport (for server-initiated push
    and instance reach lifecycle), lifecycle hooks, and sibling controllers
    only through this object -- they never import one another directly, and no
    controller touches a socket.

    Args:
        registry: The shared instance registry.
        transport: The coordinator transport (push + reach lifecycle).
        lifecycle: Join/leave hook registry.
    """

    def __init__(
        self,
        registry: InstanceRegistry,
        transport: CoordinatorTransport,
        lifecycle: LifecycleHooks,
    ) -> None:
        """Initialize the context with shared collaborators."""
        self.registry = registry
        self.transport = transport
        self.lifecycle = lifecycle
        self._controllers: dict[type, "Controller"] = {}

    def register_controller(self, controller: "Controller") -> None:
        """Make a controller discoverable by its concrete type.

        Args:
            controller: The controller instance to register.
        """
        self._controllers[type(controller)] = controller

    def get_controller(self, controller_type: type) -> "Controller":
        """Return a sibling controller by its concrete type.

        Args:
            controller_type: The controller class to look up.

        Returns:
            The registered controller instance.

        Raises:
            KeyError: If no controller of that type is registered.
        """
        if controller_type not in self._controllers:
            raise KeyError(f"No controller registered for {controller_type.__name__}")
        return self._controllers[controller_type]


class Controller:
    """Base class for a coordinator controller.

    Subclasses override the handler-declaration methods to claim message types
    and override :meth:`post_init` to wire up shared collaborators. The default
    methods make every override optional, so this is a concrete base rather than
    an abstract one.
    """

    def push_handlers(self) -> dict[type[PushMsg], PushHandler]:
        """Declare handlers for fire-and-forget (PULL) messages.

        Returns:
            A mapping of concrete push-message type to its handler. Empty by
            default.
        """
        return {}

    def req_handlers(self) -> dict[type[ReqMsg], ReqHandler]:
        """Declare handlers for request/reply (ROUTER) messages.

        Returns:
            A mapping of concrete request-message type to its handler. Empty by
            default.
        """
        return {}

    def post_init(self, ctx: ControllerContext) -> None:
        """Receive shared collaborators after all controllers are constructed.

        Subclasses store what they need and subscribe to lifecycle hooks here.

        Args:
            ctx: The shared controller context.
        """
        return None

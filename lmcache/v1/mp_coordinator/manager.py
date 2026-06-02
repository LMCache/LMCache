# SPDX-License-Identifier: Apache-2.0
"""The mp coordinator manager: dispatch and health, over a transport.

The manager decodes inbound bytes into typed messages,
routes them by type to controller handlers (built into a dispatch table at
startup), and runs a health-check thread that evicts stale instances. All wire
I/O lives behind the injected :class:`CoordinatorTransport`.

Concurrency model:

- The transport runs the inbound loops on the single asyncio event loop.
- The health-check thread only *detects* stale instances; eviction is scheduled
  back onto the event loop via ``run_coroutine_threadsafe`` so the transport's
  connection lifecycle stays on one thread.
"""

# Standard
import asyncio
import threading

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.controllers.base import (
    Controller,
    ControllerContext,
    PushHandler,
    ReqHandler,
)
from lmcache.v1.mp_coordinator.controllers.registration import (
    RegistrationController,
)
from lmcache.v1.mp_coordinator.lifecycle import LifecycleHooks
from lmcache.v1.mp_coordinator.message import (
    CoordMsg,
    ErrorMsg,
    PushMsg,
    ReqMsg,
    ReqRetMsg,
)
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.mp_coordinator.transport import (
    CoordinatorTransport,
    Inbound,
    InboundKind,
)

logger = init_logger(__name__)


class MPCoordinatorManager:
    """Owns controller dispatch and health checks over a transport.

    Args:
        config: The coordinator configuration (timeouts; addresses live in the
            transport).
        transport: The wire transport the coordinator runs on.
    """

    def __init__(
        self, config: MPCoordinatorConfig, transport: CoordinatorTransport
    ) -> None:
        """Build controllers, dispatch tables, and shared state."""
        self.config = config
        self.transport = transport

        # Shared state and collaborators.
        self.registry = InstanceRegistry()
        self.lifecycle = LifecycleHooks()
        self._ctx = ControllerContext(
            registry=self.registry,
            transport=self.transport,
            lifecycle=self.lifecycle,
        )

        # Controllers. Future controllers are appended here only.
        self._registration = RegistrationController()
        self.controllers: list[Controller] = [self._registration]

        # Wire controllers: make each discoverable, then hand out the context.
        for controller in self.controllers:
            self._ctx.register_controller(controller)
        for controller in self.controllers:
            controller.post_init(self._ctx)

        # Build the dispatch tables from controller declarations.
        self._push_dispatch: dict[type[PushMsg], PushHandler] = {}
        self._req_dispatch: dict[type[ReqMsg], ReqHandler] = {}
        self._build_dispatch()

        # Health-check coordination. The loop reference is captured in
        # start_all() so the health thread can schedule work onto it.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown = threading.Event()
        self._health_thread = threading.Thread(
            target=self._health_loop,
            name="coordinator-health",
            daemon=True,
        )

    def _build_dispatch(self) -> None:
        """Merge every controller's handlers into the dispatch tables.

        Raises:
            ValueError: If two controllers claim the same message type.
        """
        for controller in self.controllers:
            for push_type, handler in controller.push_handlers().items():
                if push_type in self._push_dispatch:
                    raise ValueError(f"Duplicate push handler for {push_type.__name__}")
                self._push_dispatch[push_type] = handler
            for req_type, req_handler in controller.req_handlers().items():
                if req_type in self._req_dispatch:
                    raise ValueError(
                        f"Duplicate request handler for {req_type.__name__}"
                    )
                self._req_dispatch[req_type] = req_handler

    async def dispatch(self, inbound: Inbound) -> bytes | None:
        """Decode an inbound message and route it to its handler.

        This is the callback handed to :meth:`CoordinatorTransport.serve`.

        Args:
            inbound: The raw inbound message from the transport.

        Returns:
            Encoded reply bytes for a ``REQUEST`` (an ``ErrorMsg`` if the
            payload cannot be decoded or has no handler), or ``None`` for a
            ``PUSH``.
        """
        try:
            msg = msgspec.msgpack.decode(inbound.payload, type=CoordMsg)
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            logger.error("Failed to decode %s message: %s", inbound.kind.value, e)
            if inbound.kind is InboundKind.REQUEST:
                return msgspec.msgpack.encode(ErrorMsg(error=str(e)))
            return None

        if inbound.kind is InboundKind.PUSH:
            if isinstance(msg, PushMsg):
                await self._dispatch_push(msg)
            else:
                logger.error("Non-push message %s on push channel", type(msg).__name__)
            return None

        if not isinstance(msg, ReqMsg):
            ret: ReqRetMsg = ErrorMsg(
                error=f"Non-request message {type(msg).__name__} on request channel"
            )
        else:
            ret = await self._dispatch_req(msg)
        return msgspec.msgpack.encode(ret)

    async def _dispatch_push(self, msg: PushMsg) -> None:
        """Route a fire-and-forget message to its handler.

        Args:
            msg: The decoded push message.
        """
        handler = self._push_dispatch.get(type(msg))
        if handler is None:
            logger.error("No push handler for message type %s", type(msg).__name__)
            return
        await handler(msg)

    async def _dispatch_req(self, msg: ReqMsg) -> ReqRetMsg:
        """Route a request message to its handler and return the reply.

        Args:
            msg: The decoded request message.

        Returns:
            The handler's reply, or an :class:`ErrorMsg` if no handler is
            registered for the message type.
        """
        handler = self._req_dispatch.get(type(msg))
        if handler is None:
            logger.error("No request handler for message type %s", type(msg).__name__)
            return ErrorMsg(error=f"No handler for {type(msg).__name__}")
        return await handler(msg)

    def _health_loop(self) -> None:
        """Evict stale instances on a timer (runs on the health thread).

        Detection happens here; eviction is scheduled back onto the event loop
        so the transport's connection lifecycle stays single-threaded.
        """
        while not self._shutdown.wait(self.config.health_check_interval):
            loop = self._loop
            if loop is None:
                continue
            for instance_id in self.registry.stale(self.config.instance_timeout):
                logger.warning("Instance %s timed out; evicting", instance_id)
                asyncio.run_coroutine_threadsafe(
                    self._registration.deregister_instance(instance_id), loop
                )

    async def start_all(self) -> None:
        """Run the coordinator until cancelled.

        Captures the running event loop for the health thread, starts the
        health thread, and serves the transport.
        """
        self._loop = asyncio.get_running_loop()
        if self.config.health_check_interval > 0:
            self._health_thread.start()
        logger.info(
            "MP coordinator listening: pull=%s reply=%s",
            self.config.pull_url,
            self.config.reply_url,
        )
        await self.transport.serve(self.dispatch)

    def close(self) -> None:
        """Stop the health thread and close the transport."""
        self._shutdown.set()
        if self._health_thread.is_alive():
            self._health_thread.join(timeout=5.0)
        self.transport.close()

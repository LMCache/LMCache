# SPDX-License-Identifier: Apache-2.0
"""The mp coordinator manager: transport, dispatch, and health.

The manager owns the three coordinator sockets (PULL for fire-and-forget
messages, a reply ROUTER for request/reply, a dedicated heartbeat ROUTER),
builds the dispatch table by merging every controller's handler declarations,
and runs a health-check thread that evicts stale instances.

Concurrency model:

- All sockets are created and used only on the single asyncio event loop, so no
  socket is touched from two threads.
- The health-check thread never touches sockets directly: it detects stale
  instances and schedules their eviction back onto the event loop via
  ``run_coroutine_threadsafe``.
"""

# Standard
import asyncio
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.command import CommandSender
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
from lmcache.v1.rpc_utils import close_zmq_socket, get_zmq_context, get_zmq_socket

logger = init_logger(__name__)


class MPCoordinatorManager:
    """Owns coordinator transport, controller dispatch, and health checks.

    Args:
        config: The coordinator configuration (socket addresses and timeouts).
    """

    def __init__(self, config: MPCoordinatorConfig) -> None:
        """Build sockets, controllers, dispatch tables, and shared state."""
        self.config = config
        self.zmq_context = get_zmq_context()

        self.pull_socket = get_zmq_socket(
            self.zmq_context,
            config.pull_url,
            protocol="tcp",
            role=zmq.PULL,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        self.reply_socket = get_zmq_socket(
            self.zmq_context,
            config.reply_url,
            protocol="tcp",
            role=zmq.ROUTER,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )
        self.heartbeat_socket = get_zmq_socket(
            self.zmq_context,
            config.heartbeat_url,
            protocol="tcp",
            role=zmq.ROUTER,  # type: ignore[attr-defined]
            bind_or_connect="bind",
        )

        # Shared state and collaborators.
        self.registry = InstanceRegistry()
        self.lifecycle = LifecycleHooks()
        self.command_sender = CommandSender(self.registry)
        self._ctx = ControllerContext(
            registry=self.registry,
            command_sender=self.command_sender,
            lifecycle=self.lifecycle,
            zmq_context=self.zmq_context,
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

    async def _handle_pull(self) -> None:
        """Receive and dispatch fire-and-forget messages from the PULL socket."""
        while True:
            parts = await self.pull_socket.recv_multipart()
            for part in parts:
                try:
                    msg = msgspec.msgpack.decode(part, type=CoordMsg)
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    logger.error("Failed to decode pull message: %s", e)
                    continue
                if isinstance(msg, PushMsg):
                    await self._dispatch_push(msg)
                else:
                    logger.error(
                        "Non-push message %s on PULL socket", type(msg).__name__
                    )

    async def _handle_router(self, socket: zmq.asyncio.Socket, label: str) -> None:
        """Receive request/reply traffic on a ROUTER socket and reply.

        Args:
            socket: The ROUTER socket to service.
            label: A short label used in log messages.
        """
        while True:
            frames = await socket.recv_multipart()
            if len(frames) < 3:
                logger.error(
                    "%s: invalid ROUTER frame count %d (expected >= 3)",
                    label,
                    len(frames),
                )
                continue
            identity = frames[0]
            payload = frames[2]
            try:
                msg = msgspec.msgpack.decode(payload, type=CoordMsg)
                if not isinstance(msg, ReqMsg):
                    ret: ReqRetMsg = ErrorMsg(
                        error=f"Non-request message {type(msg).__name__} on {label}"
                    )
                else:
                    ret = await self._dispatch_req(msg)
            except (msgspec.DecodeError, msgspec.ValidationError) as e:
                logger.error("%s: failed to decode request: %s", label, e)
                ret = ErrorMsg(error=str(e))
            await socket.send_multipart([identity, b"", msgspec.msgpack.encode(ret)])

    def _health_loop(self) -> None:
        """Evict stale instances on a timer (runs on the health thread).

        Detection happens here; the actual eviction (which closes a socket) is
        scheduled back onto the event loop so socket access stays
        single-threaded.
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
        health thread, and serves all three sockets concurrently.
        """
        self._loop = asyncio.get_running_loop()
        if self.config.health_check_interval > 0:
            self._health_thread.start()
        logger.info(
            "MP coordinator listening: pull=%s reply=%s heartbeat=%s",
            self.config.pull_url,
            self.config.reply_url,
            self.config.heartbeat_url,
        )
        await asyncio.gather(
            self._handle_pull(),
            self._handle_router(self.reply_socket, "reply"),
            self._handle_router(self.heartbeat_socket, "heartbeat"),
        )

    def close(self) -> None:
        """Stop the health thread and close all coordinator sockets."""
        self._shutdown.set()
        if self._health_thread.is_alive():
            self._health_thread.join(timeout=5.0)
        for node in self.registry.all_instances():
            removed = self.registry.deregister(node.instance_id)
            if removed is not None:
                close_zmq_socket(removed.command_socket)
        self.pull_socket.close()
        self.reply_socket.close()
        self.heartbeat_socket.close()

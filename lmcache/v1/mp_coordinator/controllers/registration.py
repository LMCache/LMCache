# SPDX-License-Identifier: Apache-2.0
"""Registration controller: fleet membership lifecycle.

This is the one concrete controller shipped with the backbone. It owns the
join / leave / heartbeat protocol, the per-instance command-socket lifecycle,
and the firing of lifecycle hooks. Every method that touches a command socket
runs on the coordinator event loop, satisfying the single-threaded socket
contract.
"""

# Standard
import time

# Third Party
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.command import open_command_socket
from lmcache.v1.mp_coordinator.controllers.base import (
    Controller,
    ControllerContext,
    PushHandler,
    ReqHandler,
)
from lmcache.v1.mp_coordinator.message import (
    DeregisterMsg,
    ErrorMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    PushMsg,
    RegisterMsg,
    RegisterRetMsg,
    ReqMsg,
    ReqRetMsg,
)
from lmcache.v1.mp_coordinator.registry import MPInstanceNode
from lmcache.v1.rpc_utils import close_zmq_socket

logger = init_logger(__name__)


class RegistrationController(Controller):
    """Handles mp-server registration, heartbeat, and deregistration."""

    def __init__(self) -> None:
        """Initialize the controller; collaborators arrive at ``post_init``."""
        self._ctx: ControllerContext | None = None

    def post_init(self, ctx: ControllerContext) -> None:
        """Store the shared controller context.

        Args:
            ctx: The shared controller context.
        """
        self._ctx = ctx

    @property
    def ctx(self) -> ControllerContext:
        """Return the controller context.

        Returns:
            The context set in :meth:`post_init`.

        Raises:
            RuntimeError: If accessed before ``post_init``.
        """
        if self._ctx is None:
            raise RuntimeError("RegistrationController used before post_init")
        return self._ctx

    def req_handlers(self) -> dict[type[ReqMsg], ReqHandler]:
        """Declare request handlers for register and heartbeat.

        Returns:
            A mapping of request message type to handler.
        """
        return {
            RegisterMsg: self._handle_register,
            HeartbeatMsg: self._handle_heartbeat,
        }

    def push_handlers(self) -> dict[type[PushMsg], PushHandler]:
        """Declare the push handler for deregistration.

        Returns:
            A mapping of push message type to handler.
        """
        return {DeregisterMsg: self._handle_deregister}

    async def _handle_register(self, msg: ReqMsg) -> ReqRetMsg:
        """Handle a registration request.

        Args:
            msg: The inbound request; must be a :class:`RegisterMsg`.

        Returns:
            A :class:`RegisterRetMsg` on success, or an :class:`ErrorMsg` if
            the connect-back command socket could not be opened.
        """
        if not isinstance(msg, RegisterMsg):
            return ErrorMsg(error=f"Expected RegisterMsg, got {type(msg).__name__}")
        return await self.register_instance(msg)

    async def _handle_heartbeat(self, msg: ReqMsg) -> ReqRetMsg:
        """Handle a heartbeat request.

        Updates the instance's heartbeat timestamp, or transparently
        re-registers the instance if the coordinator no longer knows it.

        Args:
            msg: The inbound request; must be a :class:`HeartbeatMsg`.

        Returns:
            A :class:`HeartbeatRetMsg`, or an :class:`ErrorMsg` on bad input.
        """
        if not isinstance(msg, HeartbeatMsg):
            return ErrorMsg(error=f"Expected HeartbeatMsg, got {type(msg).__name__}")

        updated = self.ctx.registry.update_heartbeat(msg.instance_id, time.monotonic())
        if updated:
            return HeartbeatRetMsg(re_registered=False)

        logger.warning(
            "Heartbeat from unknown instance %s; re-registering", msg.instance_id
        )
        register_msg = RegisterMsg(
            instance_id=msg.instance_id,
            ip=msg.ip,
            control_port=msg.control_port,
            metadata=msg.metadata,
        )
        ret = await self.register_instance(register_msg)
        if isinstance(ret, ErrorMsg):
            return ret
        return HeartbeatRetMsg(re_registered=True)

    async def _handle_deregister(self, msg: PushMsg) -> None:
        """Handle a deregistration notice.

        Args:
            msg: The inbound push message; must be a :class:`DeregisterMsg`.
        """
        if not isinstance(msg, DeregisterMsg):
            logger.error("Expected DeregisterMsg, got %s", type(msg).__name__)
            return
        await self.deregister_instance(msg.instance_id)

    async def register_instance(self, msg: RegisterMsg) -> ReqRetMsg:
        """Register (or re-register) an mp server.

        Opens a REQ command socket connected back to the mp server's control
        REP socket, stores the instance in the registry, and fires the
        ``on_join`` lifecycle hook. Re-registering a known instance closes the
        stale command socket first and fires ``on_join`` again so subscribers
        can re-broadcast any per-instance state.

        Args:
            msg: The registration request.

        Returns:
            A :class:`RegisterRetMsg` on success, or an :class:`ErrorMsg` if
            the command socket could not be opened.
        """
        existing = self.ctx.registry.get(msg.instance_id)
        if existing is not None:
            logger.info("Instance %s already registered; replacing", msg.instance_id)
            close_zmq_socket(existing.command_socket)

        control_addr = f"{msg.ip}:{msg.control_port}"
        try:
            command_socket = open_command_socket(
                self.ctx.zmq_context, msg.ip, msg.control_port
            )
        except zmq.ZMQError as e:
            logger.error("Failed to open command socket to %s: %s", control_addr, e)
            return ErrorMsg(error=f"Cannot connect to {control_addr}: {e}")

        node = MPInstanceNode(
            instance_id=msg.instance_id,
            ip=msg.ip,
            control_port=msg.control_port,
            command_socket=command_socket,
            registration_time=time.time(),
            last_heartbeat_time=time.monotonic(),
            metadata=dict(msg.metadata),
        )
        self.ctx.registry.register(node)
        logger.info("Registered instance %s at %s", msg.instance_id, control_addr)

        self.ctx.lifecycle.fire_join(msg.instance_id)
        return RegisterRetMsg()

    async def deregister_instance(self, instance_id: str) -> None:
        """Remove an mp server, close its command socket, fire ``on_leave``.

        Safe to call for an unknown instance (logged and ignored). Used both by
        the deregister push handler and by the manager's health-check eviction;
        callers must invoke it on the event loop thread so the socket close is
        single-threaded.

        Args:
            instance_id: Identifier of the instance to remove.
        """
        node = self.ctx.registry.deregister(instance_id)
        if node is None:
            logger.warning("Deregister for unknown instance %s", instance_id)
            return

        close_zmq_socket(node.command_socket)
        logger.info("Deregistered instance %s", instance_id)
        self.ctx.lifecycle.fire_leave(instance_id)

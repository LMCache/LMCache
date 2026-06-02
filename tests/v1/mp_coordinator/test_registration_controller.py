# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RegistrationController and the lifecycle wiring.

Tests run each async flow via ``asyncio.run`` with a fresh ZMQ context, so they
do not depend on the pytest-asyncio plugin. The connect-back command socket is
opened against an unused loopback port; ZMQ ``connect`` succeeds lazily without
a listener, which is enough to exercise the registration path.
"""

# Standard
import asyncio

# Third Party
import zmq.asyncio

# First Party
from lmcache.v1.mp_coordinator.command import CommandSender
from lmcache.v1.mp_coordinator.controllers.base import ControllerContext
from lmcache.v1.mp_coordinator.controllers.registration import (
    RegistrationController,
)
from lmcache.v1.mp_coordinator.lifecycle import LifecycleHooks
from lmcache.v1.mp_coordinator.message import (
    ErrorMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    RegisterMsg,
    RegisterRetMsg,
)
from lmcache.v1.mp_coordinator.registry import InstanceRegistry


def _register_msg(instance_id: str = "a", control_port: int = 5999) -> RegisterMsg:
    return RegisterMsg(
        instance_id=instance_id,
        ip="127.0.0.1",
        control_port=control_port,
    )


def _build():
    """Build a registration controller with a fresh context.

    Returns:
        A tuple of (zmq_context, registry, lifecycle, controller, joins, leaves)
        where joins/leaves are lists recording fired lifecycle events.
    """
    zmq_context = zmq.asyncio.Context()
    registry = InstanceRegistry()
    lifecycle = LifecycleHooks()
    joins: list[str] = []
    leaves: list[str] = []
    lifecycle.on_join(joins.append)
    lifecycle.on_leave(leaves.append)
    ctx = ControllerContext(
        registry=registry,
        command_sender=CommandSender(registry, zmq_context),
        lifecycle=lifecycle,
        zmq_context=zmq_context,
    )
    controller = RegistrationController()
    ctx.register_controller(controller)
    controller.post_init(ctx)
    return zmq_context, registry, lifecycle, controller, joins, leaves


def test_register_adds_instance_and_fires_join():
    async def run():
        zmq_context, registry, _, controller, joins, _ = _build()
        try:
            ret = await controller.register_instance(_register_msg("a"))
            assert isinstance(ret, RegisterRetMsg)
            assert registry.contains("a")
            assert joins == ["a"]
        finally:
            zmq_context.destroy(linger=0)

    asyncio.run(run())


def test_deregister_removes_instance_and_fires_leave():
    async def run():
        zmq_context, registry, _, controller, _, leaves = _build()
        try:
            await controller.register_instance(_register_msg("a"))
            await controller.deregister_instance("a")
            assert not registry.contains("a")
            assert leaves == ["a"]
        finally:
            zmq_context.destroy(linger=0)

    asyncio.run(run())


def test_heartbeat_known_instance():
    async def run():
        zmq_context, _, _, controller, _, _ = _build()
        try:
            await controller.register_instance(_register_msg("a"))
            ret = await controller._handle_heartbeat(
                HeartbeatMsg(
                    instance_id="a",
                    ip="127.0.0.1",
                    control_port=5999,
                )
            )
            assert isinstance(ret, HeartbeatRetMsg)
            assert ret.re_registered is False
        finally:
            zmq_context.destroy(linger=0)

    asyncio.run(run())


def test_heartbeat_unknown_instance_reregisters():
    async def run():
        zmq_context, registry, _, controller, joins, _ = _build()
        try:
            ret = await controller._handle_heartbeat(
                HeartbeatMsg(
                    instance_id="ghost",
                    ip="127.0.0.1",
                    control_port=5998,
                )
            )
            assert isinstance(ret, HeartbeatRetMsg)
            assert ret.re_registered is True
            assert registry.contains("ghost")
            assert joins == ["ghost"]
        finally:
            zmq_context.destroy(linger=0)

    asyncio.run(run())


def test_register_wrong_message_type_returns_error():
    async def run():
        zmq_context, _, _, controller, _, _ = _build()
        try:
            ret = await controller._handle_register(
                HeartbeatMsg(
                    instance_id="a",
                    ip="127.0.0.1",
                    control_port=5999,
                )
            )
            assert isinstance(ret, ErrorMsg)
        finally:
            zmq_context.destroy(linger=0)

    asyncio.run(run())


def test_reregister_replaces_socket():
    async def run():
        zmq_context, registry, _, controller, joins, _ = _build()
        try:
            await controller.register_instance(_register_msg("a", control_port=5999))
            first_socket = registry.get("a").command_socket
            await controller.register_instance(_register_msg("a", control_port=6001))
            second_socket = registry.get("a").command_socket
            assert first_socket is not second_socket
            assert registry.get("a").control_port == 6001
            assert joins == ["a", "a"]
        finally:
            zmq_context.destroy(linger=0)

    asyncio.run(run())

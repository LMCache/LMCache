# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RegistrationController and the lifecycle wiring.

A fake transport stands in for the wire, so these tests are pure logic with no
sockets and run via ``asyncio.run`` (no pytest-asyncio plugin needed).
"""

# Standard
import asyncio

# First Party
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
from lmcache.v1.mp_coordinator.transport import (
    CoordinatorTransport,
    InboundHandler,
    ReachInfo,
    TransportError,
)


class _FakeTransport(CoordinatorTransport):
    """Records reach lifecycle; never opens a real socket.

    Set ``fail_add`` to make :meth:`add_instance` raise, simulating an
    unreachable mp server.
    """

    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []
        self.fail_add = False

    async def serve(self, handler: InboundHandler) -> None:  # pragma: no cover
        raise NotImplementedError

    async def send_command(self, instance_id: str, payload: bytes) -> bytes:
        raise NotImplementedError

    async def broadcast(self, payload: bytes) -> dict[str, bytes]:
        return {}

    def add_instance(self, instance_id: str, reach: ReachInfo) -> None:
        if self.fail_add:
            raise TransportError("unreachable")
        self.added.append(instance_id)

    def remove_instance(self, instance_id: str) -> None:
        self.removed.append(instance_id)

    def close(self) -> None:
        return None


def _register_msg(instance_id: str = "a", control_port: int = 5999) -> RegisterMsg:
    return RegisterMsg(
        instance_id=instance_id, ip="127.0.0.1", control_port=control_port
    )


def _build():
    """Build a registration controller over a fake transport.

    Returns:
        A tuple (registry, transport, controller, joins, leaves) where
        joins/leaves record fired lifecycle events.
    """
    registry = InstanceRegistry()
    lifecycle = LifecycleHooks()
    transport = _FakeTransport()
    joins: list[str] = []
    leaves: list[str] = []
    lifecycle.on_join(joins.append)
    lifecycle.on_leave(leaves.append)
    ctx = ControllerContext(registry=registry, transport=transport, lifecycle=lifecycle)
    controller = RegistrationController()
    ctx.register_controller(controller)
    controller.post_init(ctx)
    return registry, transport, controller, joins, leaves


def test_register_adds_instance_and_fires_join():
    registry, transport, controller, joins, _ = _build()
    ret = asyncio.run(controller.register_instance(_register_msg("a")))
    assert isinstance(ret, RegisterRetMsg)
    assert registry.contains("a")
    assert transport.added == ["a"]
    assert joins == ["a"]


def test_register_returns_error_when_unreachable():
    registry, transport, controller, joins, _ = _build()
    transport.fail_add = True
    ret = asyncio.run(controller.register_instance(_register_msg("a")))
    assert isinstance(ret, ErrorMsg)
    # Not registered and no join fired when the transport cannot reach it.
    assert not registry.contains("a")
    assert joins == []


def test_deregister_removes_instance_and_fires_leave():
    registry, transport, controller, _, leaves = _build()
    asyncio.run(controller.register_instance(_register_msg("a")))
    asyncio.run(controller.deregister_instance("a"))
    assert not registry.contains("a")
    assert transport.removed == ["a"]
    assert leaves == ["a"]


def _req_handler(controller: RegistrationController, msg_type: type):
    """Return the handler the controller declares for a request type.

    Goes through the public ``req_handlers()`` declaration rather than naming a
    private method, matching how the manager dispatches.
    """
    return controller.req_handlers()[msg_type]


def test_heartbeat_known_instance():
    _, _, controller, _, _ = _build()
    asyncio.run(controller.register_instance(_register_msg("a")))
    handler = _req_handler(controller, HeartbeatMsg)
    ret = asyncio.run(
        handler(HeartbeatMsg(instance_id="a", ip="127.0.0.1", control_port=5999))
    )
    assert isinstance(ret, HeartbeatRetMsg)
    assert ret.re_registered is False


def test_heartbeat_unknown_instance_reregisters():
    registry, _, controller, joins, _ = _build()
    handler = _req_handler(controller, HeartbeatMsg)
    ret = asyncio.run(
        handler(HeartbeatMsg(instance_id="ghost", ip="127.0.0.1", control_port=5998))
    )
    assert isinstance(ret, HeartbeatRetMsg)
    assert ret.re_registered is True
    assert registry.contains("ghost")
    assert joins == ["ghost"]


def test_register_wrong_message_type_returns_error():
    _, _, controller, _, _ = _build()
    # The register handler defends against a mismatched message type.
    handler = _req_handler(controller, RegisterMsg)
    ret = asyncio.run(
        handler(HeartbeatMsg(instance_id="a", ip="127.0.0.1", control_port=5999))
    )
    assert isinstance(ret, ErrorMsg)


def test_reregister_replaces_reach():
    registry, transport, controller, joins, _ = _build()
    asyncio.run(controller.register_instance(_register_msg("a", control_port=5999)))
    asyncio.run(controller.register_instance(_register_msg("a", control_port=6001)))
    assert registry.get("a").control_port == 6001
    assert transport.added == ["a", "a"]
    assert joins == ["a", "a"]

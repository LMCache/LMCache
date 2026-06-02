# SPDX-License-Identifier: Apache-2.0
"""Wire messages exchanged between mp servers and the mp coordinator.

Messages are msgspec tagged structs decoded into the :data:`CoordMsg` union.
They are split by transport channel so the manager can route by socket rather
than by guessing intent:

- :class:`PushMsg`     -- fire-and-forget, arrives on the coordinator PULL socket.
- :class:`ReqMsg`      -- request that requires a reply, arrives on a ROUTER socket.
- :class:`ReqRetMsg`   -- reply payload sent back over the ROUTER socket.

This module carries lifecycle messages only (register / deregister / heartbeat).
Domain messages (quota, routing, KV ops) are introduced by their own follow-up
controllers.
"""

# Standard
from typing import Union

# Third Party
import msgspec


class CoordMsgBase(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    """Base class for every coordinator message.

    Returns:
        Not applicable; subclasses define fields.
    """

    def describe(self) -> str:
        """Return a short human-readable description for logging.

        Returns:
            A description string; empty for the base class.
        """
        return ""


class PushMsg(CoordMsgBase):
    """Base for fire-and-forget messages (mp server -> coordinator PULL socket)."""


class ReqMsg(CoordMsgBase):
    """Base for request messages requiring a reply (mp server -> ROUTER socket)."""


class ReqRetMsg(CoordMsgBase):
    """Base for reply messages (coordinator -> mp server over ROUTER socket)."""


class RegisterMsg(ReqMsg):
    """Request from an mp server to join the coordinator.

    Attributes:
        instance_id: Globally unique identifier of the mp server.
        ip: IP address the mp server is reachable at.
        control_port: Port of the mp server's control REP socket that the
            coordinator connects back to for pushing commands.
        metadata: Free-form string key/value pairs for future use.
    """

    instance_id: str
    ip: str
    control_port: int
    metadata: dict[str, str] = msgspec.field(default_factory=dict)

    def describe(self) -> str:
        """Return a description of the registration request.

        Returns:
            A description naming the instance and control address.
        """
        return f"Register instance {self.instance_id} at {self.ip}:{self.control_port}"


class RegisterRetMsg(ReqRetMsg):
    """Reply to a :class:`RegisterMsg`.

    Attributes:
        extra_config: Configuration the coordinator hands back to the mp
            server at join time (e.g. a dedicated heartbeat address). Empty
            when there is nothing extra to send.
    """

    extra_config: dict[str, str] = msgspec.field(default_factory=dict)

    def describe(self) -> str:
        """Return a description of the registration reply.

        Returns:
            A description including the returned extra config.
        """
        return f"RegisterRet extra_config={self.extra_config}"


class DeregisterMsg(PushMsg):
    """Notification that an mp server is leaving the coordinator.

    Attributes:
        instance_id: Identifier of the departing mp server.
    """

    instance_id: str

    def describe(self) -> str:
        """Return a description of the deregistration notice.

        Returns:
            A description naming the departing instance.
        """
        return f"Deregister instance {self.instance_id}"


class HeartbeatMsg(ReqMsg):
    """Periodic liveness signal from an mp server.

    Carries the full registration payload so the coordinator can transparently
    re-register an instance it has forgotten (e.g. after a coordinator restart).

    Attributes:
        instance_id: Identifier of the mp server.
        ip: IP address the mp server is reachable at.
        control_port: Control REP port for the connect-back command socket.
        metadata: Free-form string key/value pairs for future use.
    """

    instance_id: str
    ip: str
    control_port: int
    metadata: dict[str, str] = msgspec.field(default_factory=dict)

    def describe(self) -> str:
        """Return a description of the heartbeat.

        Returns:
            A description naming the instance.
        """
        return f"Heartbeat from instance {self.instance_id}"


class HeartbeatRetMsg(ReqRetMsg):
    """Reply to a :class:`HeartbeatMsg`.

    Attributes:
        re_registered: ``True`` when the coordinator did not know this instance
            and re-registered it as part of handling the heartbeat.
    """

    re_registered: bool = False

    def describe(self) -> str:
        """Return a description of the heartbeat reply.

        Returns:
            A description including whether a re-registration occurred.
        """
        return f"HeartbeatRet re_registered={self.re_registered}"


class ErrorMsg(ReqRetMsg):
    """Error reply returned when a request cannot be handled.

    Attributes:
        error: Human-readable error description.
    """

    error: str

    def describe(self) -> str:
        """Return a description of the error.

        Returns:
            A description carrying the error string.
        """
        return f"Error: {self.error}"


# Tagged union used by msgspec to decode any inbound payload.
CoordMsg = Union[
    RegisterMsg,
    RegisterRetMsg,
    DeregisterMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    ErrorMsg,
]

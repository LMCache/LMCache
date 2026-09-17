# SPDX-License-Identifier: Apache-2.0
"""Protocol envelope for out-of-tree server-module extensions."""

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

REQUEST_NAMES = [
    "SERVER_MODULE_CALL",
]


class ServerModuleCallRequest(msgspec.Struct):
    """Opaque request envelope dispatched by namespaced extension method.

    Args:
        method: Namespaced extension method, such as
            ``"my_package.echo"``.
        payload: Plugin-owned serialized payload bytes.
    """

    method: str
    payload: bytes


class ServerModuleCallResponse(msgspec.Struct):
    """Opaque response envelope for extension method calls.

    Args:
        success: Whether the extension handler completed successfully.
        payload: Plugin-owned serialized response bytes.
        error: Human-readable error string when ``success`` is false.
    """

    success: bool
    payload: bytes
    error: str


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return protocol definitions for server-module extensions."""
    return {
        "SERVER_MODULE_CALL": ProtocolDefinition(
            payload_classes=[ServerModuleCallRequest],
            response_class=ServerModuleCallResponse,
            handler_type=HandlerType.BLOCKING,
        ),
    }

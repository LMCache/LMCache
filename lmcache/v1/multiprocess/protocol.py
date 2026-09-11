# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral multiprocess RPC names and Python message contracts."""

# Standard
# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.rpc_messages import (
    RPC_MESSAGE_TYPES,
    RpcRequest,
    RpcResponse,
)

_request_names = {request_type.name for request_type in RequestType}
_message_names = set(RPC_MESSAGE_TYPES)
if _request_names != _message_names:
    missing = sorted(_request_names - _message_names)
    unknown = sorted(_message_names - _request_names)
    raise RuntimeError(
        "Python RPC message registry does not match RequestType: "
        f"missing={missing}, unknown={unknown}"
    )

# Type aliases for backwards compatibility
InstanceID = int
KeyType = IPCCacheServerKey


def get_request_message_class(req_type: RequestType) -> type[RpcRequest]:
    """Return the transport-neutral request message class for an RPC."""
    try:
        return RPC_MESSAGE_TYPES[req_type.name][0]
    except KeyError as exc:
        raise ValueError(f"Invalid request type: {req_type}") from exc


def get_response_message_class(req_type: RequestType) -> type[RpcResponse]:
    """Return the transport-neutral response message class for an RPC."""
    try:
        return RPC_MESSAGE_TYPES[req_type.name][1]
    except KeyError as exc:
        raise ValueError(f"Invalid request type: {req_type}") from exc

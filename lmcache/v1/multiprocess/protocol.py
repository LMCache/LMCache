# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral multiprocess RPC names and Python message contracts."""

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.rpc_messages import (
    RpcRequest,
    RpcResponse,
)
from lmcache.v1.multiprocess.rpc_messages.registry import (
    get_request_message_class as _get_request_message_class,
)
from lmcache.v1.multiprocess.rpc_messages.registry import (
    get_response_message_class as _get_response_message_class,
)

# Type aliases for backwards compatibility
InstanceID = int
KeyType = IPCCacheServerKey


def get_request_message_class(req_type: RequestType) -> type[RpcRequest]:
    """Return the transport-neutral request message class for an RPC."""
    return _get_request_message_class(req_type)


def get_response_message_class(req_type: RequestType) -> type[RpcResponse]:
    """Return the transport-neutral response message class for an RPC."""
    return _get_response_message_class(req_type)

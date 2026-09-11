# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral multiprocess RPC routes and Python message contracts."""

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
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
RpcOperation = str


def get_request_message_class(operation: RpcOperation) -> type[RpcRequest]:
    """Return the transport-neutral request message class for an RPC route."""
    return _get_request_message_class(operation)


def get_response_message_class(operation: RpcOperation) -> type[RpcResponse]:
    """Return the transport-neutral response message class for an RPC route."""
    return _get_response_message_class(operation)

# SPDX-License-Identifier: Apache-2.0
"""Former import path of the server-module envelopes, kept for out-of-tree plugins."""

# First Party
from lmcache.v1.transport.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)

__all__ = ["ServerModuleCallRequest", "ServerModuleCallResponse"]

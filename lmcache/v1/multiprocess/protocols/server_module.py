# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for server-module extension wire contracts."""

# First Party
from lmcache.v1.transport.server_module import (
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)

__all__ = ["ServerModuleCallRequest", "ServerModuleCallResponse"]

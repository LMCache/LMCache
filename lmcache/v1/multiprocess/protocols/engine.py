# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for engine-driven business types.

RPC message contracts live in ``rpc_messages.py`` and handler scheduling lives
on ``@request_handler`` annotations.
"""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)

KeyType = IPCCacheServerKey

__all__ = [
    "KeyType",
    "PrepareRetrieveResponse",
    "PrepareStoreResponse",
    "RegisterEngineDrivenContextResponse",
]

# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports used by legacy MP integrations."""

# First Party
from lmcache.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    CustomizedSerdeConfig,
    DeviceIPCWrapper,
    IPCCacheServerKey,
    KVCache,
    RegisterEngineDrivenContextPayload,
    get_customized_decoder,
    get_customized_encoder,
)

__all__ = [
    "DeviceIPCWrapper",
    "IPCCacheServerKey",
    "KVCache",
    "RegisterEngineDrivenContextPayload",
    "CustomizedSerdeConfig",
    "BlockAllocationRecord",
    "CBMatchResult",
    "CBUnifiedLookupResult",
    "get_customized_encoder",
    "get_customized_decoder",
]

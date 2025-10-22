# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, Optional
import enum

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey, KVCache

"""
Main RPC protocol for the LMCache core server and clients. The following 
functions are supported:

- REGISTER_KV_CACHE:
    gpu_id: int
    kv_caches: KVCache

- UNREGISTER_KV_CACHE:
    gpu_id: int

- STORE: 
    keys: list[KeyType]
    gpu_id: int
    gpu_block_ids: list[int]

- RETRIEVE:
    keys: list[KeyType]
    gpu_id: int
    gpu_block_ids: list[int]
    enable_layerwise: Optional[bool]

- LOOKUP:
    keys: list[KeyType]
    lock: Optional[bool]
"""


class RequestType(enum.Enum):
    REGISTER_KV_CACHE = enum.auto()
    UNREGISTER_KV_CACHE = enum.auto()
    STORE = enum.auto()
    RETRIEVE = enum.auto()
    LOOKUP = enum.auto()

    # For debug, could be used as heartbeats
    NOOP = enum.auto()


@dataclass
class ProtocolDefinition:
    """
    Helper class for the protocol definition
    """

    payload_classes: list[Any]
    response_class: Optional[Any]


def get_payload_classes(req_type: RequestType) -> list[Any]:
    if pd := _PROTOCOL_DEFINTIONS.get(req_type, None):
        return pd.payload_classes
    else:
        raise ValueError(f"Invalid request type: {req_type}")


def get_response_class(req_type: RequestType) -> Optional[Any]:
    if pd := _PROTOCOL_DEFINTIONS.get(req_type, None):
        return pd.response_class
    else:
        raise ValueError(f"Invalid request type: {req_type}")


KeyType = IPCCacheEngineKey

_PROTOCOL_DEFINTIONS = {
    # Register KV Cache
    # - gpu_id: int
    # - kv_cache: KVCacheType
    # Returns: None
    RequestType.REGISTER_KV_CACHE: ProtocolDefinition(
        payload_classes=[int, KVCache],
        response_class=None,
    ),
    # Unregister KV Cache
    # - gpu_id: int
    # Returns: None
    RequestType.UNREGISTER_KV_CACHE: ProtocolDefinition(
        payload_classes=[int],
        response_class=None,
    ),
    # Store
    # - keys: list[KeyType]
    # - gpu_id: int
    # - gpu_block_ids: list[int]
    # Returns: bool (success)
    RequestType.STORE: ProtocolDefinition(
        payload_classes=[list[KeyType], int, list[int]],
        response_class=bool,
    ),
    # Retrieve
    # - keys: list[KeyType]
    # - gpu_id: int
    # - gpu_block_ids: list[int]
    # Returns: bool
    # NOTE: no layerwise support for now
    RequestType.RETRIEVE: ProtocolDefinition(
        payload_classes=[list[KeyType], int, list[int]],
        response_class=bool,
    ),
    # Lookup
    # - keys: list[KeyType]
    # - lock: Optional[bool]
    # Returns: list[bool] (found or not for each key)
    RequestType.LOOKUP: ProtocolDefinition(
        payload_classes=[list[KeyType], Optional[bool]],
        response_class=list[bool],
    ),
    # Debug commands
    RequestType.NOOP: ProtocolDefinition(
        payload_classes=[],
        response_class=str,
    ),
}

# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for the positional multiprocess client API."""

# Standard
from dataclasses import fields
from typing import Any

# First Party
from lmcache.v1.multiprocess.custom_types import (
    PrepareRetrieveResponse as LegacyPrepareRetrieveResponse,
)
from lmcache.v1.multiprocess.custom_types import (
    PrepareStoreResponse as LegacyPrepareStoreResponse,
)
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextResponse as LegacyRegisterContextResponse,
)
from lmcache.v1.multiprocess.rpc_messages.blend import CbRetrievePreComputedResponse
from lmcache.v1.multiprocess.rpc_messages.engine_driven import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterKvCacheEngineDrivenContextRequest,
    RegisterKvCacheEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.rpc_messages.lmcache_driven import (
    RetrieveResponse,
    StoreResponse,
)
from lmcache.v1.multiprocess.rpc_messages.registry import get_request_message_class


def make_request_message(request_name: str, *payloads: Any) -> Any:
    """Build one canonical request from the positional compatibility API."""
    request_class = get_request_message_class(request_name)
    if (
        request_class is RegisterKvCacheEngineDrivenContextRequest
        and len(payloads) == 1
        and isinstance(payloads[0], RegisterEngineDrivenContextPayload)
    ):
        payload = payloads[0]
        return request_class(
            **{name: getattr(payload, name) for name in payload.__struct_fields__}
        )
    return request_class(*payloads)


def unwrap_response_message(response: Any) -> Any:
    """Return the legacy caller result represented by a Python response."""
    if isinstance(
        response,
        (StoreResponse, RetrieveResponse, CbRetrievePreComputedResponse),
    ):
        return response.result.event_ipc_handle, response.result.success
    if isinstance(response, RegisterKvCacheEngineDrivenContextResponse):
        return LegacyRegisterContextResponse(response.shm_name, response.pool_size)
    if isinstance(response, PrepareStoreResponse):
        return LegacyPrepareStoreResponse(context=response.context)
    if isinstance(response, PrepareRetrieveResponse):
        return LegacyPrepareRetrieveResponse(
            success=response.success,
            data=response.data,
            context=response.context,
        )
    response_fields = fields(response)
    if not response_fields:
        return None
    values = tuple(getattr(response, item.name) for item in response_fields)
    return values[0] if len(values) == 1 else values


__all__ = ["make_request_message", "unwrap_response_message"]

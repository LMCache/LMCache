# SPDX-License-Identifier: Apache-2.0
"""Runtime Device-DAX L1 reconfiguration endpoints for MP mode.

See ``docs/source/mp/http_api.rst`` for the public HTTP contract and
``docs/design/v1/distributed/memory_manager/device-dax-l1.md`` for design details.
"""

# Standard
from dataclasses import asdict
from typing import Protocol, cast

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.memory_manager.reconfiguration import (
    L1ReconfigureError,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaStatus,
    DevDaxRemoveMode,
)
from lmcache.v1.multiprocess.http_apis.reconfigure_schemas import (
    SIZE_ERROR,
    SizeRequest,
    get_engine_from_request,
    reconfigure_error_response,
    resolve_size_bytes,
)

logger = init_logger(__name__)

router = APIRouter()


class _StorageManagerLike(Protocol):
    def get_l1_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]: ...

    def add_l1_devdax_device(
        self,
        device_path: str,
        size_in_bytes: int,
    ) -> DevDaxArenaStatus: ...

    def remove_l1_devdax_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode,
    ) -> DevDaxArenaStatus: ...


class L1DaxAddRequest(BaseModel):
    """Request body for ``POST /reconfigure/dax/l1/add``."""

    model_config = ConfigDict(extra="forbid")

    device_path: str
    size: SizeRequest


class L1DaxRemoveRequest(BaseModel):
    """Request body for ``POST /reconfigure/dax/l1/remove``."""

    model_config = ConfigDict(extra="forbid")

    device_path: str
    mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN


def _get_storage_manager(request: Request) -> _StorageManagerLike | JSONResponse:
    """Resolve the storage manager used by the L1 reconfigure routes.

    Args:
        request: Incoming request whose application state holds the engine.

    Returns:
        The storage manager, or a 503 response before engine initialization.
    """
    engine = get_engine_from_request(request)
    if engine is None:
        return JSONResponse(
            status_code=503, content={"error": "engine not initialized"}
        )
    storage_manager = getattr(engine, "storage_manager", None)
    if storage_manager is None:
        return JSONResponse(
            status_code=503, content={"error": "engine not initialized"}
        )
    return cast(_StorageManagerLike, storage_manager)


def _arena_dict(status: DevDaxArenaStatus) -> dict[str, object]:
    """Convert an arena status to a JSON-safe dictionary.

    Args:
        status: Arena status returned by the Device-DAX L1 manager.

    Returns:
        The arena fields with the lifecycle state represented as a string.
    """
    d = asdict(status)
    d["state"] = status.state.value
    return d


def _device_dict(
    device_path: str, arenas: list[DevDaxArenaStatus]
) -> dict[str, object]:
    """Build the device-level response envelope.

    Args:
        device_path: Path identifying the mapped Device-DAX device.
        arenas: Arena statuses associated with the device operation.

    Returns:
        A JSON-safe device object containing its arena statuses.
    """
    return {
        "device_path": device_path,
        "arenas": [_arena_dict(arena) for arena in arenas],
    }


@router.get("/reconfigure/dax/l1/status")
async def l1_dax_status(request: Request) -> JSONResponse:
    """Return Device-DAX L1 arena status.

    Args:
        request: Incoming request used to resolve the running engine.

    Returns:
        A response containing the arena statuses or a mapped error.
    """
    storage_manager = _get_storage_manager(request)
    if isinstance(storage_manager, JSONResponse):
        return storage_manager
    try:
        statuses = storage_manager.get_l1_devdax_arena_statuses()
    except L1ReconfigureError as exc:
        return reconfigure_error_response(exc)
    return JSONResponse(
        status_code=200,
        content={"arenas": [_arena_dict(status) for status in statuses]},
    )


@router.post("/reconfigure/dax/l1/add")
async def l1_dax_add(
    body: L1DaxAddRequest,
    request: Request,
) -> JSONResponse:
    """Add a Device-DAX device to the L1 arena pool.

    Args:
        body: Device path and mapped size requested by the client.
        request: Incoming request used to resolve the running engine.

    Returns:
        A response containing the added arena status or a mapped error.
    """
    storage_manager = _get_storage_manager(request)
    if isinstance(storage_manager, JSONResponse):
        return storage_manager
    try:
        size_bytes = resolve_size_bytes(body.size)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": SIZE_ERROR})
    try:
        status = storage_manager.add_l1_devdax_device(body.device_path, size_bytes)
    except L1ReconfigureError as exc:
        logger.warning("l1 dax add failed for %s: %s", body.device_path, exc)
        return reconfigure_error_response(exc)
    return JSONResponse(status_code=200, content={"added": _arena_dict(status)})


@router.post("/reconfigure/dax/l1/remove")
async def l1_dax_remove(
    body: L1DaxRemoveRequest,
    request: Request,
) -> JSONResponse:
    """Remove a Device-DAX device using the currently supported drain mode.

    The response envelope reports the arena's canonical path, which may differ
    from the alias used in the request.

    Args:
        body: Device path and removal mode requested by the client.
        request: Incoming request used to resolve the running engine.

    Returns:
        A response containing the removed arena status or a mapped error.
    """
    storage_manager = _get_storage_manager(request)
    if isinstance(storage_manager, JSONResponse):
        return storage_manager
    try:
        status = storage_manager.remove_l1_devdax_device(body.device_path, body.mode)
    except L1ReconfigureError as exc:
        logger.warning("l1 dax remove failed for %s: %s", body.device_path, exc)
        return reconfigure_error_response(exc)
    return JSONResponse(
        status_code=200,
        content={"removed": _device_dict(status.device_path, [status])},
    )

# SPDX-License-Identifier: Apache-2.0
"""Runtime Device-DAX hotplug endpoints for MP mode."""

# Standard
from decimal import Decimal
from typing import Literal, Protocol, cast

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

# First Party
from lmcache.v1.distributed.l2_adapters.hotplug import L2HotplugError

router = APIRouter()

_MAX_SIZE_STRING_LEN = 64
_SIZE_ERROR = "size must be a positive integer byte count or a string like '100GiB'"
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
}
SizeRequest = int | str


class _StorageManagerLike(Protocol):
    def dax_hotplug_status(self) -> dict: ...

    def dax_hotplug_add(
        self,
        adapter_index: int,
        device_path: str,
        size_bytes: int,
    ) -> dict: ...

    def dax_hotplug_remove(
        self,
        adapter_index: int,
        device_path: str,
        mode: str,
        force: bool,
    ) -> dict: ...

    def dax_hotplug_resize(
        self,
        adapter_index: int,
        device_path: str,
        size_bytes: int,
        mode: str,
        force: bool,
    ) -> dict: ...


class _EngineLike(Protocol):
    storage_manager: _StorageManagerLike


class DaxAddRequest(BaseModel):
    """Request body for ``POST /dax/add``."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    size: SizeRequest


class DaxRemoveRequest(BaseModel):
    """Request body for ``POST /dax/remove``."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    mode: Literal["migrate", "evict", "drain"] = "migrate"
    force: bool = False


class DaxResizeRequest(BaseModel):
    """Request body for ``POST /dax/resize``."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    size: SizeRequest
    mode: Literal["migrate", "evict"] = "migrate"
    force: bool = False


def _get_storage_manager(request: Request) -> _StorageManagerLike | JSONResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return cast(_EngineLike, engine).storage_manager


def _parse_size_string(size: str) -> int:
    text = size.strip()
    if not text or len(text) > _MAX_SIZE_STRING_LEN:
        raise ValueError(_SIZE_ERROR)

    unit_start = len(text)
    while unit_start > 0 and text[unit_start - 1].isalpha():
        unit_start -= 1

    value_text = text[:unit_start].strip()
    unit = text[unit_start:].lower()
    if unit not in _SIZE_UNITS:
        raise ValueError(_SIZE_ERROR)
    if "." in value_text:
        whole, fraction = value_text.split(".", 1)
        if not whole or not fraction:
            raise ValueError(_SIZE_ERROR)
        if not whole.isdigit() or not fraction.isdigit():
            raise ValueError(_SIZE_ERROR)
    elif not value_text.isdigit():
        raise ValueError(_SIZE_ERROR)

    value = Decimal(value_text)
    if value <= 0:
        raise ValueError(_SIZE_ERROR)
    return int(value * _SIZE_UNITS[unit])


def _resolve_size_bytes(size: SizeRequest) -> int:
    if isinstance(size, bool):
        raise ValueError(_SIZE_ERROR)
    resolved = size if isinstance(size, int) else _parse_size_string(size)
    if resolved <= 0:
        raise ValueError(_SIZE_ERROR)
    return resolved


def _api_error_response(exc: L2HotplugError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


@router.get("/dax/status", response_model=None)
async def dax_status(request: Request) -> dict | JSONResponse:
    """Return runtime Device-DAX hotplug status."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        return sm.dax_hotplug_status()
    except L2HotplugError as exc:
        return _api_error_response(exc)


@router.post("/dax/add", response_model=None)
async def dax_add(body: DaxAddRequest, request: Request) -> dict | JSONResponse:
    """Add a DAX device to a hotplug-enabled MP DAX adapter."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        size_bytes = _resolve_size_bytes(body.size)
        return sm.dax_hotplug_add(
            body.adapter_index,
            body.device_path,
            size_bytes,
        )
    except ValueError:
        return JSONResponse(status_code=400, content={"error": _SIZE_ERROR})
    except L2HotplugError as exc:
        return _api_error_response(exc)


@router.post("/dax/remove", response_model=None)
async def dax_remove(
    body: DaxRemoveRequest,
    request: Request,
) -> dict | JSONResponse:
    """Remove, evict, or drain a DAX device from an MP DAX adapter."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        return sm.dax_hotplug_remove(
            body.adapter_index,
            body.device_path,
            body.mode,
            body.force,
        )
    except L2HotplugError as exc:
        return _api_error_response(exc)


@router.post("/dax/resize", response_model=None)
async def dax_resize(
    body: DaxResizeRequest,
    request: Request,
) -> dict | JSONResponse:
    """Resize a DAX device in a hotplug-enabled MP DAX adapter."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        size_bytes = _resolve_size_bytes(body.size)
        return sm.dax_hotplug_resize(
            body.adapter_index,
            body.device_path,
            size_bytes,
            body.mode,
            body.force,
        )
    except ValueError:
        return JSONResponse(status_code=400, content={"error": _SIZE_ERROR})
    except L2HotplugError as exc:
        return _api_error_response(exc)

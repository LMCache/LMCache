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
from lmcache.v1.distributed.l2_adapters.reconfiguration import L2ReconfigureError

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
    def get_l2_adapter_reconfigure_status(self) -> dict: ...

    def reconfigure_l2_adapter(
        self,
        adapter_index: int,
        operation: str,
        payload: dict[str, object],
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


def _api_error_response(exc: L2ReconfigureError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


def _dax_adapter_entries(status: dict) -> list[tuple[int, dict]]:
    raw_adapters = status.get("adapters", [])
    if not isinstance(raw_adapters, list):
        return []

    dax_adapters = []
    for raw_index, adapter in enumerate(raw_adapters):
        if not isinstance(adapter, dict) or adapter.get("type") != "dax":
            continue
        generic_index = adapter.get("adapter_index", raw_index)
        if not isinstance(generic_index, int):
            generic_index = raw_index
        dax_adapters.append((generic_index, adapter))
    return dax_adapters


def _dax_status_response(status: dict) -> dict:
    adapters = []
    for dax_index, (_, adapter) in enumerate(_dax_adapter_entries(status)):
        public_adapter = dict(adapter)
        public_adapter["adapter_index"] = dax_index
        adapters.append(public_adapter)
    return {
        "enabled": bool(adapters),
        "hotplug_enabled": any(
            bool(adapter.get("hotplug_enabled", False)) for adapter in adapters
        ),
        "num_dax_adapters": len(adapters),
        "adapters": adapters,
    }


def _resolve_dax_adapter_index(
    sm: _StorageManagerLike,
    adapter_index: int,
) -> int:
    adapters = _dax_adapter_entries(sm.get_l2_adapter_reconfigure_status())
    if adapter_index < 0 or adapter_index >= len(adapters):
        raise L2ReconfigureError(404, "DAX adapter not found")
    generic_index, _ = adapters[adapter_index]
    return generic_index


@router.get("/dax/status", response_model=None)
async def dax_status(request: Request) -> dict | JSONResponse:
    """Return runtime Device-DAX hotplug status."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        return _dax_status_response(sm.get_l2_adapter_reconfigure_status())
    except L2ReconfigureError as exc:
        return _api_error_response(exc)


@router.post("/dax/add", response_model=None)
async def dax_add(body: DaxAddRequest, request: Request) -> dict | JSONResponse:
    """Add a DAX device to a hotplug-enabled MP DAX adapter."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        size_bytes = _resolve_size_bytes(body.size)
        adapter_index = _resolve_dax_adapter_index(sm, body.adapter_index)
        return sm.reconfigure_l2_adapter(
            adapter_index,
            "add",
            {
                "device_path": body.device_path,
                "size_bytes": size_bytes,
            },
        )
    except ValueError:
        return JSONResponse(status_code=400, content={"error": _SIZE_ERROR})
    except L2ReconfigureError as exc:
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
        adapter_index = _resolve_dax_adapter_index(sm, body.adapter_index)
        return sm.reconfigure_l2_adapter(
            adapter_index,
            "remove",
            {
                "device_path": body.device_path,
                "mode": body.mode,
                "force": body.force,
            },
        )
    except L2ReconfigureError as exc:
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
        adapter_index = _resolve_dax_adapter_index(sm, body.adapter_index)
        return sm.reconfigure_l2_adapter(
            adapter_index,
            "resize",
            {
                "device_path": body.device_path,
                "size_bytes": size_bytes,
                "mode": body.mode,
                "force": body.force,
            },
        )
    except ValueError:
        return JSONResponse(status_code=400, content={"error": _SIZE_ERROR})
    except L2ReconfigureError as exc:
        return _api_error_response(exc)

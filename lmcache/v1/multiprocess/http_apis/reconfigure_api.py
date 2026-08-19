# SPDX-License-Identifier: Apache-2.0
"""Runtime L2 adapter reconfiguration endpoints for MP mode."""

# Standard
from typing import Literal, Protocol, cast

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

# First Party
from lmcache.v1.distributed.l2_adapters.reconfiguration import L2ReconfigureError
from lmcache.v1.multiprocess.http_apis.reconfigure_schemas import (
    SIZE_ERROR,
    SizeRequest,
    get_engine_from_request,
    reconfigure_error_response,
    resolve_size_bytes,
    validation_error_response,
)

router = APIRouter()


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


class GenericReconfigureRequest(BaseModel):
    """Request body for generic ``POST /reconfigure/{backend}/{operation}``."""

    model_config = ConfigDict(extra="allow")

    adapter_index: int = 0


class DaxAddRequest(BaseModel):
    """Request body for ``POST /reconfigure/dax/add``."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    size: SizeRequest


class DaxRemoveRequest(BaseModel):
    """Request body for ``POST /reconfigure/dax/remove``."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    mode: Literal["migrate", "evict", "drain"] = "migrate"
    force: bool = False


class DaxResizeRequest(BaseModel):
    """Request body for ``POST /reconfigure/dax/resize``."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    size: SizeRequest
    mode: Literal["migrate", "evict"] = "migrate"
    force: bool = False


def _get_storage_manager(request: Request) -> _StorageManagerLike | JSONResponse:
    engine = get_engine_from_request(request)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return cast(_EngineLike, engine).storage_manager


def _normalize_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    if not normalized:
        raise L2ReconfigureError(400, "backend must be non-empty")
    return normalized


def _normalize_operation(operation: str) -> str:
    normalized = operation.strip().lower()
    if not normalized:
        raise L2ReconfigureError(400, "operation must be non-empty")
    return normalized


def _adapter_backend_name(adapter: dict) -> str | None:
    backend = adapter.get("backend", adapter.get("type"))
    if isinstance(backend, str) and backend:
        return backend
    return None


def _backend_adapter_entries(status: dict, backend: str) -> list[tuple[int, dict]]:
    raw_adapters = status.get("adapters", [])
    if not isinstance(raw_adapters, list):
        return []

    backend_adapters = []
    for raw_index, adapter in enumerate(raw_adapters):
        if not isinstance(adapter, dict) or _adapter_backend_name(adapter) != backend:
            continue
        generic_index = adapter.get("adapter_index", raw_index)
        if not isinstance(generic_index, int):
            generic_index = raw_index
        backend_adapters.append((generic_index, adapter))
    return backend_adapters


def _backend_status_response(status: dict, backend: str) -> dict:
    adapters = []
    for backend_index, (_, adapter) in enumerate(
        _backend_adapter_entries(status, backend)
    ):
        public_adapter = dict(adapter)
        public_adapter["adapter_index"] = backend_index
        adapters.append(public_adapter)
    return {
        "enabled": bool(adapters),
        "backend": backend,
        "num_adapters": len(adapters),
        "adapters": adapters,
    }


def _resolve_backend_adapter_index(
    sm: _StorageManagerLike,
    backend: str,
    adapter_index: int,
) -> int:
    adapters = _backend_adapter_entries(
        sm.get_l2_adapter_reconfigure_status(),
        backend,
    )
    if adapter_index < 0 or adapter_index >= len(adapters):
        raise L2ReconfigureError(404, f"{backend} adapter not found")
    generic_index, _ = adapters[adapter_index]
    return generic_index


def _dax_operation_payload(
    operation: str,
    payload: dict[str, object],
) -> tuple[int, dict[str, object]] | JSONResponse:
    try:
        if operation == "add":
            add_body = DaxAddRequest.model_validate(payload)
            try:
                size_bytes = resolve_size_bytes(add_body.size)
            except ValueError:
                return JSONResponse(status_code=400, content={"error": SIZE_ERROR})
            return (
                add_body.adapter_index,
                {
                    "device_path": add_body.device_path,
                    "size_bytes": size_bytes,
                },
            )

        if operation == "remove":
            remove_body = DaxRemoveRequest.model_validate(payload)
            return (
                remove_body.adapter_index,
                {
                    "device_path": remove_body.device_path,
                    "mode": remove_body.mode,
                    "force": remove_body.force,
                },
            )

        if operation == "resize":
            resize_body = DaxResizeRequest.model_validate(payload)
            try:
                size_bytes = resolve_size_bytes(resize_body.size)
            except ValueError:
                return JSONResponse(status_code=400, content={"error": SIZE_ERROR})
            return (
                resize_body.adapter_index,
                {
                    "device_path": resize_body.device_path,
                    "size_bytes": size_bytes,
                    "mode": resize_body.mode,
                    "force": resize_body.force,
                },
            )
    except ValidationError as exc:
        return validation_error_response(exc)

    raise L2ReconfigureError(
        400,
        f"unsupported dax reconfigure operation: {operation}",
    )


def _generic_operation_payload(
    payload: dict[str, object],
) -> tuple[int, dict[str, object]] | JSONResponse:
    try:
        body = GenericReconfigureRequest.model_validate(payload)
    except ValidationError as exc:
        return validation_error_response(exc)

    operation_payload = dict(body.model_extra or {})
    return body.adapter_index, operation_payload


def _operation_payload(
    backend: str,
    operation: str,
    payload: dict[str, object],
) -> tuple[int, dict[str, object]] | JSONResponse:
    if backend == "dax":
        return _dax_operation_payload(operation, payload)
    return _generic_operation_payload(payload)


@router.get("/reconfigure/{backend}/status", response_model=None)
async def reconfigure_status(backend: str, request: Request) -> dict | JSONResponse:
    """Return runtime reconfiguration status for one backend type."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        normalized_backend = _normalize_backend(backend)
        status = sm.get_l2_adapter_reconfigure_status()
        return _backend_status_response(status, normalized_backend)
    except L2ReconfigureError as exc:
        return reconfigure_error_response(exc)


@router.post("/reconfigure/{backend}/{operation}", response_model=None)
async def reconfigure_backend(
    backend: str,
    operation: str,
    payload: dict[str, object],
    request: Request,
) -> dict | JSONResponse:
    """Apply a runtime reconfiguration operation to one backend type."""
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm
    try:
        normalized_backend = _normalize_backend(backend)
        normalized_operation = _normalize_operation(operation)
        resolved = _operation_payload(normalized_backend, normalized_operation, payload)
        if isinstance(resolved, JSONResponse):
            return resolved
        adapter_index, operation_payload = resolved
        generic_adapter_index = _resolve_backend_adapter_index(
            sm,
            normalized_backend,
            adapter_index,
        )
        return sm.reconfigure_l2_adapter(
            generic_adapter_index,
            normalized_operation,
            operation_payload,
        )
    except L2ReconfigureError as exc:
        return reconfigure_error_response(exc)

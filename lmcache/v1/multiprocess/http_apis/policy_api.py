# SPDX-License-Identifier: Apache-2.0
"""HTTP API for node-local runtime management-policy updates."""

# Standard
from typing import Protocol, cast

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# First Party
from lmcache.v1.distributed.runtime_policy import (
    RuntimeL2EvictionUpdate,
    RuntimePolicyError,
    RuntimePolicyTunables,
    RuntimePolicyUpdate,
    RuntimePolicyUpdateResult,
    RuntimePolicyValidation,
)

router = APIRouter()

_RESTART_REQUIRED_FIELDS = frozenset(
    {
        "chunk_size",
        "l1_size",
        "l1_size_gb",
        "l1_memory",
        "memory_layout",
        "serde",
        "l2_adapter",
        "l2_adapter_type",
    }
)


class EvictionTunablesRequest(BaseModel):
    """Wire model for numeric eviction knobs."""

    model_config = ConfigDict(extra="forbid")

    trigger_watermark: float | None = Field(default=None, ge=0.0, le=1.0)
    eviction_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class EvictionUpdateRequest(BaseModel):
    """Wire model for one eviction policy plane."""

    model_config = ConfigDict(extra="forbid")

    policy: str | None = None
    tunables: EvictionTunablesRequest | None = None


class L2EvictionUpdateRequest(EvictionUpdateRequest):
    """Wire model for a stable-id addressed L2 eviction update."""

    adapter_id: int = Field(ge=0)


class RuntimePolicyUpdateRequest(BaseModel):
    """Wire model shared by ``validate`` and ``PATCH`` policy endpoints."""

    model_config = ConfigDict(extra="allow")

    expected_version: int | None = Field(default=None, ge=0)
    store_policy: str | None = None
    prefetch_policy: str | None = None
    l1_eviction: EvictionUpdateRequest | None = None
    l2_eviction: list[L2EvictionUpdateRequest] = Field(default_factory=list)


class _StorageManagerLike(Protocol):
    def get_runtime_policy(self) -> dict[str, object]: ...

    def validate_runtime_policy_update(
        self,
        update: RuntimePolicyUpdate,
    ) -> RuntimePolicyValidation: ...

    def update_runtime_policy(
        self,
        update: RuntimePolicyUpdate,
    ) -> RuntimePolicyUpdateResult: ...


class _EngineLike(Protocol):
    storage_manager: _StorageManagerLike


def _get_storage_manager(request: Request) -> _StorageManagerLike | JSONResponse:
    """Resolve the live StorageManager or return the standard 503 response."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return cast(_EngineLike, engine).storage_manager


def _to_tunables(
    request: EvictionUpdateRequest,
) -> RuntimePolicyTunables:
    """Convert one HTTP eviction model to a domain update."""
    tunables = request.tunables
    return RuntimePolicyTunables(
        policy=request.policy,
        trigger_watermark=(
            tunables.trigger_watermark if tunables is not None else None
        ),
        eviction_ratio=tunables.eviction_ratio if tunables is not None else None,
    )


def _to_runtime_update(body: RuntimePolicyUpdateRequest) -> RuntimePolicyUpdate:
    """Convert the validated HTTP body to a transport-independent update."""
    extras = tuple(body.model_extra or {})
    return RuntimePolicyUpdate(
        expected_version=body.expected_version,
        store_policy=body.store_policy,
        prefetch_policy=body.prefetch_policy,
        l1_eviction=(
            _to_tunables(body.l1_eviction) if body.l1_eviction is not None else None
        ),
        l2_eviction=tuple(
            RuntimeL2EvictionUpdate(
                adapter_id=entry.adapter_id,
                tunables=_to_tunables(entry),
            )
            for entry in body.l2_eviction
        ),
        restart_required_fields=tuple(
            sorted(field for field in extras if field in _RESTART_REQUIRED_FIELDS)
        ),
        unsupported_fields=tuple(
            sorted(field for field in extras if field not in _RESTART_REQUIRED_FIELDS)
        ),
    )


def _error_response(error: RuntimePolicyError) -> JSONResponse:
    """Convert a domain validation error to its stable HTTP response."""
    return JSONResponse(status_code=error.status_code, content=error.as_dict())


@router.get("/config/policies", response_model=None)
async def get_policies(request: Request) -> dict[str, object] | JSONResponse:
    """Return current runtime policy state and update capabilities."""
    storage_manager = _get_storage_manager(request)
    if isinstance(storage_manager, JSONResponse):
        return storage_manager
    return storage_manager.get_runtime_policy()


@router.post("/config/policies/validate", response_model=None)
async def validate_policies(
    body: RuntimePolicyUpdateRequest,
    request: Request,
) -> dict[str, object] | JSONResponse:
    """Validate a policy update without changing runtime state."""
    storage_manager = _get_storage_manager(request)
    if isinstance(storage_manager, JSONResponse):
        return storage_manager
    try:
        validation = storage_manager.validate_runtime_policy_update(
            _to_runtime_update(body)
        )
    except RuntimePolicyError as error:
        return _error_response(error)
    return {
        "status": "valid",
        "version": validation.version,
        "changed": list(validation.changed_fields),
        "effective_on": validation.effective_on,
    }


@router.patch("/config/policies", response_model=None)
async def update_policies(
    body: RuntimePolicyUpdateRequest,
    request: Request,
) -> dict[str, object] | JSONResponse:
    """Apply a fully validated node-local runtime policy update."""
    storage_manager = _get_storage_manager(request)
    if isinstance(storage_manager, JSONResponse):
        return storage_manager
    try:
        result = storage_manager.update_runtime_policy(_to_runtime_update(body))
    except RuntimePolicyError as error:
        return _error_response(error)
    return {
        "status": result.status,
        "version": result.version,
        "applied": list(result.applied),
        "effective_on": result.effective_on,
    }

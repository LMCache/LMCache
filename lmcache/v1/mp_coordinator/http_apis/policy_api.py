# SPDX-License-Identifier: Apache-2.0
"""Coordinator proxy and fleet fan-out for runtime management policies."""

# Standard
from typing import Annotated, Any
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import httpx

# First Party
from lmcache.v1.mp_coordinator.cache_control.policy_manager import (
    RuntimePolicyManager,
)
from lmcache.v1.mp_coordinator.http_apis.dependencies import (
    CoordinatorContext,
    get_context,
    get_outbound_client,
)
from lmcache.v1.mp_coordinator.registry import MPInstance
from lmcache.v1.multiprocess.http_apis.policy_api import RuntimePolicyUpdateRequest

router = APIRouter()
_policy_manager = RuntimePolicyManager()


class FleetRuntimePolicyUpdateRequest(BaseModel):
    """Body for coordinator fleet policy validation and updates.

    ``expected_versions`` is keyed by instance id because policy versions are
    node-local. When a key is omitted from a PATCH request, the coordinator
    uses the version observed during its all-target validation pass as the
    precondition for that node.
    """

    model_config = ConfigDict(extra="forbid")

    update: RuntimePolicyUpdateRequest
    expected_versions: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict
    )

    @field_validator("expected_versions")
    @classmethod
    def _validate_instance_ids(cls, value: dict[str, int]) -> dict[str, int]:
        """Reject blank instance ids in the optimistic version map."""
        if any(not instance_id.strip() for instance_id in value):
            raise ValueError("expected_versions contains a blank instance_id")
        return value


def _target_result(instance_id: str, status_code: int, body: Any) -> dict[str, Any]:
    """Build one stable per-instance fan-out result."""
    return {
        "instance_id": instance_id,
        "status_code": status_code,
        "body": body,
    }


def _error_result(instance_id: str, error: Exception) -> dict[str, Any]:
    """Represent an unreachable target without leaking an exception."""
    return _target_result(
        instance_id,
        502,
        {"error": "instance_unreachable", "detail": str(error)},
    )


def _all_succeeded(results: list[dict[str, Any]]) -> bool:
    """Return whether every target completed with a 2xx response."""
    return all(200 <= result["status_code"] < 300 for result in results)


def _aggregate_error_status(results: list[dict[str, Any]]) -> int:
    """Choose an actionable HTTP status for a failed fleet operation."""
    codes = [result["status_code"] for result in results]
    if 502 in codes:
        return 502
    if 409 in codes:
        return 409
    if 503 in codes:
        return 503
    if any(400 <= code < 500 for code in codes):
        return 400
    return 502


def _instances(
    ctx: CoordinatorContext,
    expected_versions: dict[str, int],
) -> list[MPInstance]:
    """Snapshot and validate the fleet targets for one request."""
    targets = sorted(ctx.registry.all_instances(), key=lambda item: item.instance_id)
    if not targets:
        raise HTTPException(status_code=404, detail="no MP servers are registered")
    unknown = sorted(set(expected_versions) - {item.instance_id for item in targets})
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"expected_versions contains unknown instance(s): {unknown}",
        )
    return targets


def _client_or_503(request: Request) -> httpx.AsyncClient:
    """Resolve the lifespan-bound client with a stable API error."""
    try:
        return get_outbound_client(request)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None


def _direct_body(body: RuntimePolicyUpdateRequest) -> dict[str, Any]:
    """Serialize a validated node-local policy request."""
    return body.model_dump(exclude_unset=True)


async def _proxy_direct(
    request: Request,
    instance_id: str,
    method: str,
    body: dict[str, Any] | None = None,
    endpoint: str = "/config/policies",
) -> JSONResponse:
    """Proxy one node-local request through the registry."""
    ctx = get_context(request)
    target = ctx.registry.get(instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={instance_id!r}",
        )
    try:
        status_code, payload = await _policy_manager.request(
            target, _client_or_503(request), method, body, endpoint
        )
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"policy request to {instance_id!r} failed: {error}",
        ) from None
    return JSONResponse(status_code=status_code, content=payload)


async def _validate_target(
    target: MPInstance,
    client: httpx.AsyncClient,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Validate one target and turn transport errors into result records."""
    try:
        status_code, payload = await _policy_manager.request(
            target, client, "POST", body, "/config/policies/validate"
        )
    except httpx.HTTPError as error:
        return _error_result(target.instance_id, error)
    return _target_result(target.instance_id, status_code, payload)


def _validation_response(results: list[dict[str, Any]]) -> JSONResponse:
    """Build the fleet validation response without applying any update."""
    succeeded = _all_succeeded(results)
    return JSONResponse(
        status_code=200 if succeeded else _aggregate_error_status(results),
        content={
            "phase": "validate",
            "status": "valid" if succeeded else "rejected",
            "results": results,
        },
    )


async def _validate_fleet(
    request: Request,
    body: FleetRuntimePolicyUpdateRequest,
) -> tuple[list[MPInstance], list[dict[str, Any]]]:
    """Run the all-target validation barrier used by validate and PATCH."""
    if body.update.expected_version is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "fleet requests must use expected_versions; "
                "update.expected_version is node-local"
            ),
        )
    ctx = get_context(request)
    targets = _instances(ctx, body.expected_versions)
    client = _client_or_503(request)
    update = body.update.model_dump(exclude_unset=True)

    async def _one(target: MPInstance) -> dict[str, Any]:
        target_update = dict(update)
        expected_version = body.expected_versions.get(target.instance_id)
        if expected_version is not None:
            target_update["expected_version"] = expected_version
        return await _validate_target(target, client, target_update)

    results = list(await asyncio.gather(*(_one(target) for target in targets)))
    return targets, results


async def _apply_target(
    target: MPInstance,
    client: httpx.AsyncClient,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Apply one target after the fleet validation barrier."""
    try:
        status_code, payload = await _policy_manager.request(
            target, client, "PATCH", body, "/config/policies"
        )
    except httpx.HTTPError as error:
        return _error_result(target.instance_id, error)
    return _target_result(target.instance_id, status_code, payload)


@router.get("/instances/{instance_id}/config/policies", response_model=None)
async def get_instance_policies(instance_id: str, request: Request) -> JSONResponse:
    """Proxy a node-local policy snapshot through the coordinator."""
    return await _proxy_direct(request, instance_id, "GET")


@router.post("/instances/{instance_id}/config/policies/validate", response_model=None)
async def validate_instance_policies(
    instance_id: str, body: RuntimePolicyUpdateRequest, request: Request
) -> JSONResponse:
    """Proxy node-local policy validation through the coordinator."""
    return await _proxy_direct(
        request,
        instance_id,
        "POST",
        _direct_body(body),
        "/config/policies/validate",
    )


@router.patch("/instances/{instance_id}/config/policies", response_model=None)
async def update_instance_policies(
    instance_id: str, body: RuntimePolicyUpdateRequest, request: Request
) -> JSONResponse:
    """Proxy a node-local policy update through the coordinator."""
    return await _proxy_direct(request, instance_id, "PATCH", _direct_body(body))


@router.post("/fleet/config/policies/validate", response_model=None)
async def validate_fleet_policies(
    body: FleetRuntimePolicyUpdateRequest, request: Request
) -> JSONResponse:
    """Validate one policy update against every registered MP server."""
    _, results = await _validate_fleet(request, body)
    return _validation_response(results)


@router.patch("/fleet/config/policies", response_model=None)
async def update_fleet_policies(
    body: FleetRuntimePolicyUpdateRequest, request: Request
) -> JSONResponse:
    """Validate all targets, then fan out a fenced update concurrently.

    The validation barrier prevents known-invalid updates from reaching any
    node. The apply phase is intentionally best-effort: a transport failure or
    version race after validation can leave a partial fleet update, which is
    reported per instance for the caller to reconcile.
    """
    targets, validation_results = await _validate_fleet(request, body)
    if not _all_succeeded(validation_results):
        return JSONResponse(
            status_code=_aggregate_error_status(validation_results),
            content={
                "phase": "validate",
                "status": "rejected",
                "results": validation_results,
            },
        )

    versions: dict[str, int] = {}
    for result in validation_results:
        payload = result["body"]
        version = payload.get("version") if isinstance(payload, dict) else None
        if not isinstance(version, int) or isinstance(version, bool):
            result["status_code"] = 502
            result["body"] = {
                "error": "invalid_validation_response",
                "detail": "successful validation response did not include version",
            }
            return JSONResponse(
                status_code=502,
                content={
                    "phase": "validate",
                    "status": "rejected",
                    "results": validation_results,
                },
            )
        versions[result["instance_id"]] = version

    client = _client_or_503(request)
    update = body.update.model_dump(exclude_unset=True)

    async def _one(target: MPInstance) -> dict[str, Any]:
        target_update = dict(update)
        target_update["expected_version"] = body.expected_versions.get(
            target.instance_id, versions[target.instance_id]
        )
        return await _apply_target(target, client, target_update)

    results = list(await asyncio.gather(*(_one(target) for target in targets)))
    succeeded = _all_succeeded(results)
    return JSONResponse(
        status_code=200 if succeeded else _aggregate_error_status(results),
        content={
            "phase": "apply",
            "status": "updated" if succeeded else "partial",
            "results": results,
        },
    )

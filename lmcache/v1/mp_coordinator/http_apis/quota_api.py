# SPDX-License-Identifier: Apache-2.0
"""Quota and usage-accounting endpoints on the coordinator (fleet-level).

The ``/quota`` surface, thin over the eviction manager on the typed
:class:`CoordinatorContext` (resolved via :func:`get_context`): it owns
both the quota registry these endpoints write and the usage view they
read, because enforcing one against the other is what it does. Usage
arrives through the fleet cache-event stream
(``POST /events``), admitted by the ingest gate.
This mirrors the MP server's node-local ``/quota`` group; warm-prefetch dispatch
is genuine cache control and lives in :mod:`cache_api` instead.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    QuotaConfigRequest,
    QuotaConfigResponse,
    QuotaResponse,
    SetQuotaRequest,
    StatusListResponse,
    StatusResponse,
)

router = APIRouter()

_GB = 1024**3
_DEFAULT_SALT_SENTINEL = "_default"
# Quota / usage / status are L2-tier accounting today; other tiers are rejected.
_SUPPORTED_TIER = Tier.L2


def _resolve_salt_from_api_path(cache_salt: str) -> str:
    """Map the ``_default`` sentinel to the empty string."""
    return "" if cache_salt == _DEFAULT_SALT_SENTINEL else cache_salt


def _require_supported_tier(tier: Tier) -> None:
    """Raise ``HTTPException(400)`` unless ``tier`` is the supported one (``l2``)."""
    if tier != _SUPPORTED_TIER:
        raise HTTPException(
            status_code=400,
            detail=f"tier {tier.value!r} not supported; only {_SUPPORTED_TIER.value!r}",
        )


def _gb(n_bytes: int) -> float:
    """Convert bytes to GiB."""
    return n_bytes / _GB


# -- Quota config  ---------------------------------------------


@router.put("/quota/config")
async def set_quota_config(
    body: QuotaConfigRequest, request: Request
) -> QuotaConfigResponse:
    """Set the default quota applied to salts with no explicit entry.

    Args:
        body: The default limit to apply (and the ``tier`` it applies to).

    Returns:
        The applied config.
    """
    _require_supported_tier(body.tier)
    default_limit_bytes = (
        None if body.default_limit_gb is None else int(body.default_limit_gb * _GB)
    )
    ctx = get_context(request)
    eviction = ctx.controllers.get(FleetEvictionController)
    eviction.quota.set_default_limit_bytes(default_limit_bytes)
    await ctx.metadata_persister.save()
    return QuotaConfigResponse(default_limit_gb=body.default_limit_gb)


@router.get("/quota/config")
async def get_quota_config(
    request: Request, tier: Tier = Tier.L2
) -> QuotaConfigResponse:
    """Read the default quota applied to salts with no explicit entry.

    Args:
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        The current config; ``default_limit_gb`` is ``null`` while
        unquota'd salts are exempt from eviction (the boot default).
    """
    _require_supported_tier(tier)
    quota = get_context(request).controllers.get(FleetEvictionController).quota
    default_limit = quota.get_default_limit_bytes()
    return QuotaConfigResponse(
        default_limit_gb=None if default_limit is None else _gb(default_limit)
    )


# -- Quota writes ------------------------------------------------------------


@router.put("/quota/{cache_salt}", response_model=None)
async def set_quota(
    cache_salt: str, body: SetQuotaRequest, request: Request
) -> QuotaResponse | JSONResponse:
    """Create or update a quota.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        body: Quota limit to apply (and the ``tier`` it applies to).

    Returns:
        The applied quota, or a 400 JSON response if the limit is invalid.
    """
    _require_supported_tier(body.tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    limit_bytes = int(body.limit_gb * _GB)
    ctx = get_context(request)
    eviction = ctx.controllers.get(FleetEvictionController)
    try:
        eviction.quota.set_quota(cache_salt, limit_bytes)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "invalid quota limit"})
    await ctx.metadata_persister.save()
    return QuotaResponse(cache_salt=cache_salt, limit_gb=body.limit_gb, status="ok")


@router.delete("/quota/{cache_salt}")
async def delete_quota(
    cache_salt: str, request: Request, tier: Tier = Tier.L2
) -> QuotaResponse:
    """Remove a salt's quota entry.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        ``QuotaResponse`` with ``status`` ``"removed"`` or ``"not_found"``.
    """
    _require_supported_tier(tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    ctx = get_context(request)
    eviction = ctx.controllers.get(FleetEvictionController)
    removed = eviction.quota.delete_quota(cache_salt)
    await ctx.metadata_persister.save()
    return QuotaResponse(
        cache_salt=cache_salt,
        limit_gb=0.0,
        status="removed" if removed else "not_found",
    )


# -- Combined status queries -------------------------------------------------


@router.get("/quota/{cache_salt}")
async def get_status(
    cache_salt: str, request: Request, tier: Tier = Tier.L2
) -> StatusResponse:
    """Read quota and usage for a single salt.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        Combined quota and live usage detail.
    """
    _require_supported_tier(tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    ctx = get_context(request)
    eviction = ctx.controllers.get(FleetEvictionController)
    quota = eviction.quota
    usage = eviction.usage.get(cache_salt)
    exists = quota.has_quota(cache_salt)
    limit = quota.get_limit_bytes(cache_salt)
    return StatusResponse(
        cache_salt=cache_salt,
        quota_limit_gb=_gb(limit) if exists else 0.0,
        quota_exists=exists,
        usage_gb=_gb(usage),
    )


@router.get("/quota")
async def list_status(request: Request, tier: Tier = Tier.L2) -> StatusListResponse:
    """List quota and usage across all cache salts.

    Args:
        tier: Cache tier (only ``l2`` is supported today).

    Returns:
        Total usage plus per-salt breakdown with quota info.
    """
    _require_supported_tier(tier)
    ctx = get_context(request)
    eviction = ctx.controllers.get(FleetEvictionController)
    usage_view = eviction.usage
    quota_registry = eviction.quota
    by_salt = usage_view.get_all()
    total = usage_view.get_total()
    quota_entries = {e.cache_salt: e.limit_bytes for e in quota_registry.list_quotas()}
    all_salts = sorted(set(by_salt) | set(quota_entries))
    return StatusListResponse(
        total_gb=_gb(total),
        by_cache_salt=[
            StatusResponse(
                cache_salt=salt,
                quota_limit_gb=_gb(quota_entries[salt])
                if salt in quota_entries
                else 0.0,
                quota_exists=salt in quota_entries,
                usage_gb=_gb(by_salt.get(salt, 0)),
            )
            for salt in all_salts
        ],
    )

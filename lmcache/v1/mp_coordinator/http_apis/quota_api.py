# SPDX-License-Identifier: Apache-2.0
"""Quota and usage-accounting endpoints on the coordinator (fleet-level).

The ``/quota`` surface, thin over the per-tier eviction controllers on
the typed :class:`CoordinatorContext` (resolved via :func:`get_context`):
each owns the quota registry these endpoints write for its tier, and
they share the usage view these endpoints read, because enforcing one
against the other is what they do. Usage arrives through the fleet
cache-event stream (``POST /events``), admitted by the ingest gate.

Every endpoint is **wholly scoped to one tier**. A key resident in both
tiers holds bytes in both and is budgeted separately in each, so there is
no cross-tier read and ``all`` is rejected.

This mirrors the MP server's node-local ``/quota`` group; warm-prefetch
dispatch is genuine cache control and lives in :mod:`cache_api` instead.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    ENFORCED_TIERS,
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.http_apis.dependencies import (
    CoordinatorContext,
    get_context,
)
from lmcache.v1.mp_coordinator.schemas import (
    QuotaConfigRequest,
    QuotaConfigResponse,
    QuotaResponse,
    SetQuotaRequest,
    StatusListResponse,
    StatusResponse,
)
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager

router = APIRouter()

_GB = 1024**3
_DEFAULT_SALT_SENTINEL = "_default"


def _resolve_salt_from_api_path(cache_salt: str) -> str:
    """Map the ``_default`` sentinel to the empty string."""
    return "" if cache_salt == _DEFAULT_SALT_SENTINEL else cache_salt


def _quota_for_tier(ctx: CoordinatorContext, tier: Tier) -> QuotaManager:
    """Return the quota registry that governs ``tier``.

    Args:
        ctx: The resolved coordinator context.
        tier: The tier the caller addressed.

    Returns:
        The registry the eviction controller enforces on that tier. Each
        tier has its own: the two govern different bytes, so one table's
        numbers are meaningless for the other.

    Raises:
        HTTPException: 400 if ``tier`` has no quota enforcement (``all``,
            which would conflate two budgets).
    """
    if tier not in ENFORCED_TIERS:
        supported = ", ".join(repr(t.value) for t in ENFORCED_TIERS)
        raise HTTPException(
            status_code=400,
            detail=f"tier {tier.value!r} has no quota accounting; only {supported}",
        )
    return ctx.controllers.get(FleetEvictionController).quota(tier)


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
        body: The default limit to apply, and the ``tier`` it applies to.

    Returns:
        The applied config.
    """
    ctx = get_context(request)
    quota = _quota_for_tier(ctx, body.tier)
    default_limit_bytes = (
        None if body.default_limit_gb is None else int(body.default_limit_gb * _GB)
    )
    quota.set_default_limit_bytes(default_limit_bytes)
    ctx.metadata_persister.save()
    return QuotaConfigResponse(default_limit_gb=body.default_limit_gb, tier=body.tier)


@router.get("/quota/config")
async def get_quota_config(
    request: Request, tier: Tier = Tier.L2
) -> QuotaConfigResponse:
    """Read the default quota applied to salts with no explicit entry.

    Args:
        tier: Cache tier to read (``l1`` or ``l2``).

    Returns:
        The current config for ``tier``; ``default_limit_gb`` is ``null``
        while its unquota'd salts are exempt from eviction (the boot
        default).
    """
    quota = _quota_for_tier(get_context(request), tier)
    default_limit = quota.get_default_limit_bytes()
    return QuotaConfigResponse(
        default_limit_gb=None if default_limit is None else _gb(default_limit),
        tier=tier,
    )


# -- Quota writes ------------------------------------------------------------


@router.put("/quota/{cache_salt}", response_model=None)
async def set_quota(
    cache_salt: str, body: SetQuotaRequest, request: Request
) -> QuotaResponse | JSONResponse:
    """Create or update a quota.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        body: Quota limit to apply, and the ``tier`` it applies to.

    Returns:
        The applied quota, or a 400 JSON response if the limit is invalid.
    """
    ctx = get_context(request)
    quota = _quota_for_tier(ctx, body.tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    limit_bytes = int(body.limit_gb * _GB)
    try:
        quota.set_quota(cache_salt, limit_bytes)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "invalid quota limit"})
    ctx.metadata_persister.save()
    return QuotaResponse(
        cache_salt=cache_salt, limit_gb=body.limit_gb, status="ok", tier=body.tier
    )


@router.delete("/quota/{cache_salt}")
async def delete_quota(
    cache_salt: str, request: Request, tier: Tier = Tier.L2
) -> QuotaResponse:
    """Remove a salt's quota entry.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        tier: Cache tier the entry belongs to (``l1`` or ``l2``).

    Returns:
        ``QuotaResponse`` with ``status`` ``"removed"`` or ``"not_found"``.
    """
    ctx = get_context(request)
    quota = _quota_for_tier(ctx, tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    removed = quota.delete_quota(cache_salt)
    ctx.metadata_persister.save()
    return QuotaResponse(
        cache_salt=cache_salt,
        limit_gb=0.0,
        status="removed" if removed else "not_found",
        tier=tier,
    )


# -- Combined status queries -------------------------------------------------


@router.get("/quota/{cache_salt}")
async def get_status(
    cache_salt: str, request: Request, tier: Tier = Tier.L2
) -> StatusResponse:
    """Read quota and usage for a single salt.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        tier: Cache tier to report (``l1`` or ``l2``). Both the usage and
            the quota fields describe that tier.

    Returns:
        Combined quota and live usage detail for ``tier``.
    """
    ctx = get_context(request)
    quota = _quota_for_tier(ctx, tier)
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    usage = ctx.views.get(CacheUsageManager).get_salt_bytes(tier, cache_salt)
    exists = quota.has_quota(cache_salt)
    limit = quota.get_limit_bytes(cache_salt) if exists else 0
    return StatusResponse(
        cache_salt=cache_salt,
        quota_limit_gb=_gb(limit),
        quota_exists=exists,
        usage_gb=_gb(usage),
        tier=tier,
    )


@router.get("/quota")
async def list_status(request: Request, tier: Tier = Tier.L2) -> StatusListResponse:
    """List quota and usage across all cache salts.

    Args:
        tier: Cache tier to report (``l1`` or ``l2``). Both the usage and
            the quota fields describe that tier.

    Returns:
        Total usage plus per-salt breakdown with quota info for ``tier``.
    """
    ctx = get_context(request)
    quota = _quota_for_tier(ctx, tier)
    usage_view = ctx.views.get(CacheUsageManager)
    by_salt = usage_view.get_bytes_by_salt(tier)
    total = usage_view.get_total_bytes(tier)
    quota_entries = {
        entry.cache_salt: entry.limit_bytes for entry in quota.list_quotas()
    }
    all_salts = sorted(set(by_salt) | set(quota_entries))
    return StatusListResponse(
        total_gb=_gb(total),
        tier=tier,
        by_cache_salt=[
            StatusResponse(
                cache_salt=salt,
                quota_limit_gb=_gb(quota_entries[salt])
                if salt in quota_entries
                else 0.0,
                quota_exists=salt in quota_entries,
                usage_gb=_gb(by_salt.get(salt, 0)),
                tier=tier,
            )
            for salt in all_salts
        ],
    )

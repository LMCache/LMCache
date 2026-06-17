# SPDX-License-Identifier: Apache-2.0
"""L2 cache management endpoints on the coordinator.

Quota writes (set/delete), usage event ingestion, and combined
status queries (quota + usage) for per-``cache_salt`` L2 data.
"""

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.l2.eviction_manager import (
    L2EvictionManager,
)
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.schemas import (
    EventType,
    L2StatusListResponse,
    L2StatusResponse,
    QuotaResponse,
    ReportUsageRequest,
    ReportUsageResponse,
    SetQuotaRequest,
)

router = APIRouter()

_GB = 1024**3
_DEFAULT_SALT_SENTINEL = "_default"


def _resolve_salt_from_api_path(cache_salt: str) -> str:
    """Map the ``_default`` sentinel to the empty string."""
    return "" if cache_salt == _DEFAULT_SALT_SENTINEL else cache_salt


def _gb(n_bytes: int) -> float:
    """Convert bytes to GiB."""
    return n_bytes / _GB


def _quota_manager(request: Request) -> QuotaManager:
    """Return the shared :class:`QuotaManager` from ``app.state``."""
    mgr = getattr(request.app.state, "quota_manager", None)
    if mgr is None:
        raise RuntimeError("quota manager not initialized")
    return mgr


def _usage_manager(request: Request) -> L2UsageManager:
    """Return the shared :class:`L2UsageManager` from ``app.state``."""
    mgr = getattr(request.app.state, "usage_manager", None)
    if mgr is None:
        raise RuntimeError("usage manager not initialized")
    return mgr


def _eviction_manager(request: Request) -> L2EvictionManager:
    """Return the shared :class:`L2EvictionManager` from ``app.state``."""
    mgr = getattr(request.app.state, "eviction_manager", None)
    if mgr is None:
        raise RuntimeError("eviction manager not initialized")
    return mgr


# -- Quota writes ------------------------------------------------------------


@router.put("/l2/quota/{cache_salt}", response_model=None)
async def set_quota(
    cache_salt: str, body: SetQuotaRequest, request: Request
) -> QuotaResponse | JSONResponse:
    """Create or update a quota.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.
        body: Quota limit to apply.

    Returns:
        The applied quota, or a 400 JSON response if the limit is invalid.
    """
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    limit_bytes = int(body.limit_gb * _GB)
    try:
        _quota_manager(request).set_quota(cache_salt, limit_bytes)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "invalid quota limit"})
    return QuotaResponse(
        cache_salt=cache_salt,
        limit_gb=body.limit_gb,
        status="ok",
    )


@router.delete("/l2/quota/{cache_salt}")
async def delete_quota(cache_salt: str, request: Request) -> QuotaResponse:
    """Remove a salt's quota entry.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.

    Returns:
        ``QuotaResponse`` with ``status`` ``"removed"`` or ``"not_found"``.
    """
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    removed = _quota_manager(request).delete_quota(cache_salt)
    return QuotaResponse(
        cache_salt=cache_salt,
        limit_gb=0.0,
        status="removed" if removed else "not_found",
    )


# -- event ingestion ---------------------------------------------------


@router.post("/l2/events")
async def report_events(
    body: ReportUsageRequest, request: Request
) -> ReportUsageResponse:
    """Record a batch of L2 ``store`` / ``lookup`` / ``delete`` events.

    Args:
        body: Event batch tagged with the reporter's ``instance_id`` and ``seq``.

    Returns:
        Number of events processed.
    """
    tracker = _usage_manager(request)
    ctrl = _eviction_manager(request)
    for event in body.events:
        ok = event.key.to_object_key()
        if event.type == EventType.STORE:
            tracker.record_stored(ok, event.bytes)
            ctrl.on_store(ok)
        elif event.type == EventType.LOOKUP:
            ctrl.on_lookup(ok)
        elif event.type == EventType.DELETE:
            tracker.record_evicted(ok)
            ctrl.on_remove(ok)
    return ReportUsageResponse(recorded=len(body.events))


# -- Combined status queries -------------------------------------------------


@router.get("/l2/status/{cache_salt}")
async def get_status(cache_salt: str, request: Request) -> L2StatusResponse:
    """Read quota and usage for a single salt.

    Args:
        cache_salt: Tenant identifier; use ``_default`` for the empty salt.

    Returns:
        Combined quota and live usage detail.
    """
    cache_salt = _resolve_salt_from_api_path(cache_salt)
    tracker = _usage_manager(request)
    store = _quota_manager(request)
    usage = tracker.get(cache_salt)
    exists = store.has_quota(cache_salt)
    limit = store.get_limit_bytes(cache_salt)
    return L2StatusResponse(
        cache_salt=cache_salt,
        quota_limit_gb=_gb(limit) if exists else 0.0,
        quota_exists=exists,
        usage_gb=_gb(usage),
    )


@router.get("/l2/status")
async def list_status(request: Request) -> L2StatusListResponse:
    """List quota and usage across all cache salts.

    Returns:
        Total usage plus per-salt breakdown with quota info.
    """
    tracker = _usage_manager(request)
    store = _quota_manager(request)
    by_salt = tracker.get_all()
    total = tracker.get_total()
    quota_entries = {e.cache_salt: e.limit_bytes for e in store.list_quotas()}
    all_salts = sorted(set(by_salt) | set(quota_entries))
    return L2StatusListResponse(
        total_gb=_gb(total),
        by_cache_salt=[
            L2StatusResponse(
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

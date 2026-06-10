# SPDX-License-Identifier: Apache-2.0
"""L2 cache management endpoints on the coordinator.

Quota writes (set/delete), usage event ingestion, and combined
status queries (quota + usage) for per-``cache_salt`` L2 data.
"""

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.mp_coordinator.l2.eviction_controller import (
    CoordinatorEvictionController,
)
from lmcache.v1.mp_coordinator.l2.quota_store import QuotaStore
from lmcache.v1.mp_coordinator.l2.usage_tracker import UsageTracker
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


def _gb(n_bytes: int) -> float:
    """Convert bytes to GiB."""
    return n_bytes / _GB


def _quota_store(request: Request) -> QuotaStore:
    """Return the shared quota store from app state.

    Args:
        request: The incoming request.

    Returns:
        The shared :class:`QuotaStore`.

    Raises:
        RuntimeError: If the store is not initialized.
    """
    store = getattr(request.app.state, "quota_store", None)
    if store is None:
        raise RuntimeError("quota store not initialized")
    return store


def _tracker(request: Request) -> UsageTracker:
    """Return the shared usage tracker from app state.

    Args:
        request: The incoming request.

    Returns:
        The shared :class:`UsageTracker`.

    Raises:
        RuntimeError: If the tracker is not initialized.
    """
    tracker = getattr(request.app.state, "usage_tracker", None)
    if tracker is None:
        raise RuntimeError("usage tracker not initialized")
    return tracker


def _eviction_controller(request: Request) -> CoordinatorEvictionController:
    """Return the shared eviction controller from app state.

    Args:
        request: The incoming request.

    Returns:
        The shared :class:`CoordinatorEvictionController`.

    Raises:
        RuntimeError: If the controller is not initialized.
    """
    ctrl = getattr(request.app.state, "eviction_controller", None)
    if ctrl is None:
        raise RuntimeError("eviction controller not initialized")
    return ctrl


# -- Quota writes ------------------------------------------------------------


@router.put("/l2/quota/{cache_salt}")
async def set_quota(
    cache_salt: str, body: SetQuotaRequest, request: Request
) -> QuotaResponse | JSONResponse:
    """Create or update a quota for the given ``cache_salt``.

    Returns:
        The applied quota.
    """
    limit_bytes = int(body.limit_gb * _GB)
    try:
        _quota_store(request).set(cache_salt, limit_bytes)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return QuotaResponse(
        cache_salt=cache_salt,
        limit_gb=body.limit_gb,
        status="ok",
    )


@router.delete("/l2/quota/{cache_salt}")
async def delete_quota(cache_salt: str, request: Request) -> QuotaResponse:
    """Remove a salt's quota entry.

    Returns:
        Whether the entry was found and removed.
    """
    removed = _quota_store(request).delete(cache_salt)
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
    """Record a batch of L2 store/lookup events.

    Returns:
        Number of events processed.
    """
    tracker = _tracker(request)
    ctrl = _eviction_controller(request)
    for event in body.events:
        if event.type == EventType.STORE:
            tracker.record_stored(event.key.cache_salt, event.bytes)
            ctrl.on_store(event.key, event.bytes)
        elif event.type == EventType.LOOKUP:
            ctrl.on_lookup(event.key)
    return ReportUsageResponse(recorded=len(body.events))


# -- Combined status queries -------------------------------------------------


@router.get("/l2/status/{cache_salt}")
async def get_status(cache_salt: str, request: Request) -> L2StatusResponse:
    """Read quota and usage for a single salt.

    Returns:
        Combined quota and usage detail.
    """
    tracker = _tracker(request)
    store = _quota_store(request)
    usage = tracker.get(cache_salt)
    limit = store.get(cache_salt)
    return L2StatusResponse(
        cache_salt=cache_salt,
        quota_limit_gb=_gb(limit) if limit is not None else 0.0,
        quota_exists=limit is not None,
        usage_gb=_gb(usage),
    )


@router.get("/l2/status")
async def list_status(request: Request) -> L2StatusListResponse:
    """List quota and usage across all cache salts.

    Returns:
        Total usage and per-salt breakdown with quota info.
    """
    tracker = _tracker(request)
    store = _quota_store(request)
    by_salt = tracker.get_all()
    total = tracker.get_total()
    quota_entries = {e.cache_salt: e.limit_bytes for e in store.list_all()}
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

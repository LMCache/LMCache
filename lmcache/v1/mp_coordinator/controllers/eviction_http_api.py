# SPDX-License-Identifier: Apache-2.0
"""The endpoints the fleet eviction controller owns.

``/quota`` is the fleet-level counterpart of the MP server's node-local group,
and ``/cache/pins`` is eviction protection expressed as cache content. Both act
on the controller's own quota registry and pin table, so both are built around
it rather than resolving it per request -- which is what makes them disappear
with it when an operator leaves it unbuilt, freeing the paths for whatever
replaced it.

Usage arrives through the fleet cache-event stream (``POST /events``), admitted
by the ingest gate; the usage view those reads come from is a view, resolved
per request like any other collaborator that is not this controller.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    PinRequest,
    PinResponse,
    QuotaConfigRequest,
    QuotaConfigResponse,
    QuotaResponse,
    SetQuotaRequest,
    StatusListResponse,
    StatusResponse,
)
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
        FleetEvictionController,
    )

_GB = 1024**3
_DEFAULT_SALT_SENTINEL = "_default"
# Quotas are enforced on L2 only, so quota writes reject any other tier.
_QUOTA_TIER = Tier.L2
# Usage is accounted on both concrete tiers, so status reads accept either.
_ACCOUNTED_TIERS = (Tier.L1, Tier.L2)


def _resolve_salt_from_api_path(cache_salt: str) -> str:
    """Map the ``_default`` sentinel to the empty string."""
    return "" if cache_salt == _DEFAULT_SALT_SENTINEL else cache_salt


def _require_quota_tier(tier: Tier) -> None:
    """Raise ``HTTPException(400)`` unless ``tier`` is quota-enforced (``l2``)."""
    if tier != _QUOTA_TIER:
        raise HTTPException(
            status_code=400,
            detail=(
                f"quotas apply to tier {_QUOTA_TIER.value!r} only, not {tier.value!r}"
            ),
        )


def _require_accounted_tier(tier: Tier) -> None:
    """Raise ``HTTPException(400)`` unless ``tier`` has usage accounting.

    ``all`` is rejected: a key resident in both tiers holds bytes in
    both, so a cross-tier total would double-count it.
    """
    if tier not in _ACCOUNTED_TIERS:
        supported = ", ".join(repr(t.value) for t in _ACCOUNTED_TIERS)
        raise HTTPException(
            status_code=400,
            detail=f"tier {tier.value!r} has no usage accounting; only {supported}",
        )


def _quota_entries_for_tier(store: QuotaManager, tier: Tier) -> dict[str, int]:
    """Return the registered byte limits that apply to ``tier``.

    Keyed ``cache_salt`` → limit in bytes. Quotas are enforced on L2
    only, so every other tier has none: its status rows report no quota
    rather than borrowing the L2 table's numbers, which govern different
    bytes entirely.

    Args:
        store: The quota registry to read.
        tier: The tier the caller is reporting on.

    Returns:
        The applicable limits, empty for any tier but ``l2``.
    """
    if tier != _QUOTA_TIER:
        return {}
    return {entry.cache_salt: entry.limit_bytes for entry in store.list_quotas()}


def _gb(n_bytes: int) -> float:
    """Convert bytes to GiB."""
    return n_bytes / _GB


def build_routers(controller: FleetEvictionController) -> tuple[APIRouter, ...]:
    """Build this controller's routers, bound to it.

    Args:
        controller: The controller the handlers act on. Closing over it
            removes the per-request lookup by class, and lets an operator
            disable the controller without leaving routes behind that have
            nothing to answer with.

    Returns:
        The routers for startup to mount.
    """
    router = APIRouter()

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
        _require_quota_tier(body.tier)
        default_limit_bytes = (
            None if body.default_limit_gb is None else int(body.default_limit_gb * _GB)
        )
        ctx = get_context(request)
        controller.quota.set_default_limit_bytes(default_limit_bytes)
        ctx.metadata_persister.save()
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
        _require_quota_tier(tier)
        quota = controller.quota
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
        _require_quota_tier(body.tier)
        cache_salt = _resolve_salt_from_api_path(cache_salt)
        limit_bytes = int(body.limit_gb * _GB)
        ctx = get_context(request)
        try:
            controller.quota.set_quota(cache_salt, limit_bytes)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"error": "invalid quota limit"}
            )
        ctx.metadata_persister.save()
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
        _require_quota_tier(tier)
        cache_salt = _resolve_salt_from_api_path(cache_salt)
        ctx = get_context(request)
        removed = controller.quota.delete_quota(cache_salt)
        ctx.metadata_persister.save()
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
            tier: Cache tier to report (``l1`` or ``l2``). Both the usage and
                the quota fields describe that tier; since quotas are
                enforced on L2 only, an ``l1`` request always reports
                ``quota_exists=False``.

        Returns:
            Combined quota and live usage detail for ``tier``.
        """
        _require_accounted_tier(tier)
        cache_salt = _resolve_salt_from_api_path(cache_salt)
        ctx = get_context(request)
        quota = controller.quota
        usage = ctx.views.get(CacheUsageManager).get_salt_bytes(tier, cache_salt)
        exists = tier == _QUOTA_TIER and quota.has_quota(cache_salt)
        limit = quota.get_limit_bytes(cache_salt) if exists else 0
        return StatusResponse(
            cache_salt=cache_salt,
            quota_limit_gb=_gb(limit),
            quota_exists=exists,
            usage_gb=_gb(usage),
        )

    @router.get("/quota")
    async def list_status(request: Request, tier: Tier = Tier.L2) -> StatusListResponse:
        """List quota and usage across all cache salts.

        Args:
            tier: Cache tier to report (``l1`` or ``l2``). Both the usage and
                the quota fields describe that tier; since quotas are
                enforced on L2 only, an ``l1`` listing carries no quota rows
                and every entry reports ``quota_exists=False``.

        Returns:
            Total usage plus per-salt breakdown with quota info for ``tier``.
        """
        _require_accounted_tier(tier)
        ctx = get_context(request)
        usage = ctx.views.get(CacheUsageManager)
        usage_view = usage
        by_salt = usage_view.get_bytes_by_salt(tier)
        total = usage_view.get_total_bytes(tier)
        quota_entries = _quota_entries_for_tier(controller.quota, tier)
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

    # -- Pin / unpin (eviction protection) --------------------------------

    @router.post("/cache/pins")
    async def request_pin(body: PinRequest, request: Request) -> PinResponse:
        """Pin a token sequence's keys in the L2 eviction plan.

        The coordinator resolves the token sequence to its object keys locally and
        records them so they are excluded from its L2 eviction plan (fleet-wide,
        per ``cache_salt``). No MP-server round-trip.

        Args:
            body: model/world_size, token_ids, cache_salt.

        Returns:
            ``PinResponse`` with ``requested`` chunks and ``affected`` L2 keys pinned.

        Raises:
            HTTPException: 400 if the token cap is exceeded or a key field is invalid.
        """
        ctx = get_context(request)
        try:
            resolved, chunks = resolve_object_keys(
                ctx.token_hasher,
                body.model_name,
                body.world_size,
                body.token_ids,
                body.cache_salt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        controller.pin(resolved)
        ctx.metadata_persister.save()
        return PinResponse(
            requested=chunks,
            affected=len(resolved),
            status="pinned",
        )

    @router.delete("/cache/pins")
    async def request_unpin(body: PinRequest, request: Request) -> PinResponse:
        """Unpin a token sequence's keys from the L2 eviction plan.

        Symmetric with pin: resolves the keys locally and releases them, making them
        eligible for L2 eviction again.

        Args:
            body: model/world_size, token_ids, cache_salt.

        Returns:
            ``PinResponse`` with ``requested`` chunks and ``affected`` L2 keys unpinned.

        Raises:
            HTTPException: 400 if the token cap is exceeded or a key field is invalid.
        """
        ctx = get_context(request)
        try:
            resolved, chunks = resolve_object_keys(
                ctx.token_hasher,
                body.model_name,
                body.world_size,
                body.token_ids,
                body.cache_salt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        controller.unpin(resolved)
        ctx.metadata_persister.save()
        return PinResponse(
            requested=chunks,
            affected=len(resolved),
            status="unpinned",
        )

    return (router,)

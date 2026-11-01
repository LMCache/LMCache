# SPDX-License-Identifier: Apache-2.0
"""The endpoints the fleet eviction controller owns.

``/quota`` is the fleet-level counterpart of the MP server's node-local group,
``/cache/pins`` is eviction protection expressed as cache content, and
``/cache/delete`` honors those pins. All three act on the controller's own
quota registry and pin table, so they are built around it rather than resolving
it per request -- which is what makes them disappear with it when an operator
leaves it unbuilt, freeing the paths for whatever replaced it.

Usage arrives through the fleet cache-event stream (``POST /events``), admitted
by the ingest gate; the usage view those reads come from is a view, resolved
per request like any other collaborator that is not this controller.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import httpx

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.mp_coordinator.http_apis.dependencies import (
    get_context,
    get_outbound_client,
)
from lmcache.v1.mp_coordinator.persistence.durable_component import DurableComponent
from lmcache.v1.mp_coordinator.persistence.metadata import MetadataWriteError
from lmcache.v1.mp_coordinator.schemas import (
    DeleteRequest,
    DeleteResponse,
    PinListResponse,
    PinnedKeyInfo,
    PinRequest,
    PinResponse,
    QuotaConfigRequest,
    QuotaConfigResponse,
    QuotaResponse,
    SetQuotaRequest,
    StatusListResponse,
    StatusResponse,
)
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
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


def _save_or_undo(
    request: Request, component: DurableComponent, before: Mapping[str, object]
) -> None:
    """Persist a change to ``component``, or undo it and answer 503.

    Pins and quotas are operator intent: a ``200`` must mean the change
    survives a restart. If the metadata document cannot be written, the
    change is rolled back in memory, so the coordinator never runs on
    intent it would lose, and the caller learns it can retry.

    Args:
        request: The request that made the change.
        component: The pin table or quota registry that was changed.
        before: ``component.capture()`` taken just before the change.

    Raises:
        HTTPException: 503 if the document could not be written.
    """
    try:
        get_context(request).metadata_persister.save()
    except MetadataWriteError as exc:
        component.restore(before)
        raise HTTPException(
            status_code=503, detail=f"change not saved and undone: {exc}"
        ) from None


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
        before = controller.quota.capture()
        controller.quota.set_default_limit_bytes(default_limit_bytes)
        _save_or_undo(request, controller.quota, before)
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
        before = controller.quota.capture()
        try:
            controller.quota.set_quota(cache_salt, limit_bytes)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"error": "invalid quota limit"}
            )
        _save_or_undo(request, controller.quota, before)
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
        before = controller.quota.capture()
        removed = controller.quota.delete_quota(cache_salt)
        _save_or_undo(request, controller.quota, before)
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

        before = controller.capture()
        controller.pin(resolved)
        _save_or_undo(request, controller, before)
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

        before = controller.capture()
        controller.unpin(resolved)
        _save_or_undo(request, controller, before)
        return PinResponse(
            requested=chunks,
            affected=len(resolved),
            status="unpinned",
        )

    @router.get("/cache/pins")
    async def list_pins(
        cache_salt: str = "",
        model_name: str = "",
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=10000),
    ) -> PinListResponse:
        """List the keys pinned against L2 eviction, one page at a time.

        Reads the coordinator's pin table. Pins are fleet-wide, so there is no
        ``instance_id`` filter, and they apply to L2 only, so there is no
        ``tier`` filter.

        Args:
            cache_salt: Keep keys with this salt. Empty keeps every salt.
            model_name: Keep keys for this model. Empty keeps every model.
            offset: Matching keys to skip.
            limit: Maximum keys to return.

        Returns:
            ``PinListResponse`` with the filtered total and the requested page,
            each key with its pin count.
        """
        total, page = controller.list_pins(cache_salt, model_name, offset, limit)
        return PinListResponse(
            total=total,
            pins=[
                PinnedKeyInfo(
                    key=pinned.key.to_encoded_object_key(),
                    pin_count=pinned.pin_count,
                )
                for pinned in page
            ],
        )

    # -- Delete (honors the pins above) ----------------------------------

    @router.post("/cache/delete")
    async def request_delete(body: DeleteRequest, request: Request) -> DeleteResponse:
        """Delete a token sequence on one MP server, per ``body.tier``.

        ``body.tier`` is ``l1``, ``l2`` or ``all``. When the delete touches L2,
        keys pinned in this controller's table are held back unless
        ``body.force`` is set; a forced delete removes them and drops their
        pins. Served by the controller that holds the pins, so a replacement
        eviction controller serves its own.

        Args:
            body: Target instance, model/world_size, token_ids, cache_salt, tier,
                force.

        Returns:
            ``DeleteResponse`` with ``requested`` chunks, ``affected`` keys
            removed, and ``skipped`` keys refused (L1 locks reported by the node,
            plus L2 pins held back here).

        Raises:
            HTTPException: 400 if the token cap is exceeded or a key field is
                invalid; 404 if ``instance_id`` is not registered; 502 if the
                target server is unreachable or rejects the delete.
        """
        ctx = get_context(request)
        target = ctx.views.get(InstanceRegistry).get(body.instance_id)
        if target is None:
            raise HTTPException(
                status_code=404,
                detail=f"no MP server registered with instance_id={body.instance_id!r}",
            )
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
        resolved = ctx.views.get(KeyDirectory).get_keys_across_object_groups(resolved)

        if not chunks:
            return DeleteResponse(
                instance_id=body.instance_id,
                requested=0,
                affected=0,
                skipped=0,
                status="noop",
            )

        # When the delete touches L2, hold back L2-pinned keys (non-force). They
        # are dropped from the delete set entirely, so a pinned key is retained
        # in both tiers; ``force`` deletes them and clears the pins.
        touches_l2 = body.tier in (Tier.L2, Tier.ALL)
        if touches_l2 and not body.force:
            delete_keys = controller.filter_unpinned(resolved)
            pin_skipped = len(resolved) - len(delete_keys)
        else:
            delete_keys = resolved
            pin_skipped = 0

        node_deleted = 0
        node_skipped = 0
        if delete_keys:
            client = get_outbound_client(request)
            url = f"http://{target.ip}:{target.http_port}/cache/objects"
            payload = {
                "keys": [asdict(k.to_encoded_object_key()) for k in delete_keys],
                "tier": body.tier.value,
                "force": body.force,
            }
            try:
                # httpx ``.delete`` can't take ``json=``; use ``request(...)``.
                resp = await client.request("DELETE", url, json=payload)
                resp.raise_for_status()
                result = resp.json()
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"delete to {body.instance_id!r} failed: {exc}",
                ) from None
            node_deleted = result.get("deleted", 0)
            node_skipped = result.get("skipped", 0)

        if touches_l2 and body.force:
            before = controller.capture()
            controller.drop_pins(resolved)
            try:
                ctx.metadata_persister.save()
            except MetadataWriteError as exc:
                # The server already deleted the keys, which cannot be
                # undone; keep the pins, so memory agrees with the
                # document, and say exactly what happened.
                controller.restore(before)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"{node_deleted} keys deleted on {body.instance_id!r}, "
                        f"but dropping their pins was not saved; the pins are "
                        f"kept: {exc}"
                    ),
                ) from None

        # ``skipped`` = L1 keys the node refused (locks) + L2 keys held back
        # for a pin.
        return DeleteResponse(
            instance_id=body.instance_id,
            requested=chunks,
            affected=node_deleted,
            skipped=node_skipped + pin_skipped,
            status="deleted",
        )

    return (router,)

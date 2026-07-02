# SPDX-License-Identifier: Apache-2.0
"""Cache-control endpoints on the coordinator (fleet-level).

Warm-prefetch dispatch to a named MP server, thin over the
:class:`PrefetchManager` on the typed :class:`CoordinatorContext` (resolved via
:func:`get_context`). Handlers map fleet-routing failures to HTTP directly --
``404`` for an unknown ``instance_id`` and ``502`` when an MP server is
unreachable or rejects a proxied call.

Quota writes, combined quota+usage status, and usage-event ingestion are
accounting concerns and live in the ``/quota`` group (:mod:`quota_api`).
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey
from lmcache.v1.distributed.tiers import Tier
from lmcache.v1.mp_coordinator.http_apis.dependencies import (
    get_context,
    get_outbound_client,
)
from lmcache.v1.mp_coordinator.schemas import (
    PinRequest,
    PinResponse,
    PrefetchRequest,
    PrefetchResponse,
)

router = APIRouter()


# -- Prefetch dispatch -------------------------------------------------------


@router.post("/cache/prefetches")
async def request_prefetch(body: PrefetchRequest, request: Request) -> PrefetchResponse:
    """Submit a warm prefetch of a token sequence on one MP server.

    Resolves ``body.instance_id`` in the registry and proxies to that server's
    ``POST /cache/prefetches``, which submits the load and returns a
    ``request_id``. Poll ``GET /cache/prefetches/{instance_id}/{request_id}``.

    Args:
        body: Target instance, model/world_size, the token_ids to warm, and the
            per-tenant cache_salt.

    Returns:
        ``PrefetchResponse`` carrying the server's ``request_id`` (empty when the
        sequence was shorter than one chunk -- ``status`` ``"noop"``).

    Raises:
        HTTPException: 404 if ``instance_id`` is not registered; 502 if the
            target server is unreachable or rejects the submit.
    """
    ctx = get_context(request)
    target = ctx.registry.get(body.instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={body.instance_id!r}",
        )

    try:
        result = await ctx.prefetch_manager.submit_prefetch(
            target=target,
            http_client=get_outbound_client(request),
            model_name=body.model_name,
            world_size=body.world_size,
            token_ids=body.token_ids,
            cache_salt=body.cache_salt,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"prefetch submit to {body.instance_id!r} failed: {exc}",
        ) from None

    return PrefetchResponse(
        instance_id=body.instance_id,
        request_id=result.get("request_id", ""),
        chunks=result.get("chunks", 0),
        status=result.get("status", "submitted"),
    )


@router.get("/cache/prefetches/{instance_id}/{request_id}")
async def get_prefetch_status(
    instance_id: str, request_id: str, request: Request
) -> JSONResponse:
    """Proxy a warm-prefetch status poll to the owning MP server.

    The warm holds no lock, so this poll only reports progress; the first poll
    that observes completion drops the job on the server (exactly-once). Poll
    until ``"completed"``.

    Args:
        instance_id: The MP server the prefetch was submitted to.
        request_id: The id returned by ``POST /cache/prefetches``.

    Returns:
        The server's status body relayed verbatim with its status code (200
        ``pending`` / ``completed``, or 404 for an unknown id).

    Raises:
        HTTPException: 404 if ``instance_id`` is not registered; 502 if the
            target server is unreachable.
    """
    ctx = get_context(request)
    target = ctx.registry.get(instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={instance_id!r}",
        )

    try:
        code, payload = await ctx.prefetch_manager.get_status(
            target=target,
            http_client=get_outbound_client(request),
            request_id=request_id,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"prefetch status from {instance_id!r} failed: {exc}",
        ) from None

    return JSONResponse(status_code=code, content=payload)


# -- Pin / unpin dispatch ----------------------------------------------------


@router.post("/cache/pins")
async def request_pin(body: PinRequest, request: Request) -> PinResponse:
    """Pin a token sequence on one MP server, per ``body.tier`` (l1 / l2 / all).

    ``l1`` pins only the server's L1; ``l2`` records only the coordinator's L2
    protection (L1 untouched); ``all`` does both. The MP server returns the
    resolved keys, which the coordinator records in L2 for ``l2``/``all``.

    Args:
        body: Target instance, model/world_size, token_ids, cache_salt, tier.

    Returns:
        ``PinResponse`` with ``requested`` chunks and ``affected`` L1 keys pinned.

    Raises:
        HTTPException: 404 if ``instance_id`` is not registered; 502 if the
            target server is unreachable or rejects the pin.
    """
    ctx = get_context(request)
    target = ctx.registry.get(body.instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={body.instance_id!r}",
        )

    try:
        result = await ctx.pin_manager.submit_pin(
            target=target,
            http_client=get_outbound_client(request),
            model_name=body.model_name,
            world_size=body.world_size,
            token_ids=body.token_ids,
            cache_salt=body.cache_salt,
            tier=body.tier.value,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"pin to {body.instance_id!r} failed: {exc}",
        ) from None

    # Pin L2 only when the tier includes it (l2 / all); l1 never touches L2.
    resolved = [
        EncodedObjectKey(**k).to_object_key() for k in result.get("resolved_keys", [])
    ]
    if body.tier in (Tier.L2, Tier.ALL):
        ctx.eviction_manager.pin(resolved)

    affected = len(resolved) if body.tier is Tier.L2 else result.get("pinned", 0)
    return PinResponse(
        instance_id=body.instance_id,
        requested=result.get("requested", 0),
        affected=affected,
        status=result.get("status", "pinned"),
    )


@router.delete("/cache/pins")
async def request_unpin(body: PinRequest, request: Request) -> PinResponse:
    """Unpin a token sequence on one MP server, per ``body.tier`` (l1 / l2 / all).

    Symmetric with pin: ``l1`` unpins only the server's L1, ``l2`` releases only
    the coordinator's L2 protection, ``all`` does both.

    Args:
        body: Target instance, model/world_size, token_ids, cache_salt, tier.

    Returns:
        ``PinResponse`` with ``requested`` chunks and ``affected`` L1 keys unpinned.

    Raises:
        HTTPException: 404 if ``instance_id`` is not registered; 502 if the
            target server is unreachable or rejects the unpin.
    """
    ctx = get_context(request)
    target = ctx.registry.get(body.instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={body.instance_id!r}",
        )

    try:
        result = await ctx.pin_manager.submit_unpin(
            target=target,
            http_client=get_outbound_client(request),
            model_name=body.model_name,
            world_size=body.world_size,
            token_ids=body.token_ids,
            cache_salt=body.cache_salt,
            tier=body.tier.value,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"unpin to {body.instance_id!r} failed: {exc}",
        ) from None

    # Release L2 only when the tier includes it (symmetric with pin).
    resolved = [
        EncodedObjectKey(**k).to_object_key() for k in result.get("resolved_keys", [])
    ]
    if body.tier in (Tier.L2, Tier.ALL):
        ctx.eviction_manager.unpin(resolved)

    affected = len(resolved) if body.tier is Tier.L2 else result.get("unpinned", 0)
    return PinResponse(
        instance_id=body.instance_id,
        requested=result.get("requested", 0),
        affected=affected,
        status=result.get("status", "unpinned"),
    )

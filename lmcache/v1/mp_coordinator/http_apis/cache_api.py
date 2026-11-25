# SPDX-License-Identifier: Apache-2.0
"""Cache-control endpoints on the coordinator (fleet-level).

Warm-prefetch dispatch to a named MP server, thin over the
:class:`PrefetchManager` on the typed :class:`CoordinatorContext` (resolved via
:func:`get_context`). Handlers map fleet-routing failures to HTTP directly --
``404`` for an unknown ``instance_id`` and ``502`` when an MP server is
unreachable or rejects a proxied call.

Quota writes, combined quota+usage status, pins and delete act on the eviction
controller's state, so that controller brings them itself
(:mod:`lmcache.v1.mp_coordinator.controllers.eviction_http_api`).
"""

# Standard
from dataclasses import asdict
from typing import Any

# Third Party
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.http_apis.dependencies import (
    get_context,
    get_outbound_client,
)
from lmcache.v1.mp_coordinator.schemas import (
    PrefetchRequest,
    PrefetchResponse,
)
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry
from lmcache.v1.multiprocess.cache_control.key_resolver import (
    resolve_chunk_object_keys,
)

router = APIRouter()


# -- Prefetch dispatch -------------------------------------------------------


@router.post("/cache/prefetches")
async def request_prefetch(body: PrefetchRequest, request: Request) -> PrefetchResponse:
    """Submit a warm prefetch of keys, chunk hashes or a token sequence on one
    MP server.

    Resolves ``body.instance_id`` in the registry and proxies to that server's
    ``POST /cache/prefetches``, which submits the load and returns a
    ``request_id``. Poll ``GET /cache/prefetches/{instance_id}/{request_id}``.

    Args:
        body: Target instance, and the ``keys`` to warm, or model/world_size,
            the token_ids or chunk_hashes to warm, and the per-tenant
            cache_salt.

    Returns:
        ``PrefetchResponse`` carrying the server's ``request_id`` (empty when the
        sequence was shorter than one chunk -- ``status`` ``"noop"``).

    Raises:
        HTTPException: 400 if a chunk-hash form's key field is invalid; 404
            if ``instance_id`` is not registered; 502 if the target server
            is unreachable or rejects the submit.
    """
    ctx = get_context(request)
    prefetch = ctx.controllers.get(PrefetchManager)
    target = ctx.views.get(InstanceRegistry).get(body.instance_id)
    if target is None:
        prefetch.record_unknown_server()
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={body.instance_id!r}",
        )

    try:
        result = await prefetch.submit_prefetch(
            target=target,
            http_client=get_outbound_client(request),
            body=_forwarded_prefetch_body(body),
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
    prefetch = ctx.controllers.get(PrefetchManager)
    target = ctx.views.get(InstanceRegistry).get(instance_id)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"no MP server registered with instance_id={instance_id!r}",
        )

    try:
        code, payload = await prefetch.get_status(
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


def _forwarded_prefetch_body(body: PrefetchRequest) -> dict[str, Any]:
    """The MP server's ``POST /cache/prefetches`` body for ``body``.

    Carries only the form ``body`` uses: a server without the keys form then
    rejects a keys request rather than reading empty token fields as a
    sub-chunk sequence and answering noop. The chunk-hash form goes as the
    keys form, one key per distinct chunk: the server warms a chunk by any
    one of its keys, and one per rank could exceed its key cap.

    Raises:
        HTTPException: 400 if a chunk-hash form's key field (e.g.
            ``cache_salt``) is invalid.
    """
    if body.keys:
        return {"keys": [asdict(key) for key in body.keys]}
    if body.chunk_hashes:
        try:
            keys, _ = resolve_chunk_object_keys(
                [bytes.fromhex(h) for h in body.chunk_hashes],
                body.model_name,
                body.world_size,
                body.cache_salt,
                tags=body.tags,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        key_per_chunk: dict[bytes, ObjectKey] = {}
        for key in keys:
            key_per_chunk.setdefault(key.chunk_hash, key)
        return {
            "keys": [
                asdict(key.to_encoded_object_key()) for key in key_per_chunk.values()
            ]
        }
    forwarded: dict[str, Any] = {
        "model_name": body.model_name,
        "world_size": body.world_size,
        "token_ids": body.token_ids,
        "cache_salt": body.cache_salt,
    }
    if body.tags:
        forwarded["tags"] = dict(body.tags)
    return forwarded

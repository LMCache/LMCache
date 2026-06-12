# SPDX-License-Identifier: Apache-2.0
"""
HTTP endpoints for managing L2 KV cache keys.

Both endpoints target the **primary** L2 adapter — the first one
configured. The HTTP surface deliberately exposes no adapter selector:
a deployment that wants these endpoints to operate on a specific
adapter must list that adapter first in its L2 configuration.

- ``POST /l2/keys:evict`` — delete a caller-supplied list of keys.
  Idempotent: keys absent from the adapter are skipped silently; keys
  currently locked by in-flight store/load tasks (S3) are skipped so
  eviction never corrupts an active transfer.

- ``GET /l2/keys`` — paginate keys resident in the primary adapter,
  filtered by ``cache_salt`` and/or ``model_name``. Returns 501 when
  the primary adapter does not implement listing (in v1 only
  ``S3L2Adapter`` does).

L1 is intentionally NOT touched by ``:evict`` — keys evicted from L2
may still return from L1 until natural L1 eviction expires them.
Callers that need an L1+L2 purge should layer their own L1 invalidation
or wait for the existing L1 eviction controller.
"""

# Standard
from dataclasses import asdict
from typing import Any

# Third Party
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey

router = APIRouter()


# Same sentinel convention as ``quota_api`` — an empty ``cache_salt``
# (un-salted / anonymous traffic) cannot be expressed in a URL query
# parameter, so callers pass ``_default`` to filter for it.
_DEFAULT_SALT_SENTINEL = "_default"

_MAX_PAGE_SIZE = 5000
_DEFAULT_PAGE_SIZE = 500
# Hard cap on how many keys a single :evict request may target. Keeps
# the request body bounded and prevents a single call from monopolizing
# the adapter's I/O loop for an unbounded interval.
_MAX_EVICT_BATCH = 10_000


def _get_storage_manager(request: Request) -> Any:
    """Resolve the shared ``StorageManager`` or return a 503 response.

    Returns either the ``StorageManager`` instance or a
    :class:`JSONResponse` (503) ready to be returned directly from the
    endpoint — keeps the endpoint bodies short.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return engine.storage_manager


class EvictRequest(BaseModel):
    """Wire body for :py:func:`evict_l2_keys`.

    Pydantic validates the ``keys`` list (length cap, per-item type
    coercion) before our handler runs; malformed bodies surface as
    FastAPI 422s without us writing manual checks. Per-item
    :class:`EncodedObjectKey` → :class:`ObjectKey` conversion still happens
    inside the handler because the ObjectKey invariants
    (hex parse, ``@``-in-model-name, salt charset) aren't expressible
    as Pydantic types — those raise ``ValueError`` which we map to 400.
    """

    keys: list[EncodedObjectKey] = Field(..., max_length=_MAX_EVICT_BATCH)


@router.post("/l2/keys:evict")
async def evict_l2_keys(body: EvictRequest, request: Request) -> Any:
    """Evict a caller-supplied list of keys from the primary L2 adapter.

    Body schema: :class:`EvictRequest` — ``{"keys": [EncodedObjectKey, ...]}``.

    Responses:
        200: ``{"requested": N, "adapter": "<type_name>", "ok": <bool>}``
            (with optional ``"error"`` field on ``ok=False``).
        400: a key's payload survived Pydantic typing but violates an
            ``ObjectKey`` invariant (bad hex, ``@`` in ``model_name``,
            forbidden ``cache_salt`` char, ...).
        422: Pydantic-level validation failure (missing ``keys``,
            wrong types, batch over ``_MAX_EVICT_BATCH``).
        503: engine not initialized OR no L2 adapters configured.
    """
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm

    parsed = []
    for i, cache_key in enumerate(body.keys):
        try:
            parsed.append(cache_key.to_object_key())
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"keys[{i}]: {exc}"},
            )

    try:
        report = sm.evict_l2_keys(parsed)
    except ValueError as exc:
        # Surfaced when no L2 adapters are configured — operationally
        # equivalent to "engine isn't ready to serve this endpoint."
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return {"requested": len(parsed), **report}


@router.get("/l2/keys")
async def list_l2_keys(
    request: Request,
    cache_salt: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE, ge=1, le=_MAX_PAGE_SIZE),
    page_token: str | None = Query(default=None),
) -> Any:
    """List keys resident in the primary L2 adapter, filtered +
    paginated.

    Query parameters:
        cache_salt: restrict to one ``cache_salt`` value. Pass
            ``"_default"`` to match the empty-string salt (un-salted
            traffic). Omit to return all salts.
        model_name: restrict to one model name. Omit to return all.
        page_size: max entries per page. Clamped to ``[1, 5000]``;
            default ``500``.
        page_token: opaque cursor returned by the previous page. Omit
            on the first call. Pass back verbatim to get the next page.

    Responses:
        200: ``{"entries": [<wire ObjectKey + size_bytes + adapter>, ...],
                "next_page_token": "<opaque>" | null}``.
        400: malformed page_token (adapter-level).
        501: primary adapter does not implement listing.
        503: engine not initialized OR no L2 adapters configured.
    """
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm

    salt_filter = "" if cache_salt == _DEFAULT_SALT_SENTINEL else cache_salt
    try:
        page = sm.list_l2_keys(
            cache_salt=salt_filter,
            model_name=model_name,
            page_size=page_size,
            page_token=page_token,
        )
    except ValueError as exc:
        msg = str(exc)
        # ``no L2 adapters configured`` → 503 (server-side state). All
        # other ``ValueError``s from this code path are adapter-level
        # validation failures (e.g. malformed page_token) → 400.
        if msg.startswith("no L2 adapters"):
            return JSONResponse(status_code=503, content={"error": msg})
        return JSONResponse(status_code=400, content={"error": msg})
    except NotImplementedError as exc:
        return JSONResponse(
            status_code=501,
            content={"error": f"primary L2 adapter does not support listing: {exc}"},
        )

    entries: list[dict[str, Any]] = []
    for entry in page.entries:
        wire = asdict(entry.key.to_encoded_object_key())
        wire["size_bytes"] = entry.size_bytes
        wire["adapter"] = entry.adapter_name
        entries.append(wire)
    return {
        "entries": entries,
        "next_page_token": page.next_page_token,
    }

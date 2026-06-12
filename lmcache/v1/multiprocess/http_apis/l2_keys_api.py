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
from typing import Any

# Third Party
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.v1.distributed.api import ObjectKey

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


def _parse_object_key(raw: Any) -> ObjectKey:
    """Decode one wire-format key into an :class:`ObjectKey`.

    Wire schema (JSON object)::

        {
            "chunk_hash_hex": "<lowercase hex>",
            "model_name": "<str>",
            "kv_rank": <int>,
            "cache_salt": "<str, optional, defaults to ''>"
        }

    Raises ``ValueError`` on any malformed field. Caller is responsible
    for catching and turning the error into a 400 response.
    """
    if not isinstance(raw, dict):
        raise ValueError("each key must be a JSON object")
    try:
        chunk_hash_hex = raw["chunk_hash_hex"]
        model_name = raw["model_name"]
        kv_rank = raw["kv_rank"]
    except KeyError as exc:
        raise ValueError(f"missing required field: {exc.args[0]}") from None
    if not isinstance(chunk_hash_hex, str):
        raise ValueError("chunk_hash_hex must be a hex string")
    if not isinstance(model_name, str):
        raise ValueError("model_name must be a string")
    if not isinstance(kv_rank, int) or isinstance(kv_rank, bool):
        raise ValueError("kv_rank must be an integer")
    cache_salt = raw.get("cache_salt", "")
    if not isinstance(cache_salt, str):
        raise ValueError("cache_salt must be a string")
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
    except ValueError as exc:
        raise ValueError(f"chunk_hash_hex is not valid hex: {exc}") from None
    # ObjectKey enforces additional invariants on model_name / cache_salt
    # (no ``@`` in model_name, etc.) — let the dataclass post_init raise
    # ValueError; we catch & relabel in the endpoint.
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name=model_name,
        kv_rank=kv_rank,
        cache_salt=cache_salt,
    )


def _encode_object_key(key: ObjectKey) -> dict[str, Any]:
    """Serialize an :class:`ObjectKey` into the wire schema."""
    return {
        "chunk_hash_hex": key.chunk_hash.hex(),
        "model_name": key.model_name,
        "kv_rank": key.kv_rank,
        "cache_salt": key.cache_salt,
    }


@router.post("/l2/keys:evict")
async def evict_l2_keys(request: Request) -> Any:
    """Evict a caller-supplied list of keys from the primary L2 adapter.

    Body::

        {"keys": [<ObjectKey wire-form>, ...]}

    See :func:`_parse_object_key` for the per-key wire schema.

    Responses:
        200: ``{"requested": N, "adapter": "<type_name>", "ok": <bool>}``
            (with optional ``"error"`` field on ``ok=False``).
        400: malformed body or unknown per-key field.
        503: engine not initialized OR no L2 adapters configured.
    """
    sm = _get_storage_manager(request)
    if isinstance(sm, JSONResponse):
        return sm

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
    if not isinstance(body, dict) or "keys" not in body:
        return JSONResponse(
            status_code=400,
            content={"error": "body must be {'keys': [<ObjectKey>, ...]}"},
        )
    raw_keys = body["keys"]
    if not isinstance(raw_keys, list):
        return JSONResponse(
            status_code=400,
            content={"error": "'keys' must be a list"},
        )
    if len(raw_keys) > _MAX_EVICT_BATCH:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"too many keys in a single request "
                    f"(limit={_MAX_EVICT_BATCH}, got={len(raw_keys)})"
                )
            },
        )
    parsed: list[ObjectKey] = []
    for i, raw in enumerate(raw_keys):
        try:
            parsed.append(_parse_object_key(raw))
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
        wire = _encode_object_key(entry.key)
        wire["size_bytes"] = entry.size_bytes
        wire["adapter"] = entry.adapter_name
        entries.append(wire)
    return {
        "entries": entries,
        "next_page_token": page.next_page_token,
    }

# SPDX-License-Identifier: Apache-2.0
"""Low-level runtime helpers for ``lmcache bench server``."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import json
import time
import urllib.error
import urllib.request

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import DeviceMessagingFuture, MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

_HELLO_TOKEN_ID = 9906
_MODEL_NAME = "test-model"
_WORLD_SIZE = 1
_INSTANCE_ID = 0

# TP > 1 support: each simulated worker registers under a distinct
# ``instance_id`` so the server can hold one context per rank. Kept
# well above legacy ``_INSTANCE_ID = 0`` so single-worker bench runs
# and the multi-worker path never collide on the server side.
_INSTANCE_ID_BASE = 1000

# ------------------------------------------------------------------ #
#  Low-level helpers                                                   #
# ------------------------------------------------------------------ #

# Default RPC call timeout (seconds) for blocking request/reply
# round-trips.
_DEFAULT_RPC_TIMEOUT_S = 10.0

# Unique sentinel returned by :func:`_call` on RPC timeout so callers
# can disambiguate it from a legitimate ``None`` (void) reply.
_TIMEOUT = object()


def _wait_for_result(
    future: MessagingFuture[Any],
    timeout_s: float = _DEFAULT_RPC_TIMEOUT_S,
    retain_refs: tuple[object, ...] = (),
) -> Any:
    """Wait for an RPC future and convert a timeout to ``_TIMEOUT``.

    Returns the decoded response (possibly ``None`` for void replies)
    on success, or the sentinel ``_TIMEOUT`` on RPC timeout. Objects in
    ``retain_refs`` stay owned by the raw future until the transport
    finishes with the request, including timeout paths where the reply may
    still arrive later.
    """
    # The transport owns the raw future, not the temporary device wrapper.
    # Retain producer events there so they also survive a caller timeout.
    owner = future.raw_future_ if isinstance(future, DeviceMessagingFuture) else future
    for ref in retain_refs:
        owner.retain_reference(ref)
    try:
        result = future.result(timeout=timeout_s)
    except TimeoutError:
        return _TIMEOUT
    release_references = getattr(owner, "release_references", None)
    if callable(release_references):
        release_references()
    return result


# ------------------------------------------------------------------ #
#  Token / key helpers                                                 #
# ------------------------------------------------------------------ #


def _build_token_ids(
    seq_no: int,
    num_tokens: int,
) -> tuple[int, ...]:
    """Build token sequence: ``(seq_no, hello, hello, ...)``."""
    return (seq_no,) + (_HELLO_TOKEN_ID,) * num_tokens


def _make_key(
    token_ids: tuple[int, ...],
    request_id: str,
    start: int = 0,
    end: int = 0,
    worker_id: int | None = None,
    world_size: int = _WORLD_SIZE,
    num_kv_readers: int = 1,
) -> IPCCacheServerKey:
    """Build an IPCCacheServerKey."""
    return IPCCacheServerKey.from_token_ids(
        model_name=_MODEL_NAME,
        world_size=world_size,
        num_kv_readers=num_kv_readers,
        worker_id=worker_id,
        token_ids=token_ids,
        start=start,
        end=end if end > 0 else len(token_ids),
        request_id=request_id,
    )


def _send_lookup(
    client: RequestClient,
    key: IPCCacheServerKey,
    tp_size: int = 1,
) -> bool:
    """LOOKUP — submit a prefix lookup.

    The server reserves ``key.num_kv_readers`` read locks per chunk
    (each reader's RETRIEVE releases one; see
    ``IPCCacheServerKey.require_num_kv_readers``). ``tp_size`` is
    a legacy wire field the server ignores.

    The server-side handler returns ``None`` (void) on success, so
    we only distinguish RPC timeout from a completed call.
    """
    result = _wait_for_result(client.lookup(key, tp_size))
    return result is not _TIMEOUT


def _poll_prefetch_status(
    client: RequestClient,
    request_id: str,
    max_polls: int = 50,
    poll_interval: float = 0.05,
) -> int | None:
    """QUERY_PREFETCH_STATUS — poll until done.

    Returns the hit chunk count, or ``None`` if the polling budget
    is exhausted. The server keys prefetch jobs by ``request_id``
    (str), not an integer job handle.
    """
    for _ in range(max_polls):
        result = _wait_for_result(client.query_prefetch_status(request_id))
        if result is _TIMEOUT:
            # RPC timeout — treat as giving up on this poll cycle.
            return None
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _send_end_session(
    client: RequestClient,
    request_id: str,
) -> None:
    """END_SESSION — clean up server-side session state."""
    _wait_for_result(client.end_session(request_id))


# ------------------------------------------------------------------ #
#  Checksum query                                                      #
# ------------------------------------------------------------------ #


def _query_checksum(
    http_base: str,
    block_ids: list[int],
    blocks_per_chunk: int,
    instance_id: int,
) -> dict[str, list[str]] | None:
    """Query KV cache checksums via the HTTP API.

    ``block_ids`` supplies physical IDs in logical order; ``blocks_per_chunk``
    counts blocks rather than tokens. ``layerwise=true`` returns a mapping
    from registered tensor names to per-chunk digests. HTTP errors or invalid
    responses return ``None``.

    ``instance_id`` selects the registered worker context to hash.
    """
    request = urllib.request.Request(
        http_base.rstrip("/") + "/cache/checksums",
        data=json.dumps(
            dict(
                block_ids=block_ids,
                chunk_size=blocks_per_chunk,
                instance_id=instance_id,
                layerwise=True,
            )
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("status") != "success":
        return None
    checksums = data.get("chunk_checksums")
    if not isinstance(checksums, dict) or not all(
        isinstance(parts, list) and all(isinstance(part, str) for part in parts)
        for parts in checksums.values()
    ):
        return None
    return checksums


# ------------------------------------------------------------------ #
#  Server query helper                                                 #
# ------------------------------------------------------------------ #


def _get_chunk_size(client: RequestClient) -> int:
    """Query the server's chunk size."""
    result = _wait_for_result(client.get_chunk_size())
    if result is _TIMEOUT or result is None:
        return 256  # fallback
    return int(result)

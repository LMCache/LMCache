# SPDX-License-Identifier: Apache-2.0
"""Resolve a token sequence to the per-rank cache object keys it maps to.

Several cache-control operations describe content by ``token_ids`` rather than
by internal :class:`ObjectKey` values, which callers cannot construct. This
hashes the tokens and expands each chunk into one key per rank, the same fan-out
the lookup path uses. Failures raise :class:`Unavailable` (the model has not
allocated KV cache on this node yet) or :class:`InvalidRequest` (an invalid key
field).
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.cache_control.errors import InvalidRequest, Unavailable
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheServer

# Hard cap on how many token ids a single token-addressed request may carry.
# Keeps the request body bounded and the synchronous hashing / key-construction
# work proportionate.
MAX_TOKEN_IDS = 1_000_000


def resolve_l1_keys(
    engine: MPCacheServer,
    model_name: str,
    world_size: int,
    token_ids: list[int],
    cache_salt: str,
) -> tuple[list[ObjectKey], int, MemoryLayoutDesc]:
    """Resolve a token sequence to the per-rank object keys it maps to.

    Args:
        engine: The node's cache engine (its ``context`` carries the token
            hasher and layout registry).
        model_name: Model whose layout and rank fan-out to use.
        world_size: Tensor-parallel world size selecting the layout.
        token_ids: The token sequence to resolve.
        cache_salt: Per-tenant isolation salt.

    Returns:
        ``(obj_keys, chunk_count, layout_desc)``. ``obj_keys`` is empty (and
        ``chunk_count`` is 0) when the sequence is shorter than one chunk --
        callers treat that as a no-op.

    Raises:
        Unavailable: No layout is registered for ``(model_name, world_size)``
            (the model has not allocated KV cache on this node yet).
        InvalidRequest: A key field (e.g. ``cache_salt``) is invalid.
    """
    ctx = engine.context
    layout_desc = ctx.layout_desc_registry.find(model_name, world_size)
    if layout_desc is None:
        raise Unavailable(
            f"no layout registered for model_name={model_name!r} "
            f"world_size={world_size}; the model has not allocated "
            f"KV cache on this node yet"
        )

    try:
        ipc_key = IPCCacheServerKey(
            model_name=model_name,
            world_size=world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=0,
            end=len(token_ids),
            request_id="",
            cache_salt=cache_salt,
        )
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from None

    chunk_hashes = ctx.token_hasher.compute_chunk_hashes(list(token_ids))
    if not chunk_hashes:
        return [], 0, layout_desc
    obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes, [0])[0]
    return obj_keys, len(chunk_hashes), layout_desc

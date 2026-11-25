# SPDX-License-Identifier: Apache-2.0
"""Resolve a token sequence to the cache object keys it maps to.

Cache-control operations describe content by ``token_ids`` rather than internal
:class:`ObjectKey` values, which callers cannot construct.
:func:`resolve_object_keys` hashes the tokens and expands each complete chunk
into one key per rank in object group ``0``, and
:func:`resolve_chunk_object_keys` does the same for chunks named by hash;
coordinator callers reach the other groups through what the directory holds
and how pins match.
:func:`resolve_grouped_object_keys` is the node's prefetch resolver, and
expands every object group the model registered, each with its own layout and
attention window; :func:`resolve_grouped_chunk_keys` does the same for chunks
named by hash rather than by tokens, and :func:`named_chunks_of_keys` splits
keys naming chunks into the parallel setup they share and their hashes.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.common.attn_window import AttnWindowDesc
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    FULL_ATTENTION_WINDOW_CHUNKS,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    ObjectKeyIdentity,
    ShardReplicationDesc,
    ipc_key_to_grouped_object_keys,
    ipc_key_to_object_keys,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.token_hasher import TokenHasher

# Hard cap on how many token ids a single token-addressed request may carry.
# Keeps the request body bounded and the synchronous hashing / key-construction
# work proportionate.
MAX_TOKEN_IDS = 1_000_000

# Hard cap on how many keys a single key-addressed request may carry, bounding
# the work one request can ask for. A delete expands each key to every stored
# copy of its chunk and sends the result to the node in batches, so this does
# not need to match the node's delete batch.
MAX_KEYS_PER_REQUEST = 10_000


def _resolve_ipc_key_and_hashes(
    token_hasher: TokenHasher,
    model_name: str,
    world_size: int,
    token_ids: list[int],
    cache_salt: str,
    *,
    tags: Mapping[str, str] | None = None,
) -> tuple[IPCCacheServerKey, list[bytes]]:
    """Build the lookup-side IPC key and hash the complete chunks.

    Raises:
        ValueError: ``token_ids`` exceeds the per-request cap.
    """
    if len(token_ids) > MAX_TOKEN_IDS:
        raise ValueError(
            f"too many token_ids in a single request "
            f"(limit={MAX_TOKEN_IDS}, got={len(token_ids)})"
        )
    ipc_key = _lookup_ipc_key(model_name, world_size, token_ids, cache_salt, tags=tags)
    chunk_hashes = token_hasher.compute_chunk_hashes(list(token_ids))
    return ipc_key, chunk_hashes


def resolve_object_keys(
    token_hasher: TokenHasher,
    model_name: str,
    world_size: int,
    token_ids: list[int],
    cache_salt: str,
    *,
    tags: Mapping[str, str] | None = None,
) -> tuple[list[ObjectKey], int]:
    """Resolve a token sequence to the object keys of its complete chunks.

    Hashes ``token_ids`` and expands each complete chunk into one key per rank
    in object group ``0``. The ``token_hasher``
    must be configured to match the fleet's ``chunk_size`` / ``hash_algorithm``
    or the resolved keys will not match what the servers stored.

    Args:
        token_hasher: Hasher configured to match the fleet's chunk size and
            hash algorithm.
        model_name: Model whose rank fan-out to use.
        world_size: Tensor-parallel world size selecting the per-rank fan-out.
        token_ids: The token sequence to resolve.
        cache_salt: Per-tenant isolation salt.
        tags: Matching tags supplied when the content was stored.

    Returns:
        ``(obj_keys, chunk_count)``. ``obj_keys`` holds one key per (chunk, rank)
        and is empty (with ``chunk_count`` 0) when the sequence is shorter than
        one chunk.

    Raises:
        ValueError: ``token_ids`` exceeds the per-request cap, or a key field
            (e.g. ``cache_salt``) is invalid.
    """
    _, chunk_hashes = _resolve_ipc_key_and_hashes(
        token_hasher, model_name, world_size, token_ids, cache_salt, tags=tags
    )
    return resolve_chunk_object_keys(
        chunk_hashes,
        model_name,
        world_size,
        cache_salt,
        tags=tags,
    )


def resolve_chunk_object_keys(
    chunk_hashes: list[bytes],
    model_name: str,
    world_size: int,
    cache_salt: str,
    *,
    tags: Mapping[str, str] | None = None,
) -> tuple[list[ObjectKey], int]:
    """Resolve chunks named by hash to their object keys.

    The same keys :func:`resolve_object_keys` gives for a token sequence, for
    chunks named by hash, so a chunk in the middle of a prompt is reached
    without its prefix. Each distinct hash expands into one key per rank in
    object group ``0``, in first-named order.

    A ``world_size`` of ``0`` names each chunk in every parallel setup: it
    resolves to one key per chunk whose ``kv_rank`` is ``0``, which packs no
    world size and so is no stored key; the coordinator's pin table and key
    directory read it as every world size.

    Args:
        chunk_hashes: The chunks' hashes; a repeated hash counts once.
        model_name: Model whose rank fan-out to use.
        world_size: Tensor-parallel world size selecting the per-rank fan-out,
            or ``0`` for every world size.
        cache_salt: Per-tenant isolation salt.
        tags: Matching tags supplied when the content was stored.

    Returns:
        ``(obj_keys, chunk_count)``: one key per (distinct chunk, rank), or
        per distinct chunk for world size ``0``, and the number of distinct
        chunks.

    Raises:
        ValueError: A key field (e.g. ``cache_salt``) is invalid.
    """
    unique_chunk_hashes = list(dict.fromkeys(chunk_hashes))
    ipc_key = _lookup_ipc_key(model_name, world_size, [], cache_salt, tags=tags)
    if not unique_chunk_hashes:
        return [], 0
    if world_size == 0:
        every_setup = ObjectKeyIdentity(
            chunk_hash=b"",
            kv_rank=0,
            model_name=model_name,
            cache_salt=cache_salt,
            tags=tags or {},
        )
        return [
            ObjectKey(identity=replace(every_setup, chunk_hash=chunk_hash))
            for chunk_hash in unique_chunk_hashes
        ], len(unique_chunk_hashes)
    # The coordinator holds no KV registration, so this resolves the
    # per-rank layout of object group 0 only.
    obj_keys = ipc_key_to_object_keys(
        ipc_key, unique_chunk_hashes, [0], DEFAULT_SHARD_REPLICATION_DESC
    )[0]
    return obj_keys, len(unique_chunk_hashes)


def resolve_grouped_object_keys(
    token_hasher: TokenHasher,
    model_name: str,
    world_size: int,
    token_ids: list[int],
    cache_salt: str,
    group_layout_descs: dict[int, MemoryLayoutDesc],
    attn_desc: AttnWindowDesc,
    shard_replication_desc: ShardReplicationDesc,
    *,
    tags: Mapping[str, str] | None = None,
) -> tuple[list[GroupedObjectKeys], int]:
    """Resolve a token sequence to prefetch key rows for every object group.

    One :class:`GroupedObjectKeys` row per ``(object group, kv rank)``,
    group-major, each chunk-ordered and carrying its group's layout and
    attention window -- the rows the lookup path submits.

    Args:
        token_hasher: Hasher matching the fleet's chunk size and hash
            algorithm.
        model_name: Model whose rank fan-out to use.
        world_size: Tensor-parallel world size selecting the per-rank fan-out.
        token_ids: The token sequence to resolve.
        cache_salt: Per-tenant isolation salt.
        tags: Matching tags supplied when the content was stored.
        group_layout_descs: Each object group's memory layout.
        attn_desc: The model's per-group attention windows.
        shard_replication_desc: How each object group is spread over the kv ranks.

    Returns:
        ``(key_groups, chunk_count)``; ``([], 0)`` for a sub-chunk sequence.

    Raises:
        ValueError: ``token_ids`` exceeds the per-request cap, a key field
            (e.g. ``cache_salt``) is invalid, or a group has no layout.
    """
    ipc_key, chunk_hashes = _resolve_ipc_key_and_hashes(
        token_hasher, model_name, world_size, token_ids, cache_salt, tags=tags
    )
    if not chunk_hashes:
        return [], 0
    key_groups = ipc_key_to_grouped_object_keys(
        ipc_key,
        chunk_hashes,
        list(range(attn_desc.num_object_groups)),
        group_layout_descs,
        attn_desc,
        shard_replication_desc,
    )
    return key_groups, len(chunk_hashes)


def resolve_grouped_chunk_keys(
    chunk_hashes: list[bytes],
    model_name: str,
    world_size: int,
    cache_salt: str,
    group_layout_descs: dict[int, MemoryLayoutDesc],
    attn_desc: AttnWindowDesc,
    shard_replication_desc: ShardReplicationDesc,
    *,
    tags: Mapping[str, str] | None = None,
) -> list[GroupedObjectKeys]:
    """Expand named chunks into prefetch key rows for every object group.

    The same rows as :func:`resolve_grouped_object_keys`, for chunks named by
    hash instead of a token sequence, except that every row has no sliding
    window: the named chunks need not be a contiguous prefix, so there is no
    trailing window to keep, and each must be loaded.

    Args:
        chunk_hashes: The chunks to load, one row column each, in order.
        model_name: Model whose rank fan-out to use.
        world_size: Tensor-parallel world size selecting the per-rank fan-out.
        cache_salt: Per-tenant isolation salt.
        group_layout_descs: Each object group's memory layout.
        attn_desc: The model's per-group attention windows (only its group
            count is used).
        shard_replication_desc: How each object group is spread over the kv ranks.
        tags: Matching tags supplied when the content was stored.

    Returns:
        One row per ``(object group, kv rank)``, group-major, each holding
        ``len(chunk_hashes)`` keys.

    Raises:
        ValueError: A key field (e.g. ``cache_salt``) is invalid, or a group
            has no layout.
    """
    ipc_key = _lookup_ipc_key(model_name, world_size, [], cache_salt, tags=tags)
    rows = ipc_key_to_grouped_object_keys(
        ipc_key,
        chunk_hashes,
        list(range(attn_desc.num_object_groups)),
        group_layout_descs,
        attn_desc,
        shard_replication_desc,
    )
    return [
        replace(row, sliding_window_size=FULL_ATTENTION_WINDOW_CHUNKS) for row in rows
    ]


@dataclass(frozen=True)
class NamedChunks:
    """Chunks named by key, split into the parallel setup that stores them
    and their chunk hashes.

    Attributes:
        model_name: Model the keys belong to.
        world_size: World size packed in the keys' ``kv_rank``.
        cache_salt: Per-tenant isolation salt.
        tags: Matching tags supplied when the content was stored.
        chunk_hashes: The distinct chunk hashes, in first-named order.
    """

    model_name: str
    world_size: int
    cache_salt: str
    tags: dict[str, str]
    chunk_hashes: list[bytes]


def named_chunks_of_keys(keys: list[ObjectKey]) -> NamedChunks:
    """Split keys naming chunks of one parallel setup into that setup and the
    chunks' hashes, ignoring each key's object group and rank.

    A key-addressed prefetch warms one model's layout, so its keys must share
    one setup.

    Args:
        keys: Keys naming the chunks.

    Returns:
        The setup the keys share and their distinct chunk hashes.

    Raises:
        ValueError: The keys disagree on model, world size, cache salt or
            tags, or their ``kv_rank`` packs no world size.
    """
    setups = {
        (
            key.identity.model_name,
            ObjectKey.WorldSizeFromKVRank(key.identity.kv_rank),
            key.identity.cache_salt,
            tuple(key.identity.tags.items()),
        )
        for key in keys
    }
    if len(setups) != 1:
        raise ValueError(
            "keys must name the same model, world size, cache_salt and tags"
        )
    [(model_name, world_size, cache_salt, tags)] = setups
    if world_size < 1:
        raise ValueError(
            "the keys' kv_rank packs no world size; pass keys as the "
            "directory lists them"
        )
    return NamedChunks(
        model_name=model_name,
        world_size=world_size,
        cache_salt=cache_salt,
        tags=dict(tags),
        chunk_hashes=list(dict.fromkeys(key.chunk_hash for key in keys)),
    )


def _lookup_ipc_key(
    model_name: str,
    world_size: int,
    token_ids: list[int],
    cache_salt: str,
    *,
    tags: Mapping[str, str] | None = None,
) -> IPCCacheServerKey:
    """Build the lookup-side IPC key (no ``worker_id``, so every rank).

    Raises:
        ValueError: A key field (e.g. ``cache_salt``) is invalid.
    """
    matching_tags = ObjectKeyIdentity(
        chunk_hash=b"",
        kv_rank=0,
        model_name=model_name,
        cache_salt=cache_salt,
        tags=tags or {},
    ).tags
    return IPCCacheServerKey.from_token_ids(
        model_name=model_name,
        world_size=world_size,
        worker_id=None,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id="",
        cache_salt=cache_salt,
        request_configs={
            f"lmcache.tag.{name}": value for name, value in matching_tags.items()
        },
    )

# SPDX-License-Identifier: Apache-2.0
"""The release paths reuse the object keys the LOOKUP expanded.

``LookupModule.lookup`` expands every (object group, kv rank, chunk) triple of
a prompt to submit its prefetch. FREE_LOOKUP_LOCKS and END_SESSION address the
same objects, so they get the LOOKUP's expansion back instead of rebuilding
it, unless the key they carry names different objects.
"""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    ObjectKey,
    ObjectKeyIdentity,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 16
N_CHUNKS = 4
START_CHUNKS = 1
HASHES = [TokenHasher.hash_to_bytes(i) for i in range(1, N_CHUNKS + 1)]


def _make_key(
    start_chunks: int = 0, request_configs: dict | None = None
) -> IPCCacheServerKey:
    n_tokens = N_CHUNKS * CHUNK_SIZE
    return IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=range(n_tokens),
        start=start_chunks * CHUNK_SIZE,
        end=n_tokens,
        request_id="r",
        request_configs=request_configs,
    )


def _ctx(lookup_key: IPCCacheServerKey) -> MagicMock:
    """A mock engine context whose one session was looked up with ``lookup_key``."""
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.token_hasher.chunk_size = CHUNK_SIZE
    ctx.token_hasher.compute_packed_chunk_hashes.side_effect = (
        lambda packed, prefix_hash=None, start=0, end=None: HASHES[
            start // CHUNK_SIZE : (end or N_CHUNKS * CHUNK_SIZE) // CHUNK_SIZE
        ]
    )
    ctx.get_read_owners.return_value = None
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_attn_desc.return_value = MagicMock(
        num_chunks_in_sw=[-1], num_object_groups=1, world_size=1
    )
    ctx.layout_desc_registry.find_shard_replication_desc.return_value = (
        DEFAULT_SHARD_REPLICATION_DESC
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    session = MagicMock()
    session.extras = {}
    session.prefetch_hit_chunks = N_CHUNKS
    session.prefetch_locked_gids = ()
    session.get_hashes.return_value = list(range(1, N_CHUNKS + 1))
    session.lookup_ipc_key = lookup_key
    ctx.session_manager.get_or_create.return_value = session
    ctx.session_manager.get.return_value = session
    ctx.session_manager.remove.return_value = session
    return ctx


def _identities(keys: list[ObjectKey]) -> list[ObjectKeyIdentity]:
    return [k.identity for k in keys]


def _lookup_keys(ctx: MagicMock) -> list[ObjectKey]:
    """The object keys the LOOKUP handed to the prefetch, in chunk order."""
    spec = ctx.storage_manager.submit_prefetch_task.call_args.args[0]
    return [key for row in spec.key_groups for key in row.keys]


def test_free_after_lookup_releases_the_lookup_expansion() -> None:
    """FREE_LOOKUP_LOCKS releases the objects the LOOKUP locked, as the same
    key objects, although its key carries other scheduler hints."""
    lookup_key = _make_key(START_CHUNKS)
    ctx = _ctx(lookup_key)
    module = LookupModule(ctx)
    module.lookup(lookup_key, 1)
    locked = _lookup_keys(ctx)

    module.free_lookup_locks(
        _make_key(
            START_CHUNKS,
            request_configs={"lmcache.hint.engine_computed_tokens": 99},
        ),
        1,
    )

    (released,), kwargs = ctx.storage_manager.finish_read_prefetched.call_args
    assert kwargs == {"read_locks": 1, "l1_owners": None}
    assert _identities(released) == _identities(locked)
    assert all(any(key is lock for lock in locked) for key in released)


def test_end_session_touches_the_whole_prompt_from_the_lookup_expansion() -> None:
    """END_SESSION touches the covered prefix and the prefetched suffix with
    the key objects the LOOKUP built for each."""
    lookup_key = _make_key(START_CHUNKS)
    ctx = _ctx(lookup_key)
    module = LookupModule(ctx)
    module.lookup(lookup_key, 1)
    prefix = ctx.storage_manager.touch_cached_keys.call_args.args[0]
    suffix = _lookup_keys(ctx)

    module.end_session("r")

    touched = ctx.storage_manager.touch_cached_keys.call_args.args[0]
    expected = ipc_key_to_object_keys(
        lookup_key, HASHES, [0], DEFAULT_SHARD_REPLICATION_DESC
    )[0]
    assert _identities(touched) == _identities(expected)
    assert all(any(key is built for built in prefix + suffix) for key in touched)


def test_release_naming_other_objects_rebuilds_its_keys() -> None:
    """A release whose key carries other tags addresses other objects, so it
    gets keys built for that key rather than the LOOKUP's."""
    lookup_key = _make_key(START_CHUNKS)
    ctx = _ctx(lookup_key)
    module = LookupModule(ctx)
    module.lookup(lookup_key, 1)
    locked = _lookup_keys(ctx)

    tagged = _make_key(START_CHUNKS, request_configs={"lmcache.tag.tenant": "b"})
    module.free_lookup_locks(tagged, 1)

    (released,), _ = ctx.storage_manager.finish_read_prefetched.call_args
    expected = ipc_key_to_object_keys(
        tagged, HASHES[START_CHUNKS:], [0], DEFAULT_SHARD_REPLICATION_DESC
    )[0]
    assert _identities(released) == _identities(expected)
    assert all(key.identity.tags == {"tenant": "b"} for key in released)
    assert not any(key is lock for key in released for lock in locked)

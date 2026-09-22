# SPDX-License-Identifier: Apache-2.0
"""Engine-supplied identities must survive every MP key/session boundary."""

# Standard
from dataclasses import replace
from unittest.mock import Mock

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.session import Session, SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def supplied_key(**kwargs) -> IPCCacheServerKey:
    key = IPCCacheServerKey(
        model_name="opaque-pages",
        world_size=1,
        worker_id=0,
        token_ids=(),
        start=0,
        end=12,
        request_id="lookup-1",
        num_kv_readers=1,
        chunk_hashes=(b"a" * 32, b"b" * 32, b"c" * 32),
    )
    return replace(key, **kwargs)


def test_msgpack_roundtrip_and_worker_copy() -> None:
    key = supplied_key(cache_salt="tenant-1")
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(key), type=IPCCacheServerKey
    )
    assert decoded == key
    assert decoded.no_worker_id_version().chunk_hashes == key.chunk_hashes
    assert decoded.no_worker_id_version().worker_id is None


@pytest.mark.parametrize("start,end", [(1, 4), (0, 5), (0, 16)])
def test_precomputed_ranges_reject_wrong_geometry(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="aligned, bounded"):
        supplied_key(start=start, end=end).precomputed_range(4)


def test_precomputed_range_and_lookup_prefix() -> None:
    key = supplied_key(start=4, end=8)
    assert key.precomputed_range(4) == [b"b" * 32]
    assert key.precomputed_range(4, prefix=True) == [b"a" * 32, b"b" * 32]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_ids": (1, 2)},
        {"chunk_hashes": (b"short",)},
        {"start": -1},
        {"start": 8, "end": 4},
    ],
)
def test_invalid_precomputed_key_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        supplied_key(**kwargs)


def test_session_uses_external_keys_without_hashing_tokens() -> None:
    hasher = TokenHasher(chunk_size=4)
    session = Session("lookup-1", hasher)
    session.set_key(supplied_key())
    assert session.get_hashes(4, 8) == [b"b" * 32]
    assert session.get_hashes(0) == list(supplied_key().chunk_hashes)
    assert session.num_chunks_processed == 0
    with pytest.raises(ValueError, match="identity mode"):
        session.set_key(replace(supplied_key(), chunk_hashes=(), token_ids=(1,) * 12))


def test_precomputed_keys_are_part_of_failed_retrieve_ownership() -> None:
    session = Session("lookup-1", TokenHasher(chunk_size=4))
    key = supplied_key()
    session.set_key(key)
    session.begin_lookup(key.no_worker_id_version(), (-1,))
    session.record_prefetch_result(3, (0,))
    assert session.prepare_failed_retrieve_release(key) is not None
    other = replace(key, chunk_hashes=(b"d" * 32,) * 3)
    assert session.prepare_failed_retrieve_release(other) is None
    assert not session.claim_failed_retrieve_release(1, other, 1)


def test_object_resolution_and_lock_cleanup_use_identical_supplied_keys() -> None:
    hasher = TokenHasher(chunk_size=4)
    sessions = SessionManager(hasher, cleanup_interval=None)
    ctx = Mock(spec=MPCacheServerContext)
    ctx.session_manager = sessions
    ctx.token_hasher = hasher
    ctx.chunk_size = 4
    key = supplied_key(start=4, end=12)
    resolved = MPCacheServerContext.resolve_obj_keys(ctx, key, [0])[0]
    cleanup = resolve_prefetched_obj_keys(ctx, key, 3, (0,), (-1,))
    assert cleanup == resolved
    assert len(resolved) == 2
    assert [obj.chunk_hash for obj in resolved] == [b"b" * 32, b"c" * 32]
    sessions.close()

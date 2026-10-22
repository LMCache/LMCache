# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``l1_residency.query`` server module."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    build_server_module_router,
    load_server_module_classes,
)
from lmcache.v1.multiprocess.l1_residency import L1ResidencyModule
from lmcache.v1.multiprocess.protocols.l1_residency import (
    L1_RESIDENCY_QUERY_METHOD,
    L1ResidencyQuery,
    L1ResidencyReply,
    compute_probe_hash,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from tests.v1.distributed.utils import (
    OBJECT_BYTES,
    commit,
    cpu_storage,
    object_key,
    stage,
)

pytestmark = pytest.mark.no_shared_allocator


@pytest.fixture
def storage(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[L1Manager, StorageManager]]:
    l1, store = cpu_storage(monkeypatch)
    yield l1, store
    store.close()


def _module(
    store: StorageManager, hasher: TokenHasher | None = None
) -> L1ResidencyModule:
    ctx = SimpleNamespace(
        storage_manager=store,
        token_hasher=hasher or TokenHasher(),
        separate_object_groups=False,
    )
    return L1ResidencyModule(cast(Any, ctx))


def _query(module: L1ResidencyModule, query: L1ResidencyQuery) -> L1ResidencyReply:
    reply = module.query(msgspec.msgpack.encode(query))
    return msgspec.msgpack.decode(reply, type=L1ResidencyReply)


def test_module_loads_as_a_server_module_plugin() -> None:
    specs = [ExtServerModuleSpec(module_path="lmcache.v1.multiprocess.l1_residency")]
    assert load_server_module_classes(specs) == [(L1ResidencyModule, {})]

    ctx = MagicMock(separate_object_groups=False)
    router = build_server_module_router(MagicMock(), [L1ResidencyModule(ctx)])
    assert router is not None
    assert router.report_status()["server_module_plugins"]["methods"] == [
        L1_RESIDENCY_QUERY_METHOD
    ]


def test_rejects_separate_object_groups() -> None:
    with pytest.raises(ValueError, match="separate_object_groups=False"):
        L1ResidencyModule(MagicMock(separate_object_groups=True))


def test_counts_only_committed_objects_of_the_model(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, object_key(b"a"), object_key(b"b"))
    commit(l1, object_key(b"c"), temporary=True)
    stage(l1, object_key(b"d"))
    commit(l1, object_key(b"a", model_name="other"))

    reply = _query(
        _module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=1,
            chunk_hashes_by_salt={"": [b"a", b"c", b"d", b"e"]},
        ),
    )

    assert (reply.matched_objects, reply.matched_bytes) == (1, OBJECT_BYTES)
    assert (reply.model_objects, reply.model_bytes) == (2, 2 * OBJECT_BYTES)


def test_keys_fan_out_to_every_rank_and_keep_their_salt(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(
        l1,
        object_key(b"a", rank=0, world_size=2),
        object_key(b"a", rank=1, world_size=2),
    )
    commit(l1, object_key(b"b", world_size=2, cache_salt="u"))

    reply = _query(
        _module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=2,
            chunk_hashes_by_salt={"": [b"a", b"b"], "u": [b"a"]},
        ),
    )

    # b's only object is salted "u", and "u" was asked about a, not b.
    assert (reply.matched_objects, reply.model_objects) == (2, 3)


def test_reply_carries_the_server_probe_hash(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    _, store = storage
    hasher = TokenHasher(hash_algorithm="sha256_cbor")
    reply = _query(
        _module(store, hasher),
        L1ResidencyQuery(model_name="m", world_size=1, chunk_hashes_by_salt={}),
    )
    assert reply.probe_hash == compute_probe_hash(hasher)
    assert reply.probe_hash != compute_probe_hash(TokenHasher())
    assert (reply.matched_objects, reply.model_objects) == (0, 0)

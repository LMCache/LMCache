# SPDX-License-Identifier: Apache-2.0
"""Tests for unfull-chunk keys, lookup and lock release.

The lookup tests run the ``LookupModule`` handlers against a real L1-only
``StorageManager`` and real sessions; only the layout registry and the event
bus are mocked.
"""

# Standard
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.object_group_transfer import (
    split_unfull_chunk_block_ids,
)
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 16
MODEL = "model"
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([CHUNK_SIZE, 1, 8])], dtypes=[torch.uint8])


def test_unfull_chunk_hash_is_separate_from_full_chunk_hashes():
    hasher = TokenHasher(chunk_size=4)
    tokens = [1, 2, 3, 4, 5, 6, 7]
    prefix = hasher.hash_tokens(tokens[:4])

    unfull = hasher.hash_unfull_chunk(tokens[4:7], prefix)

    assert unfull != hasher.hash_tokens(tokens[4:7], prefix)
    assert unfull != hasher.hash_unfull_chunk(tokens[4:6], prefix)
    assert unfull == hasher.hash_unfull_chunk(tokens[4:7], prefix)


def test_session_unfull_chunk_hash_chains_from_full_chunks():
    hasher = TokenHasher(chunk_size=4)
    session = SessionManager(hasher, cleanup_interval=None).get_or_create("req")
    tokens = list(range(10))
    session.set_tokens(tokens)

    expected = hasher.hash_unfull_chunk(
        tokens[8:10], hasher.hash_tokens(tokens[4:8], hasher.hash_tokens(tokens[:4]))
    )
    assert session.get_unfull_chunk_hash(10) == expected
    assert session.get_unfull_chunk_hash(3) == hasher.hash_unfull_chunk(tokens[:3])
    with pytest.raises(ValueError, match="multiple"):
        session.get_unfull_chunk_hash(8)
    with pytest.raises(ValueError, match="exceeds"):
        session.get_unfull_chunk_hash(11)


def test_key_has_unfull_chunk_only_with_flag_and_unaligned_end():
    def key(end: int, include: bool) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=MODEL,
            world_size=1,
            worker_id=None,
            token_ids=tuple(range(40)),
            start=0,
            end=end,
            request_id="req",
            include_unfull_chunk=include,
        )

    assert key(35, include=True).has_unfull_chunk(CHUNK_SIZE)
    assert not key(32, include=True).has_unfull_chunk(CHUNK_SIZE)
    assert not key(35, include=False).has_unfull_chunk(CHUNK_SIZE)
    assert key(35, include=True).no_worker_id_version().include_unfull_chunk


def test_split_unfull_chunk_block_ids():
    full, unfull = split_unfull_chunk_block_ids(
        [[1, 2, 3, 4, 5], [7, 8, 9], [42, 42, 42]],
        blocks_per_chunk=[2, 1, 1],
        num_chunks=2,
        has_unfull_chunk=True,
    )
    assert full == [[1, 2, 3, 4], [7, 8], [42, 42]]
    assert unfull == [[5], [9], [42]]

    block_ids = [[1, 2]]
    assert split_unfull_chunk_block_ids(block_ids, [1], 2, False) == (block_ids, [])


@pytest.fixture
def lookup_module():
    storage_manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=16 * 1024 * 1024,
                    use_lazy=False,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
        )
    )
    hasher = TokenHasher(chunk_size=CHUNK_SIZE)
    ctx = mock.Mock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.storage_manager = storage_manager
    ctx.token_hasher = hasher
    ctx.session_manager = SessionManager(hasher, cleanup_interval=None)
    ctx.get_read_owners.return_value = None
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = LAYOUT
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1]
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: LAYOUT}
    module = LookupModule(ctx)
    yield module, storage_manager, hasher
    storage_manager.close()


def test_lookup_hits_the_unfull_chunk_and_free_deletes_a_temporary_one(
    lookup_module,
):
    module, storage_manager, hasher = lookup_module
    tokens = list(range(2 * CHUNK_SIZE + 5))
    lookup_end = len(tokens)
    _write(storage_manager, _chunk_keys(hasher, tokens, 2), is_temporary=False)
    unfull_keys = _unfull_keys(hasher, tokens, lookup_end)
    _write(storage_manager, unfull_keys, is_temporary=True)

    assert _lookup(module, tokens, lookup_end, request_id="r1") == 3

    module.free_lookup_locks(
        _key(tokens, 0, lookup_end, request_id="r1", include=True), tp_size=1
    )
    # Read once: releasing the lookup's last lock deleted the object.
    assert _lookup(module, tokens, lookup_end, request_id="r2") == 2


def test_lookup_ignores_the_unfull_chunk_without_the_flag(lookup_module):
    module, storage_manager, hasher = lookup_module
    tokens = list(range(CHUNK_SIZE + 3))
    _write(storage_manager, _chunk_keys(hasher, tokens, 1), is_temporary=False)
    _write(
        storage_manager, _unfull_keys(hasher, tokens, len(tokens)), is_temporary=True
    )

    assert _lookup(module, tokens, len(tokens), request_id="r1", include=False) == 1
    # Never locked, so still there for a lookup that asks for it.
    assert _lookup(module, tokens, len(tokens), request_id="r2") == 2


def test_unfull_chunk_needs_every_full_chunk_and_the_exact_end(lookup_module):
    module, storage_manager, hasher = lookup_module
    tokens = list(range(2 * CHUNK_SIZE + 5))
    # Chunk 0 only: the unfull chunk cannot extend a broken prefix.
    _write(storage_manager, _chunk_keys(hasher, tokens, 1), is_temporary=False)
    _write(
        storage_manager, _unfull_keys(hasher, tokens, len(tokens)), is_temporary=False
    )
    assert _lookup(module, tokens, len(tokens), request_id="r1") == 1

    _write(storage_manager, _chunk_keys(hasher, tokens, 2)[1:], is_temporary=False)
    # A different end addresses a different unfull chunk.
    assert _lookup(module, tokens, len(tokens) - 1, request_id="r2") == 2
    assert _lookup(module, tokens, len(tokens), request_id="r3") == 3


def _key(
    tokens: list[int], start: int, end: int, request_id: str, include: bool
) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=MODEL,
        world_size=1,
        worker_id=None,
        token_ids=tuple(tokens),
        start=start,
        end=end,
        request_id=request_id,
        num_kv_readers=1,
        include_unfull_chunk=include,
    )


def _chunk_keys(
    hasher: TokenHasher, tokens: list[int], num_chunks: int
) -> list[ObjectKey]:
    hashes = hasher.compute_chunk_hashes(tokens, end=num_chunks * CHUNK_SIZE)
    return ipc_key_to_object_keys(
        _key(tokens, 0, num_chunks * CHUNK_SIZE, "writer", False), hashes, [0]
    )[0]


def _unfull_keys(hasher: TokenHasher, tokens: list[int], end: int) -> list[ObjectKey]:
    full = end - end % CHUNK_SIZE
    prefix = None
    for start in range(0, full, CHUNK_SIZE):
        prefix = hasher.hash_tokens(tokens[start : start + CHUNK_SIZE], prefix)
    unfull_hash = TokenHasher.hash_to_bytes(
        hasher.hash_unfull_chunk(tokens[full:end], prefix)
    )
    return ipc_key_to_object_keys(
        _key(tokens, full, end, "writer", True), [unfull_hash], [0]
    )[0]


def _write(
    storage_manager: StorageManager, keys: list[ObjectKey], is_temporary: bool
) -> None:
    reserved = storage_manager.reserve_write(keys, LAYOUT, is_temporary=is_temporary)
    assert set(reserved) == set(keys)
    storage_manager.finish_write(keys)


def _lookup(
    module: LookupModule,
    tokens: list[int],
    end: int,
    request_id: str,
    include: bool = True,
) -> int:
    module.lookup(_key(tokens, 0, end, request_id, include), tp_size=1)
    hits = module.wait_prefetch_status(request_id, timeout=5.0)
    assert hits is not None
    return hits

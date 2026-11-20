# SPDX-License-Identifier: Apache-2.0
"""Lock release of aborted lookups at world size 2.

Runs the ``LookupModule`` handlers against a real L1-only ``StorageManager``
and real sessions, for per-rank objects and for a group replicated on both
ranks (one object read by two workers). Only the layout registry and the
event bus are mocked.
"""

# Standard
from unittest import mock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ShardReplicationDesc,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager

CHUNK_SIZE = 16
NUM_CHUNKS = 2
WORLD_SIZE = 2
MODEL = "model"
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([CHUNK_SIZE, 1, 8])], dtypes=[torch.uint8])
TOKENS = list(range(NUM_CHUNKS * CHUNK_SIZE))
REPLICATED = ShardReplicationDesc((WORLD_SIZE,), (1,))


def _key(request_id: str) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=MODEL,
        world_size=WORLD_SIZE,
        worker_id=None,
        token_bytes=pack_token_ids(TOKENS),
        start=0,
        end=len(TOKENS),
        request_id=request_id,
    )


def _object_keys(desc: ShardReplicationDesc) -> list[ObjectKey]:
    hashes = TokenHasher(chunk_size=CHUNK_SIZE).compute_chunk_hashes(
        TOKENS, end=len(TOKENS)
    )
    return ipc_key_to_object_keys(_key("writer"), hashes, [0], desc)[0]


def _read_locked(storage_manager: StorageManager) -> int:
    return sum(
        l1["read_locked_count"]
        for l1 in storage_manager.report_status()["l1_managers"].values()
    )


def _wait_read_locked(storage_manager: StorageManager, count: int) -> None:
    """Wait for a prefetch to take its locks without consuming its result."""
    deadline = time.monotonic() + 5.0
    while _read_locked(storage_manager) != count:
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.fixture(params=["per_rank", "replicated"])
def setup(request):
    desc = DEFAULT_SHARD_REPLICATION_DESC if request.param == "per_rank" else REPLICATED
    storage_manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=16 * 1024 * 1024, use_lazy=False
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
    ctx.layout_desc_registry.get_chunk_size.return_value = CHUNK_SIZE
    ctx.layout_desc_registry.get_token_hasher.return_value = hasher
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1]
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: LAYOUT}
    ctx.layout_desc_registry.find_shard_replication_desc.return_value = desc
    ctx.layout_desc_registry.find_group_unfull_chunk_layout_descs.return_value = {
        0: LAYOUT
    }
    keys = _object_keys(desc)
    reserved = storage_manager.reserve_write(keys, LAYOUT, is_temporary=False)
    assert set(reserved) == set(keys)
    storage_manager.finish_write(keys, l1_owners_of(reserved))
    yield LookupModule(ctx), storage_manager, keys, desc
    storage_manager.close()


def _consumed_lookup(module: LookupModule, request_id: str) -> None:
    module.lookup(_key(request_id), tp_size=WORLD_SIZE)
    assert module.wait_prefetch_status(request_id, timeout=5.0) == len(TOKENS)


def test_end_session_releases_an_unconsumed_lookup(setup):
    """Aborted before anyone polled its lookup: END_SESSION alone releases
    every object, including both readers' locks on a replicated one."""
    module, storage_manager, keys, desc = setup
    module.lookup(_key("r1"), tp_size=WORLD_SIZE)
    _wait_read_locked(storage_manager, len(keys))

    module.end_session("r1")

    assert _read_locked(storage_manager) == 0


def test_free_then_end_session_releases_a_consumed_lookup(setup):
    """Aborted after the hit reached the client: the connector's FREE, then
    END_SESSION, release every object."""
    module, storage_manager, keys, desc = setup
    _consumed_lookup(module, "r1")
    assert _read_locked(storage_manager) == len(keys)

    module.free_lookup_locks(_key("r1"), tp_size=WORLD_SIZE)
    module.end_session("r1")

    assert _read_locked(storage_manager) == 0


@pytest.mark.parametrize("aborted", ["unconsumed", "consumed"])
def test_abort_keeps_another_requests_locks_on_the_same_chunks(setup, aborted):
    """Releasing one aborted request's locks leaves a second request's locks
    on the same objects held until that request releases them."""
    module, storage_manager, keys, desc = setup
    if aborted == "unconsumed":
        module.lookup(_key("r1"), tp_size=WORLD_SIZE)
        _wait_read_locked(storage_manager, len(keys))
        _consumed_lookup(module, "r2")
    else:
        _consumed_lookup(module, "r1")
        _consumed_lookup(module, "r2")
        module.free_lookup_locks(_key("r1"), tp_size=WORLD_SIZE)
    module.end_session("r1")

    assert _read_locked(storage_manager) == len(keys)
    module.free_lookup_locks(_key("r2"), tp_size=WORLD_SIZE)
    module.end_session("r2")
    assert _read_locked(storage_manager) == 0

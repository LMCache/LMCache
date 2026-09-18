# SPDX-License-Identifier: Apache-2.0
"""Native PAGE aliases and sparse recurrent checkpoints share one safe endpoint.

Uses real group metadata, prefix hashes, the native bitmap fold, and the public
lookup handler. Only storage is an in-memory presence/lock double.
"""

# Standard
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap, EngineKVFormat
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_ranked
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import all_null_chunk_masks
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK = 16
TOKENS = tuple(range(CHUNK * 8))


class _PresenceStorage:
    """Emulate completed L1 prefetches with real fold-selected read locks."""

    def __init__(self, present: set[ObjectKey]) -> None:
        self.present = present
        self.locked: Counter[ObjectKey] = Counter()
        self.results: dict[int, Bitmap] = {}
        self.requests: list[PrefetchRequestSpec] = []

    def submit_prefetch_task(
        self,
        request: PrefetchRequestSpec,
        external_request_id: str,
    ) -> PrefetchHandle:
        """Retain exactly the objects needed at the longest joint endpoint."""
        presence = Bitmap(len(request.keys))
        for index, key in enumerate(request.keys):
            if key in self.present:
                presence.set(index)
        attn = request.attn_desc
        chunks = len(request.keys) // (attn.num_object_groups * attn.world_size)
        hit, retained = fold_unfold_ranked(
            presence, chunks, attn.world_size, attn.num_chunks_in_sw
        )
        task_id = len(self.requests)
        self.requests.append(request)
        self.results[task_id] = retained
        for index in retained.get_indices_list():
            self.locked[request.keys[index]] += request.num_kv_readers
        return PrefetchHandle(
            prefetch_request_id=task_id,
            external_request_id=external_request_id,
            l1_found_indices=tuple(retained.get_indices_list()),
            l1_hit_chunks=hit,
            total_requested_keys=len(request.keys),
            submit_time=0.0,
        )

    def query_prefetch_status(self, handle: PrefetchHandle) -> Bitmap:
        """Return the completed prefetch's retain mask."""
        return self.results[handle.prefetch_request_id]

    def finish_read_prefetched(self, keys: list[ObjectKey], read_locks: int) -> None:
        """Release only locks actually acquired by a lookup."""
        for key in keys:
            assert self.locked[key] >= read_locks
            self.locked[key] -= read_locks


def _groups() -> KVLayerGroupsManager:
    # State aliases truncate each checkpoint to its payload; the row stride
    # still includes neighboring bytes in the checkpoint allocation.
    page = torch.empty(32, 1, 64, dtype=torch.uint8)
    state = torch.empty(8, 1, 128, dtype=torch.uint8)
    aliases = [
        page,
        state.as_strided((8, 1, 32), (128, 32, 1)),
        state.as_strided((8, 1, 48), (128, 48, 1), storage_offset=48),
    ]
    for alias in aliases:
        normalized = attempt_permute_to_contiguous_view(alias)
        assert not isinstance(normalized, list)
        assert normalized.data_ptr() == alias.data_ptr()
        assert normalized.shape == alias.shape
    manager = KVLayerGroupsManager(
        aliases,
        [EngineKVFormat.NL_X_NB_BS_HS] * 3,
        [
            EngineGroupInfo(0, (0,), tokens_per_block=4, null_block_id=None),
            EngineGroupInfo(
                1,
                (1,),
                tokens_per_block=CHUNK,
                sw_size_tokens=CHUNK,
                recurrent_state=True,
                null_block_id=-1,
            ),
            EngineGroupInfo(
                2,
                (2,),
                tokens_per_block=CHUNK,
                sw_size_tokens=CHUNK,
                recurrent_state=True,
                null_block_id=-1,
            ),
        ],
        lmcache_tokens_per_chunk=CHUNK,
    )
    assert manager.get_attn_desc().num_chunks_in_sw == [-1, 1]
    assert manager.get_attn_desc().group_kinds == ("attention", "recurrent")
    assert [g.shape_desc.block_stride_elems for g in manager.kernel_groups] == [
        64,
        128,
        128,
    ]
    return manager


def _key(world_size: int = 1, tokens: tuple[int, ...] = TOKENS) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="native-state",
        world_size=world_size,
        worker_id=None,
        token_ids=tokens,
        start=0,
        end=len(tokens),
        request_id="native-request",
        num_kv_readers=1,
    )


def _lookup(
    checkpoint_ends: list[list[int]],
    *,
    query_tokens: tuple[int, ...] = TOKENS,
    query_end: int | None = None,
    missing_page_chunk: int | None = None,
) -> tuple[int, _PresenceStorage, LookupModule, IPCCacheServerKey]:
    manager = _groups()
    hasher = TokenHasher(CHUNK)
    hashes = hasher.compute_chunk_hashes(list(TOKENS))
    world_size = len(checkpoint_ends)
    key = _key(world_size, query_tokens)
    if query_end is not None:
        key = replace(key, end=query_end)
    present: set[ObjectKey] = set()
    for rank, endpoints in enumerate(checkpoint_ends):
        worker_key = replace(key, worker_id=rank)
        per_group = ipc_key_to_object_keys(worker_key, hashes, [0, 1])
        # Every native store contributes PAGE for its token range and only
        # one STATE object at its endpoint, exactly as the ATOM worker emits.
        for ordinal, endpoint in enumerate(endpoints):
            masks = all_null_chunk_masks(
                [
                    list(range(endpoint * 4)),
                    [-1] * (endpoint - 1) + [ordinal],
                    [-1] * (endpoint - 1) + [ordinal],
                ],
                manager.object_groups,
                [4, 1, 1],
                endpoint,
                [None, -1, -1],
            )
            for group_id, mask in enumerate(masks):
                present.update(
                    key
                    for key, skipped in zip(per_group[group_id], mask, strict=False)
                    if not skipped
                )
        # PAGE may advance beyond the latest available native checkpoint.
        present.update(per_group[0])
        if missing_page_chunk is not None:
            present.discard(per_group[0][missing_page_chunk])

    registry = LayoutDescRegistry()
    layouts = {
        gid: MemoryLayoutDesc(
            shapes=[
                torch.Size(
                    (
                        manager.get_slots_per_chunk_in_sw(kg),
                        manager.kernel_groups[kg].hidden_dim_size,
                    )
                )
                for kg in group.kernel_group_indices
            ],
            dtypes=[torch.uint8] * len(group.kernel_group_indices),
        )
        for gid, group in enumerate(manager.object_groups)
    }
    registry.register(
        "native-state", world_size, layouts[0], manager.get_attn_desc(), layouts
    )
    storage = _PresenceStorage(present)
    ctx = SimpleNamespace(
        storage_manager=storage,
        token_hasher=hasher,
        layout_desc_registry=registry,
        session_manager=SessionManager(hasher, cleanup_interval=None),
        event_bus=MagicMock(),
        chunk_size=CHUNK,
    )
    ctx.event_bus.has_subscribers.return_value = False
    module = LookupModule(cast(MPCacheServerContext, ctx))
    module.lookup(key, tp_size=world_size)
    hit = module.query_prefetch_status(key.request_id)
    assert hit is not None
    return hit, storage, module, key


@pytest.mark.parametrize(
    ("endpoints", "missing_page", "expected"),
    [
        ([[2, 5]], None, 5),
        ([[]], None, 0),
        ([[2, 5]], 3, 2),
        ([[2, 5], [2]], None, 2),
        ([[2, 5], [2, 5]], None, 5),
    ],
)
def test_sparse_state_selects_latest_joint_endpoint(
    endpoints: list[list[int]],
    missing_page: int | None,
    expected: int,
) -> None:
    hit, storage, module, key = _lookup(endpoints, missing_page_chunk=missing_page)
    assert hit == expected
    state_keys = [
        k for k, count in storage.locked.items() if count and k.object_group_id == 1
    ]
    assert len(state_keys) == (len(endpoints) if expected else 0)
    if expected:
        expected_hash = module.context.token_hasher.compute_chunk_hashes(list(TOKENS))[
            expected - 1
        ]
        assert {k.chunk_hash for k in state_keys} == {expected_hash}
    module.free_lookup_locks(key, tp_size=len(endpoints))
    assert not +storage.locked


def test_endpoint_limit_bounds_sparse_checkpoint_lookup() -> None:
    hit, storage, module, key = _lookup([[2, 5]], query_end=4 * CHUNK)
    assert hit == 2
    module.free_lookup_locks(key, tp_size=1)
    assert not +storage.locked


def test_leaving_last_prompt_token_uncached_uses_previous_checkpoint() -> None:
    hit, _storage, _module, _key_value = _lookup(
        [[2, 5]], query_tokens=TOKENS[: 5 * CHUNK - 1]
    )
    assert hit == 2

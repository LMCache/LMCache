# SPDX-License-Identifier: Apache-2.0
"""The node's token-addressed warm prefetch.

A prefetch arrives as tokens -- the coordinator forwards them rather than
keys -- so the node resolves them itself, and it is the only place that
knows every object group's layout and attention window. These cases pin
that it resolves all of them, the way the lookup path does, and that a
model it cannot resolve for is reported as unavailable rather than as a
bad request.
"""

# Standard
from dataclasses import dataclass, field

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.attn_window import AttnWindowDesc
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import MemoryLayoutDesc, ShardReplicationDesc, Tier
from lmcache.v1.multiprocess.cache_control import key_resolver
from lmcache.v1.multiprocess.cache_control.errors import (
    InvalidRequest,
    Unavailable,
)
from lmcache.v1.multiprocess.cache_control.prefetch_service import PrefetchService

CHUNK_SIZE = 4
CHUNKS = 2
TOKENS = list(range(CHUNKS * CHUNK_SIZE))


class _Handle:
    def __init__(self, total: int) -> None:
        self.total_requested_keys = total


@dataclass
class _StorageManager:
    """Records what a prefetch asked for; nothing is loaded."""

    rows: list = field(default_factory=list)

    def submit_prefetch_task(self, spec):
        self.rows = list(spec.key_groups)
        return _Handle(sum(len(row.keys) for row in self.rows))


class _Registry:
    """The node's per-(model, world size) layouts and windows."""

    def __init__(self, layouts, windows, attn_raises: bool = False) -> None:
        self._layouts = layouts
        self._windows = windows
        self._attn_raises = attn_raises
        self._token_hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")

    def find_group_layout_descs(self, model_name, world_size):
        return self._layouts

    def find_attn_desc(self, model_name, world_size):
        if self._attn_raises:
            raise ValueError("no attention-window descriptor registered")
        return self._windows

    def find_shard_replication_desc(self, model_name, world_size):
        return ShardReplicationDesc()

    def get_token_hasher(self, model_name, world_size):
        return self._token_hasher


@dataclass
class _Context:
    layout_desc_registry: _Registry


@dataclass
class _Engine:
    context: _Context
    storage_manager: _StorageManager


def _layout(width: int) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([width])], dtypes=[torch.float16])


def _service(registry: _Registry) -> tuple[PrefetchService, _StorageManager]:
    storage = _StorageManager()
    engine = _Engine(context=_Context(registry), storage_manager=storage)
    return PrefetchService(engine), storage


def _submit(service: PrefetchService, tokens=TOKENS, world_size: int = 2):
    return service.submit("m", world_size, tokens, "alice", Tier.L2, Tier.L1)


def test_a_prefetch_warms_every_object_group():
    """Each chunk is stored once per group, so a warm that reached group 0
    alone would load part of every chunk and report success. The rows the
    storage manager receives cover every group, with that group's own
    layout, on every rank."""
    layouts = {0: _layout(8), 1: _layout(16), 2: _layout(32)}
    windows = AttnWindowDesc(num_chunks_in_sw=[-1, 3, 1], world_size=2)
    service, storage = _service(_Registry(layouts, windows))

    reply = _submit(service)

    assert reply["chunks"] == CHUNKS
    assert reply["status"] == "submitted"
    assert [row.object_group_id for row in storage.rows] == [0, 0, 1, 1, 2, 2]
    for row in storage.rows:
        assert row.layout_desc == layouts[row.object_group_id]
        assert len(row.keys) == CHUNKS
    total = sum(len(row.keys) for row in storage.rows)
    assert total == CHUNKS * 2 * len(layouts)  # chunks * ranks * groups


def test_a_model_without_separate_groups_still_warms_its_one_group():
    """The ordinary case is one group; it must come out exactly as before."""
    windows = AttnWindowDesc(num_chunks_in_sw=[-1], world_size=2)
    service, storage = _service(_Registry({0: _layout(8)}, windows))

    _submit(service)

    assert [row.object_group_id for row in storage.rows] == [0, 0]


def test_a_model_not_registered_here_is_unavailable():
    service, _ = _service(_Registry(None, None))

    with pytest.raises(Unavailable, match="no layout registered"):
        _submit(service)


def test_a_model_unregistered_between_the_two_reads_is_still_unavailable():
    """Layouts found, then the attention windows gone: the model left while
    the request was being served. That is the same answer as never having
    been registered -- not a malformed request, which is what the
    resolver's own ValueError would have turned it into."""
    service, storage = _service(_Registry({0: _layout(8)}, None, attn_raises=True))

    with pytest.raises(Unavailable, match="no layout registered"):
        _submit(service)
    assert storage.rows == []


def test_a_sequence_past_the_token_cap_is_still_a_bad_request(monkeypatch):
    monkeypatch.setattr(key_resolver, "MAX_TOKEN_IDS", 4)
    windows = AttnWindowDesc(num_chunks_in_sw=[-1], world_size=1)
    service, _ = _service(_Registry({0: _layout(8)}, windows))

    with pytest.raises(InvalidRequest, match="too many token_ids"):
        _submit(service, world_size=1)


def _chunk_keys(chunk: int, world_size: int = 2, salt: str = "alice") -> list:
    """Every rank's group-0 key for one chunk of ``TOKENS``, as a caller
    holding keys (from the directory, say) would name it."""
    keys, _ = key_resolver.resolve_object_keys(
        TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3"),
        "m",
        world_size,
        TOKENS,
        salt,
    )
    return keys[chunk * world_size : (chunk + 1) * world_size]


def test_a_prefetch_by_key_warms_that_chunk_in_every_group_and_rank():
    """Keys name chunks: the named mid-prompt chunk is warmed in every
    object group and rank, whichever rank's key named it, and with no
    sliding window -- named chunks are not a contiguous prefix, and the
    full fetch refuses windowed rows."""
    layouts = {0: _layout(8), 1: _layout(16), 2: _layout(32)}
    windows = AttnWindowDesc(num_chunks_in_sw=[-1, 3, 1], world_size=2)
    service, storage = _service(_Registry(layouts, windows))
    second_chunk = _chunk_keys(1)

    reply = service.submit_keys(second_chunk, Tier.L2, Tier.L1)

    assert reply["chunks"] == 1
    assert reply["status"] == "submitted"
    assert [row.object_group_id for row in storage.rows] == [0, 0, 1, 1, 2, 2]
    assert {row.sliding_window_size for row in storage.rows} == {-1}
    assert {key.chunk_hash for row in storage.rows for key in row.keys} == {
        second_chunk[0].chunk_hash
    }
    for row in storage.rows:
        assert row.layout_desc == layouts[row.object_group_id]
        assert len(row.keys) == 1


def test_a_prefetch_by_keys_of_different_tenants_is_a_bad_request():
    """One prefetch warms one model's content for one tenant; keys that
    disagree on identity cannot share its layout lookup."""
    windows = AttnWindowDesc(num_chunks_in_sw=[-1], world_size=2)
    service, storage = _service(_Registry({0: _layout(8)}, windows))
    mixed = [_chunk_keys(0, salt="alice")[0], _chunk_keys(1, salt="bob")[0]]

    with pytest.raises(InvalidRequest, match="same model"):
        service.submit_keys(mixed, Tier.L2, Tier.L1)
    assert storage.rows == []

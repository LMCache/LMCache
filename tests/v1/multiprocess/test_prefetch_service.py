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
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, Tier
from lmcache.v1.multiprocess.cache_control import key_resolver
from lmcache.v1.multiprocess.cache_control.errors import (
    InvalidRequest,
    Unavailable,
)
from lmcache.v1.multiprocess.cache_control.prefetch_service import PrefetchService
from lmcache.v1.multiprocess.token_hasher import TokenHasher

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

    def find_group_layout_descs(self, model_name, world_size):
        return self._layouts

    def find_attn_desc(self, model_name, world_size):
        if self._attn_raises:
            raise ValueError("no attention-window descriptor registered")
        return self._windows


@dataclass
class _Context:
    layout_desc_registry: _Registry
    token_hasher: TokenHasher


@dataclass
class _Engine:
    context: _Context
    storage_manager: _StorageManager


def _layout(width: int) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([width])], dtypes=[torch.float16])


def _service(registry: _Registry) -> tuple[PrefetchService, _StorageManager]:
    storage = _StorageManager()
    hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")
    engine = _Engine(context=_Context(registry, hasher), storage_manager=storage)
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

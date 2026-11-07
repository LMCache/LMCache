# SPDX-License-Identifier: Apache-2.0
"""Tests for optional-window lookup rows and the QUERY_LOOKUP_WINDOW handler."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import DEFAULT_SHARD_REPLICATION_DESC, AttnWindowDesc
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.token_codec import pack_token_ids

REPLAY_DESC = AttnWindowDesc(num_chunks_in_sw=[-1, 1], optional_window=(False, True))


def _lookup_key() -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_bytes=pack_token_ids(range(32)),
        start=0,
        end=32,
        request_id="r",
    )


def _module(attn_desc: AttnWindowDesc, session: object) -> LookupModule:
    ctx = MagicMock()
    ctx.session_manager.get.return_value = session
    ctx.layout_desc_registry.find_attn_desc.return_value = attn_desc
    return LookupModule(ctx)


def _session(hit_chunks: int, locked_gids: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        lookup_ipc_key=_lookup_key(),
        prefetch_hit_chunks=hit_chunks,
        prefetch_locked_gids=locked_gids,
    )


def test_lookup_rows_carry_optional_flag():
    ctx = MagicMock()
    session = ctx.session_manager.get_or_create.return_value
    session.take_prefetch_release.return_value = None
    ctx.chunk_size = 16
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {
        0: MagicMock(),
        1: MagicMock(),
    }
    ctx.layout_desc_registry.find_attn_desc.return_value = REPLAY_DESC
    ctx.layout_desc_registry.find_shard_replication_desc.return_value = (
        DEFAULT_SHARD_REPLICATION_DESC
    )
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = [b"c0", b"c1"]

    LookupModule(ctx).lookup(_lookup_key(), tp_size=1)

    spec = ctx.storage_manager.submit_prefetch_task.call_args.args[0]
    assert [(row.object_group_id, row.optional_window) for row in spec.key_groups] == [
        (0, False),
        (1, True),
    ]


@pytest.mark.parametrize(
    "hit_chunks,locked_gids,expected",
    [
        pytest.param(2, (0, 1), True, id="window-locked"),
        pytest.param(2, (0,), False, id="window-missing"),
        pytest.param(0, (0, 1), False, id="no-hit"),
    ],
)
def test_query_lookup_window(hit_chunks, locked_gids, expected):
    module = _module(REPLAY_DESC, _session(hit_chunks, locked_gids))
    assert module.query_lookup_window("r") is expected


def test_query_lookup_window_false_without_optional_groups():
    desc = AttnWindowDesc(num_chunks_in_sw=[-1, 1])
    module = _module(desc, _session(2, (0, 1)))
    assert module.query_lookup_window("r") is False


def test_query_lookup_window_false_for_unknown_request():
    assert _module(REPLAY_DESC, None).query_lookup_window("r") is False

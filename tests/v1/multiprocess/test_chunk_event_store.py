# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from tests.v1.multiprocess.test_lmcache_driven_transfer_skip import (
    _make_module,
    _og,
)


def test_chunk_event_store_enqueues_each_chunk_separately(monkeypatch) -> None:
    module, _read_calls, transfer_calls = _make_module(
        monkeypatch, num_chunks=2, num_chunks_in_sw=[-1]
    )
    cache_context = module.get_and_touch_context_entry(1).cache_context
    cache_context.kv_layer_groups_manager.object_groups = [_og([0])]
    module.context.event_bus.has_subscribers.return_value = False
    module.context.storage_manager.reserve_write.side_effect = (
        lambda keys, layout, mode: {
            key: MagicMock(get_size=MagicMock(return_value=10)) for key in keys
        }
    )
    monkeypatch.setattr(mod, "get_layout_desc", lambda *args, **kwargs: object())
    monkeypatch.setattr(mod, "kept_blocks_per_chunk", lambda context, group: 1)

    _handle, chunk_events, ok = module.store_with_chunk_events(
        SimpleNamespace(request_id="req", worker_id=1, start=8, end=520),
        1,
        [[1, 2]],
        b"producer",
    )

    assert ok
    assert [group_id for group_id, _objects in transfer_calls] == [0, 0]
    assert all(len(objects) == 1 for _group_id, objects in transfer_calls)
    assert [(start, end) for _event, start, end in chunk_events] == [
        (8, 264),
        (264, 520),
    ]

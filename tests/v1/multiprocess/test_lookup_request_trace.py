# SPDX-License-Identifier: Apache-2.0
"""The ``MP_LOOKUP`` event carries the lookup key's prompt token ids."""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule

TOKEN_IDS = (11, 22, 33)


def test_lookup_event_carries_key_token_ids():
    """The handler attaches ``key.token_ids`` itself, by reference."""
    ctx = MagicMock()
    ctx.chunk_size = 256
    ctx.event_bus.has_subscribers.side_effect = (
        lambda event_type: event_type is EventType.MP_LOOKUP
    )
    ctx.layout_desc_registry.find.return_value = MagicMock(dtypes=[], shapes=[])
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {}
    ctx.token_hasher.compute_chunk_hashes.return_value = [b"h"]
    key = IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=TOKEN_IDS,
        start=0,
        end=0,
        request_id="req-1",
        num_kv_readers=1,
    )

    LookupModule(ctx).lookup(key, tp_size=1)

    (event,) = [
        call.args[0]
        for call in ctx.event_bus.publish.call_args_list
        if call.args[0].event_type is EventType.MP_LOOKUP
    ]
    assert event.metadata["token_ids"] is key.token_ids

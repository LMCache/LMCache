# SPDX-License-Identifier: Apache-2.0
"""LookupModule consults its KV load policy before prefetching.

The storage manager is mocked; a declined lookup must report a miss without
submitting a prefetch, so nothing is fetched from L2 and nothing is locked.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchHandle, PrefetchResult
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.kv_load_policy import (
    ENGINE_COMPUTED_TOKENS_HINT_KEY,
    KVLoadContext,
    KVLoadPolicy,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule

CHUNK_SIZE = 256
CHUNK_HASHES = [b"h0", b"h1", b"h2"]


class _RecordingPolicy(KVLoadPolicy):
    """Returns a fixed answer and records every context it is asked about."""

    def __init__(self, answer: bool) -> None:
        super().__init__({})
        self.answer = answer
        self.contexts: list[KVLoadContext] = []

    def should_load(self, ctx: KVLoadContext) -> bool:
        self.contexts.append(ctx)
        return self.answer


def _ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.token_hasher.compute_chunk_hashes.return_value = CHUNK_HASHES
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id="req-1",
        total_requested_keys=len(CHUNK_HASHES),
        submit_time=time.monotonic(),
    )
    ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
        hit_cells=[], l1_hit_cells=[], l2_hit_cells=[]
    )
    return ctx


def _key(request_configs: dict[str, Any] | None) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=tuple(range(len(CHUNK_HASHES) * CHUNK_SIZE)),
        start=0,
        end=len(CHUNK_HASHES) * CHUNK_SIZE,
        request_id="req-1",
        request_configs=request_configs,
        num_kv_readers=1,
    )


def _end_metadata(ctx: MagicMock) -> dict:
    for call in ctx.event_bus.publish.call_args_list:
        event = call.args[0]
        if event.event_type is EventType.MP_LOOKUP_PREFETCH_END:
            return event.metadata
    raise AssertionError("no MP_LOOKUP_PREFETCH_END event was published")


def test_declined_lookup_reports_miss_without_prefetch() -> None:
    """A recompute decision skips the prefetch and reports zero hit chunks."""
    ctx = _ctx()
    policy = _RecordingPolicy(answer=False)
    module = LookupModule(ctx, policy)

    module.lookup(_key({ENGINE_COMPUTED_TOKENS_HINT_KEY: 300}), tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_not_called()
    assert module.query_prefetch_status("req-1") == 0
    metadata = _end_metadata(ctx)
    assert metadata["early_exit_reason"] == "load_policy_recompute"
    # Declined tokens still count as requested, so hit rate shows them unserved.
    assert metadata["requested_tokens"] == len(CHUNK_HASHES) * CHUNK_SIZE
    assert metadata["hit_tokens"] == 0


def test_policy_sees_lookup_and_engine_hint() -> None:
    """The policy receives the lookup length and the connector's hint."""
    ctx = _ctx()
    policy = _RecordingPolicy(answer=True)
    module = LookupModule(ctx, policy)
    configs = {ENGINE_COMPUTED_TOKENS_HINT_KEY: 300, "lmcache.skip_save": True}

    module.lookup(_key(configs), tp_size=1)

    ctx.storage_manager.submit_prefetch_task.assert_called_once()
    assert policy.contexts == [
        KVLoadContext(
            request_id="req-1",
            model_name="m",
            chunk_size=CHUNK_SIZE,
            num_lookup_tokens=len(CHUNK_HASHES) * CHUNK_SIZE,
            engine_computed_tokens=300,
            request_configs=configs,
        )
    ]


@pytest.mark.parametrize(
    "request_configs, prefetched",
    [
        (None, True),
        ({"lmcache.skip_load": False}, True),
        ({"lmcache.skip_load": True}, False),
    ],
)
def test_default_policy_honors_skip_load(
    request_configs: dict[str, Any] | None, prefetched: bool
) -> None:
    """Without an explicit policy, only lmcache.skip_load declines a lookup."""
    ctx = _ctx()
    module = LookupModule(ctx)

    module.lookup(_key(request_configs), tp_size=1)

    assert ctx.storage_manager.submit_prefetch_task.called is prefetched

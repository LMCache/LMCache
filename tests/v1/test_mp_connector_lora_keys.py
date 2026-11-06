# SPDX-License-Identifier: Apache-2.0
"""Tests for LoRA cache tags in the MP connector.

The connector carries a request's LoRA adapter as the ``lmcache.tag.lora``
request config, which the MP server makes part of every object key. The
tracker derives the tag from the request's ``lora_request``; these tests
exercise the public tracker/metadata interfaces of ``lmcache_mp_connector``
and the lazy offload manager.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import hashlib

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadManager,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.utils import LORA_TAG_KEY  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator

ADAPTER_A = LoRARequest("adapter-a", 1, "/adapters/a")
ADAPTER_B = LoRARequest("adapter-b", 2, "/adapters/b")


@dataclass
class _FakeSamplingParams:
    extra_args: dict[str, object] | None = None


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(
        self,
        prompt_token_ids: list[int],
        cache_salt: str = "",
        lora_request: LoRARequest | None = None,
        kv_transfer_params: dict[str, object] | None = None,
    ):
        self.request_id = "req-0"
        self.resumable = False
        self.status = RequestStatus.WAITING
        self.cache_salt = cache_salt
        self.lora_request = lora_request
        self.prompt_token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(prompt_token_ids)
        self.all_token_ids = ConstantList(list(prompt_token_ids))
        self.mm_features: list[object] = []
        extra_args: dict[str, object] | None = None
        if kv_transfer_params is not None:
            extra_args = {"kv_transfer_params": kv_transfer_params}
        self.sampling_params = _FakeSamplingParams(extra_args)


class _NoopCompletionTracker:
    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return False


def _configs(request: _FakeRequest) -> dict[str, object] | None:
    return LMCacheMPRequestTracker(request).request_configs


def _documented_tag(lora_request: LoRARequest) -> str:
    payload = f"{lora_request.lora_name}\0{lora_request.lora_path}"
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _scheduler_connector(scheduler_adapter: MagicMock) -> LMCacheMPConnector:
    """A scheduler-side connector whose server calls go to the mock."""
    scheduler_adapter.lmcache_tokens_per_chunk = 64
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = scheduler_adapter
    connector._reserve_last_token_for_lookup = False
    connector.lazy_offload = False
    connector._hit_alignment_tokens = 64
    connector._connector_stats = MagicMock()
    connector._mamba_relocation_window = 0
    return connector


def test_base_model_request_gets_no_lora_tag():
    assert _configs(_FakeRequest([1, 2, 3])) is None
    client_configs = {"lmcache.max_offload_tokens": 64}
    assert _configs(_FakeRequest([1, 2, 3], kv_transfer_params=client_configs)) == (
        client_configs
    )


def test_lora_tag_is_the_documented_digest():
    # The value is part of every stored key: changing how it is derived
    # moves adapter entries to new keys.
    assert _documented_tag(ADAPTER_A) == "77b2860add3cb7ed806a43981c93a529"
    assert _configs(_FakeRequest([1, 2, 3], lora_request=ADAPTER_A)) == {
        LORA_TAG_KEY: _documented_tag(ADAPTER_A)
    }


def test_adapters_and_base_model_never_share_a_tag():
    tags = {
        (_configs(_FakeRequest([1, 2, 3], lora_request=lora)) or {}).get(LORA_TAG_KEY)
        for lora in (None, ADAPTER_A, ADAPTER_B)
    }
    assert len(tags) == 3


def test_adapter_identity_is_name_and_path_not_int_id():
    same_adapter_other_id = LoRARequest("adapter-a", 7, "/adapters/a")
    same_name_other_path = LoRARequest("adapter-a", 1, "/adapters/a-v2")
    configs = _configs(_FakeRequest([1, 2, 3], lora_request=ADAPTER_A))
    assert _configs(_FakeRequest([1, 2, 3], lora_request=same_adapter_other_id)) == (
        configs
    )
    assert _configs(_FakeRequest([1, 2, 3], lora_request=same_name_other_path)) != (
        configs
    )


def test_lora_request_keeps_its_cache_salt():
    tracker = LMCacheMPRequestTracker(
        _FakeRequest([1, 2, 3], cache_salt="user-a", lora_request=ADAPTER_A)
    )
    assert tracker.cache_salt == "user-a"


def test_client_sent_lora_tag_is_dropped():
    spoofed = {LORA_TAG_KEY: _documented_tag(ADAPTER_B), "lmcache.tag.team": "t1"}
    assert _configs(_FakeRequest([1, 2, 3], kv_transfer_params=spoofed)) == {
        "lmcache.tag.team": "t1"
    }
    assert _configs(
        _FakeRequest([1, 2, 3], lora_request=ADAPTER_A, kv_transfer_params=spoofed)
    ) == {"lmcache.tag.team": "t1", LORA_TAG_KEY: _documented_tag(ADAPTER_A)}


def test_lookup_carries_the_lora_tag():
    request = _FakeRequest(
        list(range(128)), cache_salt="user-a", lora_request=ADAPTER_A
    )
    scheduler_adapter = MagicMock()
    scheduler_adapter.check_lookup_result.return_value = None
    connector = _scheduler_connector(scheduler_adapter)

    assert connector.get_num_new_matched_tokens(request, 16) == (None, True)
    lookup = scheduler_adapter.maybe_submit_lookup_request.call_args.kwargs
    assert lookup["cache_salt"] == "user-a"
    assert lookup["request_configs"] == {
        LORA_TAG_KEY: _documented_tag(ADAPTER_A),
        "lmcache.hint.engine_computed_tokens": 16,
    }


def test_eager_lookup_carries_the_lora_tag():
    request = _FakeRequest([1, 2, 3], lora_request=ADAPTER_A)
    tracker = LMCacheMPRequestTracker(request)
    scheduler_adapter = MagicMock()
    connector = SimpleNamespace(
        role=KVConnectorRole.SCHEDULER,
        _eager_prefetch=True,
        _reserve_last_token_for_lookup=False,
        scheduler_adapter=scheduler_adapter,
        _get_or_create_request_tracker=MagicMock(return_value=tracker),
    )

    LMCacheMPConnector.on_new_request(connector, request)

    scheduler_adapter.maybe_submit_lookup_request.assert_called_once_with(
        request.request_id,
        packed_token_ids=tracker.packed_token_ids(),
        cache_salt="",
        request_configs={LORA_TAG_KEY: _documented_tag(ADAPTER_A)},
        reserve_last_token=False,
    )


def test_lock_release_carries_the_lora_tag():
    request = _FakeRequest(
        list(range(128)), cache_salt="user-a", lora_request=ADAPTER_A
    )
    scheduler_adapter = MagicMock()
    scheduler_adapter.check_lookup_result.return_value = 64
    connector = _scheduler_connector(scheduler_adapter)
    assert connector.get_num_new_matched_tokens(request, 0) == (64, True)

    # Admitted without a load, so every chunk the lookup locked is released.
    connector.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([0, 1],)), 0
    )

    release = scheduler_adapter.free_lookup_locks.call_args.kwargs
    assert (release["start"], release["end"]) == (0, 64)
    assert release["cache_salt"] == "user-a"
    assert release["request_configs"] == {LORA_TAG_KEY: _documented_tag(ADAPTER_A)}


def test_store_metadata_carries_the_lora_tag():
    tracker = LMCacheMPRequestTracker(
        _FakeRequest(list(range(8)), cache_salt="user-a", lora_request=ADAPTER_B)
    )
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_scheduled_tokens = 8

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
    )

    assert metadata is not None
    assert metadata.cache_salt == "user-a"
    assert metadata.request_configs == {LORA_TAG_KEY: _documented_tag(ADAPTER_B)}


def test_retrieve_metadata_carries_the_lora_tag():
    tracker = LMCacheMPRequestTracker(
        _FakeRequest(list(range(8)), cache_salt="user-a", lora_request=ADAPTER_B)
    )
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_lmcache_hit_tokens = 8
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
    )

    assert metadata is not None
    assert metadata.cache_salt == "user-a"
    assert metadata.request_configs == {LORA_TAG_KEY: _documented_tag(ADAPTER_B)}


def test_lazy_offload_keeps_request_configs_when_coalescing():
    tracker = LMCacheMPRequestTracker(
        _FakeRequest(
            list(range(8)),
            cache_salt="user-a",
            lora_request=ADAPTER_A,
            kv_transfer_params={"lmcache.priority": "high"},
        )
    )
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "FIFO",
            "lmcache.mp.lazy_offload_threshold": 1,
        },
        [4],
        _NoopCompletionTracker(),
    )
    pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=bytes([i])) for i in range(2)],
        touch=MagicMock(),
        free_blocks=MagicMock(),
    )
    manager.bind_block_pool(cast(Any, pool))
    manager.on_request_arrived(tracker.request_id)
    # Two prefill steps of one block each are buffered as two store ops.
    for num_tokens in (4, 8):
        tracker.allocated_block_ids = {0: list(range(num_tokens // 4))}
        tracker.num_scheduled_tokens = num_tokens
        metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
        )
        assert metadata is not None
        manager.add_store_candidate(metadata)
    manager.on_request_finished(tracker.request_id)

    actions = manager.on_scheduler_step(
        cast(
            Any,
            SimpleNamespace(
                total_num_scheduled_tokens=1,
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
            ),
        )
    )

    [store] = actions.stores_to_submit
    assert (store.op.start, store.op.end, store.op.block_ids) == (0, 8, [[0, 1]])
    assert store.cache_salt == "user-a"
    assert store.request_configs == {
        "lmcache.priority": "high",
        LORA_TAG_KEY: _documented_tag(ADAPTER_A),
    }

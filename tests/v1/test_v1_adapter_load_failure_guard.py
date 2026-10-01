# SPDX-License-Identifier: Apache-2.0
"""A failed KV load from LMCache must not be promised again.

Under GPU KV-cache pressure, a request can be preempted and resumed many
times. If its retrieve returns fewer tokens than the lookup promised, vLLM
recomputes the missing blocks (``kv_load_failure_policy=recompute``), but the
scheduler side was never told, so the next lookup promised the same chunks
again. One request repeated this ~360 times in 9 minutes and stalled the whole
engine.

This locks in:

1. ``record_load_failures`` marks only the requests that own a failed block.
2. ``get_num_new_matched_tokens`` returns 0 for a marked request without a new
   lookup, and records a no-load ``LoadSpec`` (a resumed request needs one).
3. ``request_finished`` drops the mark.
4. ``LMCacheEngine.lookup`` keeps the pins of an earlier lookup with the same
   id, so ``lookup_unpin`` releases all of them.
"""

# Standard
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (  # noqa: E402
    LMCacheConnectorV1Impl,
)
from lmcache.utils import CacheEngineKey  # noqa: E402
from lmcache.v1.cache_engine import LMCacheEngine  # noqa: E402


class _FakeLookupClient:
    def __init__(self, hit_tokens: int) -> None:
        self.hit_tokens = hit_tokens
        self.lookups: list[str] = []

    def lookup_cache(self, lookup_id: str) -> int:
        return -1

    def lookup(self, token_ids, lookup_id: str, request_configs=None) -> int:
        self.lookups.append(lookup_id)
        return self.hit_tokens


def _make_scheduler_connector(hit_tokens: int) -> LMCacheConnectorV1Impl:
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.kv_role = "kv_both"
    # ``lookup_client`` is a read-only property backed by ``self._manager``.
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lookup_client=_FakeLookupClient(hit_tokens)
    )
    connector.load_specs = {}
    connector._load_failed_req_ids = set()
    connector._request_trackers = {
        "req-a": SimpleNamespace(allocated_block_ids=[10, 11, 12]),
        "req-b": SimpleNamespace(allocated_block_ids=[20, 21]),
    }
    connector._requests_priority = {}
    connector.skip_last_n_tokens = 0
    connector._max_tokens_per_load = 0
    connector._lmcache_chunk_size = 256
    connector.config = SimpleNamespace(
        min_retrieve_tokens=0,
        get_extra_config_value=lambda key, default=None: default,
    )
    connector.use_layerwise = False
    return connector


def _make_request(req_id: str, num_tokens: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        all_token_ids=list(range(num_tokens)),
        prompt_token_ids=list(range(num_tokens)),
        num_tokens=num_tokens,
        sampling_params=None,
        mm_features=None,
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params=None,
    )


def test_record_load_failures_marks_only_owners_of_failed_blocks() -> None:
    connector = _make_scheduler_connector(hit_tokens=512)

    connector.record_load_failures({11, 99})

    assert connector._load_failed_req_ids == {"req-a"}


def test_failed_load_is_not_promised_again() -> None:
    connector = _make_scheduler_connector(hit_tokens=512)
    connector.record_load_failures({12})

    num_external = connector.get_num_new_matched_tokens(
        _make_request("req-a"), num_computed_tokens=448
    )

    assert num_external == 0
    assert connector.lookup_client.lookups == []
    load_spec = connector.load_specs["req-a"]
    assert load_spec.vllm_cached_tokens == 448
    assert load_spec.lmcache_cached_tokens == 448
    assert load_spec.can_load is False


def test_other_requests_still_load() -> None:
    connector = _make_scheduler_connector(hit_tokens=512)
    connector.record_load_failures({12})

    num_external = connector.get_num_new_matched_tokens(
        _make_request("req-b"), num_computed_tokens=256
    )

    assert num_external == 256
    assert connector.lookup_client.lookups == ["req-b"]


def test_request_finished_drops_the_mark() -> None:
    connector = _make_scheduler_connector(hit_tokens=512)
    connector.record_load_failures({10})

    connector.request_finished(_make_request("req-a"), block_ids=[])

    assert "req-a" not in connector._load_failed_req_ids


def test_lookup_keeps_pins_of_an_earlier_lookup_with_the_same_id() -> None:
    engine = MagicMock()
    engine.is_healthy.return_value = True
    engine.use_layerwise = False
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.lookup_pins = defaultdict(lambda: defaultdict(list))
    k0 = CacheEngineKey("test", 1, 0, 0, torch.bfloat16)
    k1 = CacheEngineKey("test", 1, 0, 1, torch.bfloat16)
    engine.token_database.process_tokens.side_effect = [
        [(0, 256, k0)],
        [(0, 256, k0), (256, 512, k1)],
    ]
    engine.storage_manager.batched_contains.side_effect = [
        (1, {"LocalCPUBackend": [k0]}),
        (2, {"LocalCPUBackend": [k0, k1]}),
    ]

    LMCacheEngine.lookup(engine, tokens=list(range(256)), lookup_id="req", pin=True)
    LMCacheEngine.lookup(engine, tokens=list(range(512)), lookup_id="req", pin=True)

    # Every pin taken by contains() is recorded, so lookup_unpin releases all.
    assert engine.lookup_pins["req"]["LocalCPUBackend"] == [k0, k0, k1]

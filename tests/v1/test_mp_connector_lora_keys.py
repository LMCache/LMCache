# SPDX-License-Identifier: Apache-2.0
"""Tests for LoRA-aware cache salts in the MP connector.

The MP server keys KV by token ids plus ``cache_salt`` and treats request
configs as metadata, so the adapter identity has to travel inside the salt.
The tracker derives it from the request's ``lora_request``; these tests
exercise the public tracker/metadata interfaces of ``lmcache_mp_connector``.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
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
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402

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
        self.sampling_params = _FakeSamplingParams()


def _salt(request: _FakeRequest) -> str:
    return LMCacheMPRequestTracker(request).cache_salt


def _documented_digest(cache_salt: str, lora_request: LoRARequest) -> str:
    payload = f"{cache_salt}\0{lora_request.lora_name}\0{lora_request.lora_path}"
    return hashlib.sha256(payload.encode()).hexdigest()


def test_base_model_request_keeps_vllm_cache_salt():
    assert _salt(_FakeRequest([1, 2, 3])) == ""
    assert _salt(_FakeRequest([1, 2, 3], cache_salt="user-a")) == "user-a"


def test_lora_request_salt_is_the_documented_digest():
    request = _FakeRequest([1, 2, 3], cache_salt="user-a", lora_request=ADAPTER_A)
    assert _salt(request) == _documented_digest("user-a", ADAPTER_A)
    assert _salt(_FakeRequest([1, 2, 3], lora_request=ADAPTER_A)) == (
        _documented_digest("", ADAPTER_A)
    )


def test_adapters_and_base_model_never_share_a_salt():
    salts = {
        _salt(_FakeRequest([1, 2, 3])),
        _salt(_FakeRequest([1, 2, 3], lora_request=ADAPTER_A)),
        _salt(_FakeRequest([1, 2, 3], lora_request=ADAPTER_B)),
    }
    assert len(salts) == 3


def test_user_salt_still_separates_tenants_on_the_same_adapter():
    salts = {
        _salt(_FakeRequest([1, 2, 3], cache_salt="user-a", lora_request=ADAPTER_A)),
        _salt(_FakeRequest([1, 2, 3], cache_salt="user-b", lora_request=ADAPTER_A)),
        _salt(_FakeRequest([1, 2, 3], lora_request=ADAPTER_A)),
    }
    assert len(salts) == 3


def test_adapter_identity_is_name_and_path_not_int_id():
    same_adapter_other_id = LoRARequest("adapter-a", 7, "/adapters/a")
    same_name_other_path = LoRARequest("adapter-a", 1, "/adapters/a-v2")
    salt = _salt(_FakeRequest([1, 2, 3], lora_request=ADAPTER_A))
    assert _salt(_FakeRequest([1, 2, 3], lora_request=same_adapter_other_id)) == salt
    assert _salt(_FakeRequest([1, 2, 3], lora_request=same_name_other_path)) != salt


def test_lora_salt_satisfies_object_key_invariants():
    hostile_name = LoRARequest("team@v2/legal", 3, "/adapters/legal v2")
    salt = _salt(
        _FakeRequest([1, 2, 3], cache_salt="user-a", lora_request=hostile_name)
    )
    ObjectKey(
        chunk_hash=b"\x00" * 32,
        model_name="model",
        kv_rank=0,
        object_group_id=0,
        cache_salt=salt,
    )
    assert len(salt) == 64


def test_lookup_uses_the_adapter_salt():
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
        token_ids=[1, 2, 3],
        cache_salt=_documented_digest("", ADAPTER_A),
        request_configs=None,
        reserve_last_token=False,
    )


def test_store_metadata_carries_the_adapter_salt():
    tracker = LMCacheMPRequestTracker(
        _FakeRequest(list(range(8)), cache_salt="user-a", lora_request=ADAPTER_B)
    )
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_scheduled_tokens = 8

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
    )

    assert metadata is not None
    assert metadata.cache_salt == _documented_digest("user-a", ADAPTER_B)


def test_retrieve_metadata_carries_the_adapter_salt():
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
    assert metadata.cache_salt == _documented_digest("user-a", ADAPTER_B)

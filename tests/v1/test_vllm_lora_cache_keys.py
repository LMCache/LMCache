# SPDX-License-Identifier: Apache-2.0
"""Check that the vLLM V1 connector keeps each LoRA adapter's cache keys separate."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock
import itertools

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.lora.request import LoRARequest  # noqa: E402

# First Party
from lmcache.integration.vllm.utils import add_lora_tag
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
)
from lmcache.utils import LORA_TAG_KEY, CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase

# Local
from .utils import dumb_metadata, generate_tokens

ADAPTER_A = LoRARequest("adapter-a", 1, "/adapters/a")
ADAPTER_B = LoRARequest("adapter-b", 2, "/adapters/b")
ADAPTER_A_CONFIGS = {LORA_TAG_KEY: "adapter-a"}


class _FakeParent:
    def __init__(self, metadata: LMCacheConnectorMetadata) -> None:
        self._connector_metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self._connector_metadata


class _FakeEngine:
    """Records the ``request_configs`` each layerwise engine or blender call gets."""

    def __init__(self) -> None:
        self.request_configs: list[object] = []

    def store_layer(self, token_ids: list[int], **kwargs: object) -> Iterator[None]:
        self.request_configs.append(kwargs["request_configs"])
        return itertools.repeat(None)

    def retrieve_layer(
        self, tokens: torch.Tensor, mask: torch.Tensor, **kwargs: object
    ) -> Iterator[None]:
        self.request_configs.append(kwargs["request_configs"])
        return itertools.repeat(None)

    def blend(self, tokens: torch.Tensor, mask: torch.Tensor, **kwargs: object) -> None:
        self.request_configs.append(kwargs["request_configs"])


class _FakeLookupClient:
    def __init__(self) -> None:
        self.request_configs: list[dict[str, object] | None] = []

    def lookup_cache(self, lookup_id: str) -> int:
        return -1

    def lookup(
        self,
        token_ids: list[int],
        lookup_id: str,
        request_configs: dict[str, object] | None,
    ) -> int:
        self.request_configs.append(request_configs)
        return 0


def _make_req(
    save_spec: SaveSpec | None = None, load_spec: LoadSpec | None = None
) -> ReqMeta:
    return ReqMeta(
        req_id="req-1",
        token_ids=list(range(16)),
        slot_mapping=torch.arange(16),
        save_spec=save_spec,
        load_spec=load_spec,
        request_configs=ADAPTER_A_CONFIGS,
    )


def _make_connector(req: ReqMeta) -> tuple[LMCacheConnectorV1Impl, _FakeEngine]:
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(LMCacheConnectorMetadata(requests=[req]))
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine
    )
    connector.kv_role = "kv_both"
    connector.use_layerwise = True
    connector.enable_blending = False
    connector.blender = engine
    connector.device = "cpu"
    connector._lmcache_chunk_size = 16
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._layerwise_save_storers = {}
    connector._stats_monitor = MagicMock()
    return connector, engine


def test_add_lora_tag_keeps_base_model_configs() -> None:
    request_configs: dict[str, object] = {"lmcache.tag.tenant": "tenant-a"}

    assert add_lora_tag(None, None) is None
    assert add_lora_tag(request_configs, None) is request_configs


def test_add_lora_tag_adds_adapter_and_keeps_existing_configs() -> None:
    request_configs: dict[str, object] = {"lmcache.tag.tenant": "tenant-a"}

    result = add_lora_tag(request_configs, ADAPTER_A)

    assert result == {"lmcache.tag.tenant": "tenant-a", LORA_TAG_KEY: "adapter-a"}
    assert request_configs == {"lmcache.tag.tenant": "tenant-a"}
    assert add_lora_tag(None, ADAPTER_A) == ADAPTER_A_CONFIGS


def test_lora_tag_isolates_cache_keys_per_adapter() -> None:
    db = ChunkedTokenDatabase(
        LMCacheEngineConfig.from_legacy(chunk_size=16, backend="cpu"),
        dumb_metadata(),
    )
    tokens = generate_tokens(64, "cpu")

    def keys_for(lora_request: LoRARequest | None) -> list[CacheEngineKey]:
        keys = []
        for _, _, key in db.process_tokens(
            tokens=tokens, request_configs=add_lora_tag(None, lora_request)
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
        return keys

    base_keys = keys_for(None)
    adapter_a_keys = keys_for(ADAPTER_A)
    adapter_b_keys = keys_for(ADAPTER_B)

    assert len(base_keys) == len(adapter_a_keys) == len(adapter_b_keys) == 4
    assert [key.lora_name for key in base_keys] == [""] * 4
    assert [key.lora_name for key in adapter_a_keys] == ["adapter-a"] * 4
    for keys, other_keys in itertools.combinations(
        [base_keys, adapter_a_keys, adapter_b_keys], 2
    ):
        assert set(keys).isdisjoint(other_keys)
        assert {key.to_string() for key in keys}.isdisjoint(
            key.to_string() for key in other_keys
        )


def test_request_tracker_carries_lora_tag_to_store_configs() -> None:
    new_request = SimpleNamespace(
        block_ids=[1],
        req_id="req-1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SimpleNamespace(extra_args=None),
        lora_request=ADAPTER_A,
    )

    tracker = RequestTracker.from_new_request(
        lmcache_config=SimpleNamespace(),
        new_request=new_request,
        num_tokens_to_compute=4,
        lmcache_cached_tokens=0,
        skip_save=False,
    )

    assert tracker.request_configs == ADAPTER_A_CONFIGS


def test_lookup_carries_lora_tag() -> None:
    lookup_client = _FakeLookupClient()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lookup_client=lookup_client
    )
    connector.kv_role = "kv_both"
    connector._requests_priority = {}
    connector.skip_last_n_tokens = 0
    connector.config = SimpleNamespace(min_retrieve_tokens=0)
    connector._max_tokens_per_load = 0
    connector._lmcache_chunk_size = 4
    connector.load_specs = {}
    request = SimpleNamespace(
        request_id="req-1",
        all_token_ids=[1, 2, 3, 4],
        sampling_params=SimpleNamespace(extra_args=None),
        lora_request=ADAPTER_A,
        num_tokens=4,
    )

    assert connector.get_num_new_matched_tokens(request, 0) == 0
    assert lookup_client.request_configs == [ADAPTER_A_CONFIGS]


def test_layerwise_store_carries_lora_tag() -> None:
    connector, engine = _make_connector(
        _make_req(save_spec=SaveSpec(skip_leading_tokens=0, can_save=True))
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)

    assert engine.request_configs == [ADAPTER_A_CONFIGS]


@pytest.mark.parametrize("enable_blending", [False, True])
def test_layerwise_load_carries_lora_tag(enable_blending: bool) -> None:
    connector, engine = _make_connector(
        _make_req(
            load_spec=LoadSpec(
                vllm_cached_tokens=0, lmcache_cached_tokens=16, can_load=True
            )
        )
    )
    connector.enable_blending = enable_blending

    connector.start_load_kv(SimpleNamespace(attn_metadata=object()))

    assert engine.request_configs == [ADAPTER_A_CONFIGS]

# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    ReqMeta,
    RequestTracker,
)


class _FakeKVTransferConfig:
    def get_from_extra_config(self, key: str, default: Any) -> Any:
        return default


class _FakeManager:
    def __init__(self, engine: Any = None) -> None:
        self.lmcache_engine = engine
        self.post_init_calls = 0

    def post_init(self) -> None:
        self.post_init_calls += 1


class _FakeGPUConnector:
    def __init__(self) -> None:
        self.reconfigured_layers: list[int] = []

    def reconfigure_for_layers(self, num_layers: int) -> None:
        self.reconfigured_layers.append(num_layers)


def _make_vllm_config(
    *,
    is_hybrid: bool,
    is_attention_free: bool = False,
    num_layers: int = 4,
) -> SimpleNamespace:
    model_config = SimpleNamespace(
        is_hybrid=is_hybrid,
        is_attention_free=is_attention_free,
        get_num_layers=lambda _parallel_config: num_layers,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(block_size=128),
        parallel_config=SimpleNamespace(),
        kv_transfer_config=_FakeKVTransferConfig(),
    )


def _make_lmcache_config() -> SimpleNamespace:
    return SimpleNamespace(
        enable_async_loading=False,
        use_layerwise=True,
        enable_blending=False,
        save_unfull_chunk=False,
        chunk_size=64,
    )


def test_get_num_new_matched_tokens_returns_zero_for_hybrid_models() -> None:
    """Test that hybrid models never report scheduler-side LMCache hits."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._has_non_attention_layers = True
    request = cast(Any, SimpleNamespace(request_id="req-1"))

    assert connector.get_num_new_matched_tokens(request, num_computed_tokens=0) == 0


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize(
    ("is_hybrid", "is_attention_free"),
    [(True, False), (False, True)],
)
def test_init_state_sets_hybrid_guard_for_scheduler_and_worker(
    role: KVConnectorRole,
    is_hybrid: bool,
    is_attention_free: bool,
) -> None:
    """Test init-time detection for both vLLM processes."""
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._manager = _FakeManager()
    connector._parent = SimpleNamespace()

    connector._init_connector_state(
        role,
        cast(
            Any,
            _make_vllm_config(
                is_hybrid=is_hybrid,
                is_attention_free=is_attention_free,
            ),
        ),
        cast(Any, _make_lmcache_config()),
    )

    request = cast(Any, SimpleNamespace(request_id="req-1"))
    assert connector.get_num_new_matched_tokens(request, num_computed_tokens=0) == 0
    assert connector.force_skip_save is is_attention_free


def test_register_kv_caches_filters_hma_layers_and_reconfigures_connector() -> None:
    """Test worker registration keeps only attention KV cache entries."""
    gpu_connector = _FakeGPUConnector()
    engine = SimpleNamespace(
        num_layers=3,
        metadata=SimpleNamespace(kv_shape=(3, 2, 64, 8, 128)),
        gpu_connector=gpu_connector,
    )
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.kv_caches = {}
    connector.num_layers = 3
    connector._has_non_attention_layers = True
    connector._manager = _FakeManager(engine)

    attn_0 = torch.randn(2, 32, 128, 8, 64, dtype=torch.float16)
    attn_1 = torch.randn(32, 128, 512, dtype=torch.float16)
    recurrent_state = [
        torch.randn(32, 16, 128, dtype=torch.float16),
        torch.randn(32, 4, 64, dtype=torch.float16),
    ]

    connector.register_kv_caches(
        {
            "model.layers.0.self_attn": attn_0,
            "model.layers.1.mamba": recurrent_state,
            "model.layers.2.self_attn": attn_1,
        }
    )

    assert connector.num_layers == 2
    assert engine.num_layers == 2
    assert engine.metadata.kv_shape == (2, 2, 64, 8, 128)
    assert gpu_connector.reconfigured_layers == [2]
    assert connector._manager.post_init_calls == 1
    assert list(connector.kv_caches) == [
        "model.layers.0.self_attn",
        "model.layers.2.self_attn",
    ]
    assert connector.kv_caches["model.layers.0.self_attn"] is attn_0
    assert connector.kv_caches["model.layers.2.self_attn"] is attn_1


def test_register_kv_caches_disables_save_when_no_attention_layers() -> None:
    """Test attention-free registration fails closed for save."""
    gpu_connector = _FakeGPUConnector()
    engine = SimpleNamespace(
        num_layers=2,
        metadata=SimpleNamespace(kv_shape=(2, 2, 64, 8, 128)),
        gpu_connector=gpu_connector,
    )
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.kv_caches = {}
    connector.num_layers = 2
    connector.force_skip_save = False
    connector._has_non_attention_layers = True
    connector._manager = _FakeManager(engine)

    connector.register_kv_caches(
        {
            "model.layers.0.mamba": [
                torch.randn(32, 16, 128, dtype=torch.float16),
                torch.randn(32, 4, 64, dtype=torch.float16),
            ],
            "model.layers.1.mamba": [
                torch.randn(32, 16, 128, dtype=torch.float16),
                torch.randn(32, 4, 64, dtype=torch.float16),
            ],
        }
    )

    assert connector.num_layers == 0
    assert connector.kv_caches == {}
    assert connector.force_skip_save is True
    assert engine.metadata.kv_shape == (0, 2, 64, 8, 128)
    assert gpu_connector.reconfigured_layers == [0]


def test_request_tracker_uses_attention_block_group_for_hma_request() -> None:
    """Test new HMA requests track the attention KV cache group."""
    new_request = SimpleNamespace(
        req_id="req-1",
        block_ids=([99], [10, 11, 12, 13]),
        prompt_token_ids=list(range(512)),
        sampling_params=SimpleNamespace(extra_args=None),
    )

    tracker = RequestTracker.from_new_request(
        cast(Any, None),
        cast(Any, new_request),
        num_tokens_to_compute=512,
        lmcache_cached_tokens=0,
        skip_save=False,
    )

    assert tracker.allocated_block_ids == [10, 11, 12, 13]


def test_request_tracker_tied_hma_block_groups_disable_save() -> None:
    """Test ambiguous HMA block groups fail closed for save."""
    new_request = SimpleNamespace(
        req_id="req-1",
        block_ids=([99], [10]),
        prompt_token_ids=list(range(128)),
        sampling_params=SimpleNamespace(extra_args=None),
    )

    tracker = RequestTracker.from_new_request(
        cast(Any, None),
        cast(Any, new_request),
        num_tokens_to_compute=128,
        lmcache_cached_tokens=0,
        skip_save=False,
    )

    assert tracker.skip_save is True
    assert ReqMeta.from_request_tracker(tracker, block_size=128) is None


def test_request_tracker_update_uses_attention_new_block_group() -> None:
    """Test HMA follow-up scheduling extends with attention blocks."""
    tracker = RequestTracker(
        req_id="req-1",
        prompt_len=512,
        token_ids=list(range(256)),
        allocated_block_ids=[10, 11],
    )

    tracker.update(
        new_token_ids=list(range(256, 512)),
        new_block_ids=([], [12, 13]),
    )

    assert tracker.allocated_block_ids == [10, 11, 12, 13]
    assert tracker.token_ids == list(range(512))


def test_request_tracker_tied_hma_update_disables_save() -> None:
    """Test ambiguous follow-up HMA block groups fail closed for save."""
    tracker = RequestTracker(
        req_id="req-1",
        prompt_len=256,
        token_ids=list(range(128)),
        allocated_block_ids=[10],
    )

    tracker.update(
        new_token_ids=list(range(128, 256)),
        new_block_ids=([99], [11]),
    )

    assert tracker.skip_save is True
    assert ReqMeta.from_request_tracker(tracker, block_size=128) is None


def test_req_meta_caps_save_to_allocated_attention_blocks() -> None:
    """Test HMA save accounting reflects the attention blocks actually stored."""
    tracker = RequestTracker(
        req_id="req-1",
        prompt_len=512,
        token_ids=list(range(512)),
        allocated_block_ids=[10],
    )

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=128,
        lmcache_chunk_size=64,
    )

    assert req_meta is not None
    assert len(req_meta.token_ids) == 128
    assert tracker.num_saved_tokens == 128
    assert req_meta.slot_mapping.tolist() == list(range(1280, 1408))


def test_req_meta_skips_save_when_attention_capacity_is_below_chunk() -> None:
    """Test HMA does not emit metadata with too few attention slots."""
    tracker = RequestTracker(
        req_id="req-1",
        prompt_len=512,
        token_ids=list(range(512)),
        allocated_block_ids=[10],
    )

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=128,
        lmcache_chunk_size=256,
    )

    assert req_meta is None
    assert tracker.num_saved_tokens == 0

# SPDX-License-Identifier: Apache-2.0
# Standard
from types import ModuleType, SimpleNamespace
import importlib
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import KVCacheRegistration
from lmcache.v1.multiprocess.gpu_context import unwrap_kv_cache_tensors
from lmcache.v1.multiprocess.server import get_sglang_chunk_tensor


def _install_fake_sglang(monkeypatch: pytest.MonkeyPatch):
    class FakeModelConfig:
        def __init__(self, model_path: str = "fake-model"):
            self.model_path = model_path

    sglang_pkg = ModuleType("sglang")
    sglang_pkg.__path__ = []
    srt_pkg = ModuleType("sglang.srt")
    srt_pkg.__path__ = []
    configs_pkg = ModuleType("sglang.srt.configs")
    configs_pkg.__path__ = []
    model_config_mod = ModuleType("sglang.srt.configs.model_config")
    model_config_mod.ModelConfig = FakeModelConfig  # type: ignore[attr-defined]

    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.configs.model_config",
        "lmcache.integration.sglang.sglang_adapter",
        "lmcache.integration.sglang.multi_process_adapter",
    ]:
        sys.modules.pop(name, None)

    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt.configs", configs_pkg)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs.model_config",
        model_config_mod,
    )
    return FakeModelConfig


def _import_sglang_modules(monkeypatch: pytest.MonkeyPatch):
    model_config_cls = _install_fake_sglang(monkeypatch)
    sglang_adapter = importlib.import_module(
        "lmcache.integration.sglang.sglang_adapter"
    )
    mp_adapter = importlib.import_module(
        "lmcache.integration.sglang.multi_process_adapter"
    )
    return model_config_cls, sglang_adapter, mp_adapter


def test_resolve_sglang_kv_pools_from_kvcache(monkeypatch: pytest.MonkeyPatch):
    _, sglang_adapter, _ = _import_sglang_modules(monkeypatch)

    k_pool = [torch.zeros(2, 3), torch.ones(2, 3)]
    v_pool = [torch.full((2, 3), 2.0), torch.full((2, 3), 3.0)]
    resolved_k_pool, resolved_v_pool = sglang_adapter.resolve_sglang_kv_pools(
        kvcache=SimpleNamespace(k_buffer=k_pool, v_buffer=v_pool)
    )

    assert resolved_k_pool == k_pool
    assert resolved_v_pool == v_pool


def test_resolve_sglang_kv_pools_rejects_non_tensor_buffers(
    monkeypatch: pytest.MonkeyPatch,
):
    _, sglang_adapter, _ = _import_sglang_modules(monkeypatch)

    with pytest.raises(TypeError, match="sequence of torch.Tensor"):
        sglang_adapter.resolve_sglang_kv_pools(
            kvcache=SimpleNamespace(k_buffer=["bad"], v_buffer=["bad"])
        )


def test_mp_connector_requires_tp_group(monkeypatch: pytest.MonkeyPatch):
    model_config_cls, _, mp_adapter = _import_sglang_modules(monkeypatch)

    fake_tensor = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    with pytest.raises(ValueError, match="tp_group is required"):
        mp_adapter.LMCacheMPLayerwiseConnector(
            sgl_config=model_config_cls("fake-model"),
            tp_size=2,
            rank=0,
            page_size=16,
            host="127.0.0.1",
            port=5555,
            k_pool=[fake_tensor],
            v_pool=[fake_tensor],
            tp_group=None,
        )


@pytest.mark.parametrize(
    ("future", "message"),
    [
        (None, "missing a future"),
        (SimpleNamespace(result=lambda timeout: False), "retrieve failed"),
    ],
)
def test_load_kv_layerwise_cleans_failed_state_before_raise(
    monkeypatch: pytest.MonkeyPatch,
    future,
    message: str,
):
    _, _, mp_adapter = _import_sglang_modules(monkeypatch)

    connector = mp_adapter.LMCacheMPLayerwiseConnector.__new__(
        mp_adapter.LMCacheMPLayerwiseConnector
    )
    connector._mq_timeout = 1.0
    connector.num_layers = 1
    cleanup_calls: list[str] = []
    end_calls: list[str] = []
    submit_calls: list[int] = []

    monkeypatch.setattr(
        connector,
        "cleanup_retrieve_state",
        lambda state: cleanup_calls.append(state.request_id),
    )
    monkeypatch.setattr(
        connector,
        "end_session",
        lambda request_id: end_calls.append(request_id),
    )
    monkeypatch.setattr(
        connector,
        "submit_retrieve",
        lambda state, layer_id: submit_calls.append(layer_id),
    )

    connector._active_retrieves = [
        mp_adapter._ActiveRetrieveState(
            request_id="req-1",
            token_ids=[1, 2, 3],
            offset=0,
            matched_end=256,
            block_ids=[0],
            in_flight_layer=0,
            future=future,
        )
    ]

    with pytest.raises(RuntimeError, match=message):
        connector.load_kv_layerwise(0)

    assert cleanup_calls == ["req-1"]
    assert end_calls == []
    assert submit_calls == []
    assert connector._active_retrieves == []


def test_kv_cache_registration_rejects_empty_payload():
    with pytest.raises(ValueError, match="must contain at least one KV cache tensor"):
        KVCacheRegistration(
            instance_id=1,
            model_name="model",
            world_size=1,
            engine_type="sglang",
            block_size=16,
            kv_caches=[[], []],
        )


def test_unwrap_kv_cache_tensors_rejects_invalid_leaf():
    with pytest.raises(TypeError, match="torch.Tensor or CUDA IPC wrapper"):
        unwrap_kv_cache_tensors(object())


def test_get_sglang_chunk_tensor_validates_shape_and_ranges():
    chunk_tensor = torch.zeros(2, 4, 256, 128)
    memory_obj = SimpleNamespace(tensor=chunk_tensor)

    returned_tensor = get_sglang_chunk_tensor(
        memory_obj,
        layer_begin=1,
        layer_end=3,
        token_begin=8,
        token_end=128,
    )
    assert returned_tensor is chunk_tensor

    with pytest.raises(ValueError, match="shape"):
        get_sglang_chunk_tensor(
            SimpleNamespace(tensor=torch.zeros(2, 4, 256)),
            layer_begin=0,
            layer_end=1,
            token_begin=0,
            token_end=16,
        )

    with pytest.raises(ValueError, match="Token window"):
        get_sglang_chunk_tensor(
            memory_obj,
            layer_begin=0,
            layer_end=2,
            token_begin=0,
            token_end=300,
        )

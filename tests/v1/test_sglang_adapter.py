# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.sglang import sglang_adapter as adapter_mod
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata


@pytest.mark.parametrize(
    ("attention_arch", "kv_head_dim", "expected_shape", "expected_use_mla"),
    [
        ("MHA", None, (2, 2, 16, 4, 128), False),
        ("MLA", 576, (2, 1, 16, 1, 576), True),
    ],
)
def test_init_lmcache_engine_propagates_sglang_attention_arch(
    monkeypatch: pytest.MonkeyPatch,
    attention_arch: str,
    kv_head_dim: int | None,
    expected_shape: tuple[int, int, int, int, int],
    expected_use_mla: bool,
) -> None:
    """Adapter metadata selects the connector layout SGLang actually uses."""
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    monkeypatch.setattr(adapter_mod, "lmcache_get_config", lambda _path: config)
    monkeypatch.setattr(adapter_mod.LMCacheEngineBuilder, "get", lambda _name: None)

    captured: dict[str, object] = {}

    def create_connector(
        _connector_config: LMCacheEngineConfig,
        metadata: object,
        engine_type: EngineType,
    ) -> object:
        captured["metadata"] = metadata
        captured["engine_type"] = engine_type
        return object()

    sentinel_engine = MagicMock()
    monkeypatch.setattr(adapter_mod, "CreateGPUConnector", create_connector)
    monkeypatch.setattr(
        adapter_mod.LMCacheEngineBuilder,
        "get_or_create",
        lambda *args, **kwargs: sentinel_engine,
    )
    model_config = SimpleNamespace(
        attention_arch=SimpleNamespace(name=attention_arch),
        num_hidden_layers=2,
        model_path="test-model",
        head_dim=128,
        get_num_kv_heads=lambda _tp_size: 4,
    )

    engine = adapter_mod.init_lmcache_engine(
        model_config,
        tp_size=1,
        local_rank=0,
        global_rank=0,
        kv_dtype=torch.float16,
        config_file="",
        kv_head_dim=kv_head_dim,
    )

    metadata = cast(LMCacheMetadata, captured["metadata"])
    assert engine is sentinel_engine
    assert metadata.kv_shape == expected_shape
    assert metadata.use_mla is expected_use_mla
    assert captured["engine_type"] is EngineType.SGLANG


def test_init_lmcache_engine_rejects_mla_without_pool_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLA metadata must not guess latent width from the attention head."""
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    monkeypatch.setattr(adapter_mod, "lmcache_get_config", lambda _path: config)
    monkeypatch.setattr(adapter_mod.LMCacheEngineBuilder, "get", lambda _name: None)
    model_config = SimpleNamespace(
        attention_arch=SimpleNamespace(name="MLA"),
        num_hidden_layers=2,
        model_path="test-model",
    )

    with pytest.raises(ValueError, match="positive KV-cache row width"):
        adapter_mod.init_lmcache_engine(
            model_config,
            tp_size=1,
            local_rank=0,
            global_rank=0,
            kv_dtype=torch.float16,
            config_file="",
        )


@pytest.mark.parametrize(
    ("attention_arch", "v_pool_size", "expected_cache_count", "expected_width"),
    [
        ("MHA", 2, 4, None),
        ("MLA", 0, 2, 576),
    ],
)
def test_connector_registers_only_the_model_kv_layout(
    monkeypatch: pytest.MonkeyPatch,
    attention_arch: str,
    v_pool_size: int,
    expected_cache_count: int,
    expected_width: int | None,
) -> None:
    """MLA uses one latent pool per layer; MHA keeps split K/V pools."""
    engine = MagicMock()
    captured: dict[str, object] = {}

    def init_engine(*args: object, **kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return engine

    monkeypatch.setattr(adapter_mod, "init_lmcache_engine", init_engine)
    width = 576 if attention_arch == "MLA" else 128
    k_pool = [torch.empty(8, 1, width) for _ in range(2)]
    v_pool = [torch.empty(8, 1, width) for _ in range(v_pool_size)]
    model_config = SimpleNamespace(
        attention_arch=SimpleNamespace(name=attention_arch),
        num_hidden_layers=2,
    )

    connector = adapter_mod.LMCacheConnector(
        model_config,
        tp_size=1,
        rank=0,
        k_pool=k_pool,
        v_pool=v_pool,
        config_file="",
    )

    assert len(connector.kvcaches) == expected_cache_count
    init_kwargs = cast(dict[str, object], captured["kwargs"])
    assert init_kwargs["kv_head_dim"] == expected_width
    engine.post_init.assert_called_once_with(kvcaches=connector.kvcaches)

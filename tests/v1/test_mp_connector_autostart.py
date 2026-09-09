# SPDX-License-Identifier: Apache-2.0
"""Connector initialization rejects unsupported auto-start deployments early."""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

# Third Party
import pytest


@pytest.fixture
def connector_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Isolate engine geometry and network adapters, not connector URL parsing."""
    pytest.importorskip("vllm")
    # First Party
    from lmcache.integration.vllm import lmcache_mp_connector as module

    monkeypatch.setattr(
        module.KVConnectorBase_V1, "__init__", MagicMock(return_value=None)
    )
    for name in (
        "validate_mamba_step_alignment",
        "validate_kv_cache_groups",
        "validate_dcp_support",
    ):
        monkeypatch.setattr(module, name, MagicMock())
    monkeypatch.setattr(module, "get_group_tokens_per_block", lambda *args: [16])
    monkeypatch.setattr(module, "get_vllm_scheduler_block_size", lambda *args: 16)
    monkeypatch.setattr(
        module, "get_dcp_decorated_model_name", lambda *args: "test-model"
    )
    monkeypatch.setattr(
        module,
        "build_parallel_strategy_from_vllm_config",
        lambda *args: SimpleNamespace(dcp_size=1, vllm_world_size=2, vllm_worker_id=0),
    )
    monkeypatch.setattr(
        module,
        "LMCacheMPSchedulerAdapter",
        MagicMock(return_value=SimpleNamespace(lmcache_tokens_per_chunk=256)),
    )
    monkeypatch.setattr(module, "LMCacheMPWorkerAdapter", MagicMock())
    monkeypatch.setattr(
        module.LMCacheMPConnector,
        "transfer_intermediate_tensors",
        PropertyMock(return_value=False),
    )
    return module


@pytest.mark.parametrize("role_name", ["SCHEDULER", "WORKER"])
@pytest.mark.parametrize("enabled", [True, "true", False, "false"])
@pytest.mark.parametrize(
    "urls", [["localhost:5555", "localhost:5556"], "localhost:5555, localhost:5556"]
)
def test_connector_multiserver_autostart_fails_before_network(
    connector_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    role_name: str,
    enabled: bool | str,
    urls: list[str] | str,
) -> None:
    """Both roles reject auto-start, but retain connect-only multi-server mode."""
    module = connector_module
    role = getattr(module.KVConnectorRole, role_name)
    monkeypatch.setattr(
        module.LMCacheMPConnector, "role", PropertyMock(return_value=role)
    )
    extra = {"lmcache.mp.server_urls": urls, "lmcache.mp.autostart": enabled}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra, get_from_extra_config=extra.get
        ),
        parallel_config=SimpleNamespace(
            world_size=2, data_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    if enabled in (True, "true"):
        with pytest.raises(ValueError, match="only supports a single server"):
            module.LMCacheMPConnector(config, role)
        module.LMCacheMPSchedulerAdapter.assert_not_called()
        module.LMCacheMPWorkerAdapter.assert_not_called()
        module.validate_dcp_support.assert_not_called()
    else:
        module.LMCacheMPConnector(config, role)
        adapter = (
            module.LMCacheMPSchedulerAdapter
            if role_name == "SCHEDULER"
            else module.LMCacheMPWorkerAdapter
        )
        adapter.assert_called_once()


@pytest.mark.parametrize("role_name", ["SCHEDULER", "WORKER"])
@pytest.mark.parametrize("explicit_url", [True, False])
def test_connector_single_server_autostart_keeps_resolved_endpoint(
    connector_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    role_name: str,
    explicit_url: bool,
) -> None:
    """Single-server auto-start reaches the adapter using connector URL precedence."""
    module = connector_module
    role = getattr(module.KVConnectorRole, role_name)
    monkeypatch.setattr(
        module.LMCacheMPConnector, "role", PropertyMock(return_value=role)
    )
    extra: dict[str, object] = {
        "lmcache.mp.autostart": True,
        "lmcache.mp.host": "127.0.0.1",
        "lmcache.mp.port": 5555,
    }
    if explicit_url:
        extra["lmcache.mp.server_urls"] = ["localhost:6000"]
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra, get_from_extra_config=extra.get
        ),
        parallel_config=SimpleNamespace(
            world_size=2, data_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    module.LMCacheMPConnector(config, role)
    expected = "tcp://localhost:6000" if explicit_url else "tcp://127.0.0.1:5555"
    if role_name == "SCHEDULER":
        assert module.LMCacheMPSchedulerAdapter.call_args.kwargs["server_urls"] == [
            expected
        ]
    else:
        assert module.LMCacheMPWorkerAdapter.call_args.kwargs["server_url"] == expected

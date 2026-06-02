# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vLLM MP connector auto-start integration."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="vLLM is required for connector glue tests")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402


class FakeKVTransferConfig:
    """Minimal vLLM KV transfer config for connector initialization."""

    def __init__(self, extra_config: dict[str, object] | None = None) -> None:
        self.kv_connector_extra_config = extra_config or {}

    def get_from_extra_config(self, key: str, default: object) -> object:
        """Return extra config values with the vLLM helper shape."""
        return self.kv_connector_extra_config.get(key, default)


class FakeZMQContext:
    """Minimal ZMQ context provider for connector initialization."""

    context = MagicMock(name="zmq_context")

    @classmethod
    def instance(cls) -> MagicMock:
        """Return the shared fake context."""
        return cls.context


@pytest.fixture(autouse=True)
def fake_base_connector_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch vLLM base connector initialization to isolate LMCache glue."""

    def fake_init(
        self,
        vllm_config: object,
        role: KVConnectorRole,
        kv_cache_config: object | None = None,
    ) -> None:
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config

    monkeypatch.setattr(connector_mod.KVConnectorBase_V1, "__init__", fake_init)


def _fake_vllm_config(extra_config: dict[str, object] | None = None) -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config = FakeKVTransferConfig(extra_config)
    config.cache_config.block_size = 16
    return config


def test_scheduler_autostarts_before_creating_scheduler_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    launcher = MagicMock(name="launcher")
    scheduler_adapter = MagicMock(name="scheduler_adapter")

    def fake_autostart(**kwargs: object) -> MagicMock:
        events.append("autostart")
        assert kwargs["server_url"] == "tcp://localhost:5555"
        assert kwargs["zmq_context"] is FakeZMQContext.context
        return launcher

    def fake_create_scheduler_adapter(*_args: object) -> MagicMock:
        events.append("create_scheduler_adapter")
        return scheduler_adapter

    monkeypatch.setattr(connector_mod.zmq, "Context", FakeZMQContext)
    monkeypatch.setattr(connector_mod, "maybe_autostart_mp_server", fake_autostart)
    monkeypatch.setattr(
        connector_mod,
        "create_scheduler_adapter",
        fake_create_scheduler_adapter,
    )

    connector = connector_mod.LMCacheMPConnector(
        _fake_vllm_config({"lmcache.mp.autostart": True}),
        KVConnectorRole.SCHEDULER,
    )

    assert events == ["autostart", "create_scheduler_adapter"]
    assert connector.scheduler_adapter is scheduler_adapter
    launcher.shutdown.assert_not_called()


def test_scheduler_init_failure_shuts_down_owned_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = MagicMock(name="launcher")

    monkeypatch.setattr(connector_mod.zmq, "Context", FakeZMQContext)
    monkeypatch.setattr(
        connector_mod,
        "maybe_autostart_mp_server",
        lambda **_kwargs: launcher,
    )
    monkeypatch.setattr(
        connector_mod,
        "create_scheduler_adapter",
        MagicMock(side_effect=RuntimeError("adapter failed")),
    )

    with pytest.raises(RuntimeError, match="adapter failed"):
        connector_mod.LMCacheMPConnector(
            _fake_vllm_config({"lmcache.mp.autostart": True}),
            KVConnectorRole.SCHEDULER,
        )

    launcher.shutdown.assert_called_once_with()


def test_worker_does_not_autostart_mp_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_adapter = MagicMock(name="worker_adapter")

    monkeypatch.setattr(connector_mod.zmq, "Context", FakeZMQContext)
    monkeypatch.setattr(
        connector_mod,
        "maybe_autostart_mp_server",
        MagicMock(side_effect=AssertionError("worker should not autostart")),
    )
    monkeypatch.setattr(
        connector_mod,
        "create_worker_adapter",
        MagicMock(return_value=worker_adapter),
    )

    connector = connector_mod.LMCacheMPConnector(
        _fake_vllm_config({"lmcache.mp.autostart": True}),
        KVConnectorRole.WORKER,
    )

    assert connector.worker_adapter is worker_adapter
    connector.shutdown()
    worker_adapter.shutdown.assert_called_once_with()

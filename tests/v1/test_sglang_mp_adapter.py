# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import os
import sys

# Mock sglang before importing the adapter
mock_sglang = MagicMock()
sys.modules["sglang"] = mock_sglang
sys.modules["sglang.srt"] = mock_sglang
sys.modules["sglang.srt.configs"] = mock_sglang
sys.modules["sglang.srt.configs.model_config"] = mock_sglang

# Third Party
import pytest  # noqa: E402
import torch  # noqa: E402

# First Party
from lmcache.integration.sglang import (  # noqa: E402
    multi_process_adapter as adapter_mod,
)
from lmcache.integration.sglang.multi_process_adapter import (  # noqa: E402
    LMCacheMPConnector,
)


def _make_sglang_connector() -> LMCacheMPConnector:
    sgl_config = MagicMock()
    sgl_config.model_path = "test-model"
    k_pool = [torch.zeros(1, device="cpu")]
    v_pool = [torch.zeros(1, device="cpu")]
    return LMCacheMPConnector(
        sgl_config=sgl_config,
        tp_size=1,
        rank=0,
        page_size=16,
        host="localhost",
        port=5555,
        k_pool=k_pool,
        v_pool=v_pool,
    )


@pytest.fixture
def mock_sglang_adapter(monkeypatch):
    # Stub the MQ boundary and chunk size
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)

    future = MagicMock(name="future")
    future.result.return_value = None
    send_mock = MagicMock(name="send_lmcache_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    # Mock HeartbeatThread
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", MagicMock())

    # Bypass CUDA IPC wrappers
    monkeypatch.setattr(adapter_mod, "_wrap_sglang_kv_caches", lambda k, v: [])


def test_sglang_instance_id_is_uuid_derived_63_bit_int(mock_sglang_adapter) -> None:
    """instance_id is a 63-bit int, not the PID, and unique per connector."""
    connector = _make_sglang_connector()

    assert isinstance(connector.instance_id, int)
    assert not isinstance(connector.instance_id, bool)
    assert 0 <= connector.instance_id < (1 << 63)
    assert connector.instance_id != os.getpid()

    connector2 = _make_sglang_connector()
    assert connector.instance_id != connector2.instance_id


def test_sglang_instance_id_logged_at_info_on_construction(
    mock_sglang_adapter, monkeypatch
) -> None:
    """The constructor logs instance_id at INFO for correlating server-side
    reap warnings. The module logger does not propagate (``propagate=False``),
    so the test spies on it directly instead of using ``caplog``."""
    messages: list[str] = []

    def spy_info(msg: object, *args: object, **kwargs: object) -> None:
        messages.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(adapter_mod.logger, "info", spy_info)

    connector = _make_sglang_connector()

    assert any(str(connector.instance_id) in msg for msg in messages)

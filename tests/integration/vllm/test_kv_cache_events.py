# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

vllm = pytest.importorskip("vllm", reason="vLLM required for KV cache events test")

# Third Party
from vllm.distributed.kv_events import BlockStored  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_connector_v1 import (  # noqa: E402
    KVConnectorRole,
    LMCacheConnectorV1Dynamic,
    LMCacheKVEvents,
)
from lmcache.utils import CacheStoreEvent  # noqa: E402


@pytest.fixture
def mock_engine():
    """Lightweight mock of LMCacheConnectorV1Impl."""
    engine = MagicMock()
    engine.get_kv_events.return_value = []
    return engine


@pytest.fixture
def mock_vllm_config():
    return MagicMock()


@pytest.fixture
def mock_kv_cache_config():
    return MagicMock()


def _make_connector(mock_engine, vllm_config, kv_cache_config):
    """Create a connector with the LMCacheConnectorV1Impl patch applied."""
    with patch(
        "lmcache.integration.vllm.lmcache_connector_v1.LMCacheConnectorV1Impl",
        return_value=mock_engine,
    ):
        return LMCacheConnectorV1Dynamic(
            vllm_config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
        )


class TestGetKvConnectorKvCacheEvents:
    def test_returns_none_when_no_events(
        self, mock_engine, mock_vllm_config, mock_kv_cache_config
    ):
        mock_engine.get_kv_events.return_value = []
        connector = _make_connector(mock_engine, mock_vllm_config, mock_kv_cache_config)
        assert connector.get_kv_connector_kv_cache_events() is None

    def test_returns_kvcache_events(
        self, mock_engine, mock_vllm_config, mock_kv_cache_config
    ):
        test_events = [
            CacheStoreEvent(
                block_hashes=[1],
                parent_block_hash=None,
                token_ids=[1],
                block_size=8,
                lora_id=1,
                medium="gpu",
                lora_name="lora_adapter_1",
            ),
            CacheStoreEvent(
                block_hashes=[2, 3],
                parent_block_hash=1,
                token_ids=[2, 3],
                block_size=16,
                lora_id=None,
                medium="cpu",
                lora_name=None,
            ),
        ]
        mock_engine.get_kv_events.return_value = test_events
        connector = _make_connector(mock_engine, mock_vllm_config, mock_kv_cache_config)
        result = connector.get_kv_connector_kv_cache_events()
        events = result.get_all_events()
        assert len(events) == 2
        assert isinstance(events[0], BlockStored)
        assert events[0].block_hashes == [1]
        assert isinstance(events[1], BlockStored)
        assert events[1].block_hashes == [2, 3]


class TestTakeEvents:
    def test_returns_empty_when_no_events(
        self, mock_engine, mock_vllm_config, mock_kv_cache_config
    ):
        connector = _make_connector(mock_engine, mock_vllm_config, mock_kv_cache_config)
        assert list(connector.take_events()) == []

    def test_ignores_none_events(
        self, mock_engine, mock_vllm_config, mock_kv_cache_config
    ):
        connector = _make_connector(mock_engine, mock_vllm_config, mock_kv_cache_config)
        connector.update_connector_output(KVConnectorOutput(kv_cache_events=None))
        assert list(connector.take_events()) == []

    def test_ignores_unknown_event_type(
        self, mock_engine, mock_vllm_config, mock_kv_cache_config
    ):
        connector = _make_connector(mock_engine, mock_vllm_config, mock_kv_cache_config)
        fake_events = MagicMock()
        connector.update_connector_output(
            KVConnectorOutput(kv_cache_events=fake_events)
        )
        assert list(connector.take_events()) == []

    def test_aggregates_before_returning(
        self, mock_engine, mock_vllm_config, mock_kv_cache_config
    ):
        event1 = BlockStored(
            block_hashes=[1],
            parent_block_hash=0,
            token_ids=[10],
            lora_id=0,
            block_size=16,
            medium="cpu",
            lora_name=None,
        )
        event2 = BlockStored(
            block_hashes=[2],
            parent_block_hash=1,
            token_ids=[20],
            lora_id=0,
            block_size=16,
            medium="gpu",
            lora_name="lora_1",
        )

        events = LMCacheKVEvents(num_workers=1)
        events.add_events([event1])
        events.add_events([event2])

        connector = _make_connector(mock_engine, mock_vllm_config, mock_kv_cache_config)

        connector.update_connector_output(KVConnectorOutput(kv_cache_events=events))

        result = list(connector.take_events())
        assert len(result) == 2
        assert result[0] == event1
        assert result[1] == event2

        result = list(connector.take_events())
        assert len(result) == 0

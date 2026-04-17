# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the remote_ttl feature on Redis/RESP connectors.

These tests verify that:
- remote_ttl=None (default) does NOT pass `ex` to SET commands
- remote_ttl=<int> passes `ex=<int>` to SET commands
- The TTL is threaded correctly from config -> adapter -> connector
- RESPConnector logs a warning when remote_ttl is set (native client limitation)

All tests use mocked Redis connections — no real Redis/Dragonfly or GPU needed.

Run: PYTHONPATH=. python -m pytest tests/v1/storage_backend/test_redis_ttl.py -v --tb=short
"""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import os
import sys

# Prevent lmcache __init__.py from trying to import CUDA backends
# This must happen before any lmcache import
os.environ.setdefault("LMCACHE_SKIP_CUDA_CHECK", "1")

# Third Party
import pytest
import torch

# Import only the specific modules we need, avoiding the full package __init__
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.connector.redis_connector import (
    RedisConnector,
    RedisClusterConnector,
    RedisSentinelConnector,
    RESPConnector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(use_mla: bool = False) -> LMCacheMetadata:
    """Create minimal LMCacheMetadata for testing."""
    kv_shape = (32, 1 if use_mla else 2, 256, 8, 128)
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        use_mla=use_mla,
    )


def _make_memory_obj() -> TensorMemoryObj:
    """Create a minimal TensorMemoryObj for put operations.

    Sets both shape/dtype (single-group) and shapes/dtypes (multi-group) metadata
    so that get_shapes()/get_dtypes() works correctly.
    """
    raw_data = torch.zeros(1024, dtype=torch.uint8)
    shape = torch.Size([2, 32, 256, 1024])
    dtype = torch.bfloat16
    meta = MemoryObjMetadata(
        shape=shape,
        dtype=dtype,
        address=0,
        phy_size=1024,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[shape],
        dtypes=[dtype],
    )
    return TensorMemoryObj(raw_data=raw_data, metadata=meta, parent_allocator=None)


def _make_key() -> CacheEngineKey:
    return CacheEngineKey("test-model", 0, 0, 12345, torch.bfloat16)


def _mock_local_cpu_backend():
    """Minimal mock of LocalCPUBackend."""
    backend = MagicMock()
    backend.config = MagicMock()
    backend.config.extra_config = {}
    backend.config.remote_ttl = None
    backend.metadata = _make_metadata()
    return backend


# ---------------------------------------------------------------------------
# RedisConnector TTL tests
# ---------------------------------------------------------------------------


class TestRedisConnectorTTL:
    """Test remote_ttl on RedisConnector (redis://)."""

    @pytest.fixture
    def mock_redis_conn(self):
        conn = AsyncMock()
        conn.set = AsyncMock(return_value=True)
        conn.exists = AsyncMock(return_value=True)
        conn.get = AsyncMock(return_value=None)
        conn.close = AsyncMock()
        return conn

    @pytest.fixture
    def loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    def lcb(self):
        return _mock_local_cpu_backend()

    def _make_connector(self, mock_redis_conn, loop, lcb, remote_ttl):
        mock_pool = MagicMock()
        with patch(
            "lmcache.v1.storage_backend.connector.redis_connector.redis",
            MagicMock(
                ConnectionPool=MagicMock(return_value=mock_pool),
                Redis=MagicMock(from_pool=MagicMock(return_value=mock_redis_conn)),
            ),
        ):
            return RedisConnector(
                url="redis://localhost:6379",
                loop=loop,
                local_cpu_backend=lcb,
                remote_ttl=remote_ttl,
            )

    @pytest.mark.asyncio
    async def test_put_without_ttl(self, mock_redis_conn, loop, lcb):
        """When remote_ttl is None, set() should NOT pass ex= to Redis."""
        connector = self._make_connector(mock_redis_conn, loop, lcb, None)
        await connector._put(_make_key(), _make_memory_obj())

        assert mock_redis_conn.set.call_count == 2
        for c in mock_redis_conn.set.call_args_list:
            ex_val = c.kwargs.get("ex")
            assert ex_val is None, f"Expected ex=None, got ex={ex_val}"

    @pytest.mark.asyncio
    async def test_put_with_ttl_3600(self, mock_redis_conn, loop, lcb):
        """When remote_ttl=3600, both SET calls should receive ex=3600."""
        connector = self._make_connector(mock_redis_conn, loop, lcb, 3600)
        await connector._put(_make_key(), _make_memory_obj())

        assert mock_redis_conn.set.call_count == 2
        for c in mock_redis_conn.set.call_args_list:
            assert c.kwargs.get("ex") == 3600, f"Expected ex=3600, got {c.kwargs}"

    @pytest.mark.asyncio
    async def test_put_with_ttl_1800(self, mock_redis_conn, loop, lcb):
        """When remote_ttl=1800, both SET calls should receive ex=1800."""
        connector = self._make_connector(mock_redis_conn, loop, lcb, 1800)
        await connector._put(_make_key(), _make_memory_obj())

        for c in mock_redis_conn.set.call_args_list:
            assert c.kwargs.get("ex") == 1800

    @pytest.mark.asyncio
    async def test_ttl_applies_to_both_kv_and_metadata_keys(self, mock_redis_conn, loop, lcb):
        """Both kv_bytes and metadata keys must receive the same TTL."""
        connector = self._make_connector(mock_redis_conn, loop, lcb, 600)
        await connector._put(_make_key(), _make_memory_obj())

        calls = mock_redis_conn.set.call_args_list
        assert len(calls) == 2
        kv_bytes_key = calls[0].args[0]
        metadata_key = calls[1].args[0]
        assert kv_bytes_key.endswith("kv_bytes"), f"Expected kv_bytes key, got {kv_bytes_key}"
        assert metadata_key.endswith("metadata"), f"Expected metadata key, got {metadata_key}"
        assert calls[0].kwargs["ex"] == 600
        assert calls[1].kwargs["ex"] == 600


# ---------------------------------------------------------------------------
# RedisClusterConnector TTL tests
# ---------------------------------------------------------------------------


class TestRedisClusterConnectorTTL:
    """Test remote_ttl on RedisClusterConnector (redis-cluster://)."""

    @pytest.fixture
    def mock_cluster(self):
        cluster = AsyncMock()
        cluster.set = AsyncMock(return_value=True)
        cluster.exists = AsyncMock(return_value=True)
        cluster.get = AsyncMock(return_value=None)
        cluster.close = AsyncMock()
        return cluster

    @pytest.fixture
    def loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    def lcb(self):
        return _mock_local_cpu_backend()

    @pytest.mark.asyncio
    async def test_cluster_put_with_ttl(self, mock_cluster, loop, lcb):
        """RedisClusterConnector should pass ex= to cluster.set()."""
        with patch(
            "lmcache.v1.storage_backend.connector.redis_connector.RedisCluster",
            return_value=mock_cluster,
        ):
            connector = RedisClusterConnector(
                hosts_and_ports=[("localhost", 7000)],
                username="",
                password="",
                loop=loop,
                local_cpu_backend=lcb,
                remote_ttl=7200,
            )

        await connector._put(_make_key(), _make_memory_obj())

        assert mock_cluster.set.call_count == 2
        for c in mock_cluster.set.call_args_list:
            assert c.kwargs.get("ex") == 7200

    @pytest.mark.asyncio
    async def test_cluster_put_without_ttl(self, mock_cluster, loop, lcb):
        """Without TTL, cluster.set() should not receive ex=."""
        with patch(
            "lmcache.v1.storage_backend.connector.redis_connector.RedisCluster",
            return_value=mock_cluster,
        ):
            connector = RedisClusterConnector(
                hosts_and_ports=[("localhost", 7000)],
                username="",
                password="",
                loop=loop,
                local_cpu_backend=lcb,
                remote_ttl=None,
            )

        await connector._put(_make_key(), _make_memory_obj())

        for c in mock_cluster.set.call_args_list:
            ex_val = c.kwargs.get("ex")
            assert ex_val is None, f"Expected ex=None, got ex={ex_val}"


# ---------------------------------------------------------------------------
# RedisSentinelConnector TTL tests
# ---------------------------------------------------------------------------


class TestRedisSentinelConnectorTTL:
    """Test remote_ttl on RedisSentinelConnector (redis-sentinel://)."""

    @pytest.fixture
    def mock_master(self):
        master = AsyncMock()
        master.set = AsyncMock(return_value=True)
        return master

    @pytest.fixture
    def mock_slave(self):
        slave = MagicMock()
        slave.get = MagicMock(return_value=None)
        slave.exists = MagicMock(return_value=False)
        return slave

    @pytest.fixture
    def loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    def lcb(self):
        return _mock_local_cpu_backend()

    @pytest.mark.asyncio
    async def test_sentinel_put_with_ttl(self, mock_master, mock_slave, loop, lcb):
        """RedisSentinelConnector should pass ex= to master.set()."""
        with patch(
            "lmcache.v1.storage_backend.connector.redis_connector.redis.Sentinel",
            return_value=MagicMock(
                master_for=MagicMock(return_value=mock_master),
                slave_for=MagicMock(return_value=mock_slave),
            ),
        ):
            connector = RedisSentinelConnector(
                hosts_and_ports=[("localhost", 26379)],
                username="",
                password="",
                loop=loop,
                local_cpu_backend=lcb,
                remote_ttl=300,
            )

        await connector.put(_make_key(), _make_memory_obj())

        assert mock_master.set.call_count == 2
        for c in mock_master.set.call_args_list:
            assert c.kwargs.get("ex") == 300

    @pytest.mark.asyncio
    async def test_sentinel_put_without_ttl(self, mock_master, mock_slave, loop, lcb):
        """Without TTL, master.set() should not receive ex=."""
        with patch(
            "lmcache.v1.storage_backend.connector.redis_connector.redis.Sentinel",
            return_value=MagicMock(
                master_for=MagicMock(return_value=mock_master),
                slave_for=MagicMock(return_value=mock_slave),
            ),
        ):
            connector = RedisSentinelConnector(
                hosts_and_ports=[("localhost", 26379)],
                username="",
                password="",
                loop=loop,
                local_cpu_backend=lcb,
                remote_ttl=None,
            )

        await connector.put(_make_key(), _make_memory_obj())

        for c in mock_master.set.call_args_list:
            ex_val = c.kwargs.get("ex")
            assert ex_val is None, f"Expected ex=None, got ex={ex_val}"


# ---------------------------------------------------------------------------
# RESPConnector TTL warning test
# ---------------------------------------------------------------------------


class TestRESPConnectorTTL:
    """Test that RESPConnector warns when remote_ttl is set."""

    @pytest.fixture
    def loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    def lcb(self):
        return _mock_local_cpu_backend()

    def test_resp_warns_on_ttl(self, loop, lcb):
        """RESPConnector should log a warning when remote_ttl is set."""
        with patch(
            "lmcache.v1.storage_backend.connector.redis_connector.RESPClient"
        ):
            with patch(
                "lmcache.v1.storage_backend.connector.redis_connector.logger"
            ) as mock_logger:
                connector = RESPConnector(
                    host="localhost",
                    port=6379,
                    loop=loop,
                    local_cpu_backend=lcb,
                    remote_ttl=3600,
                )
                mock_logger.warning.assert_called_once()
                warn_args = str(mock_logger.warning.call_args)
                assert "remote_ttl" in warn_args or "3600" in warn_args

    def test_resp_no_warn_without_ttl(self, loop, lcb):
        """RESPConnector should NOT warn when remote_ttl is None."""
        with patch(
            "lmcache.v1.storage_backend.connector.redis_connector.RESPClient"
        ):
            with patch(
                "lmcache.v1.storage_backend.connector.redis_connector.logger"
            ) as mock_logger:
                connector = RESPConnector(
                    host="localhost",
                    port=6379,
                    loop=loop,
                    local_cpu_backend=lcb,
                    remote_ttl=None,
                )
                # No TTL-related warning should be emitted
                for c in mock_logger.warning.call_args_list:
                    assert "remote_ttl" not in str(c) and "TTL" not in str(c)


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestRemoteTTLConfig:
    """Test that remote_ttl flows through LMCacheEngineConfig correctly."""

    def test_default_ttl_is_none(self):
        """Default remote_ttl should be None (no expiry)."""
        config = LMCacheEngineConfig.from_defaults(
            remote_url="redis://localhost:6379",
        )
        assert config.remote_ttl is None

    def test_ttl_from_kwargs(self):
        """remote_ttl should be settable via from_defaults kwargs."""
        config = LMCacheEngineConfig.from_defaults(
            remote_url="redis://localhost:6379",
            remote_ttl=3600,
        )
        assert config.remote_ttl == 3600

    def test_ttl_zero_is_rejected(self):
        """remote_ttl=0 should be rejected — Redis requires positive integers for EX."""
        with pytest.raises(ValueError, match="remote_ttl must be a positive integer"):
            LMCacheEngineConfig.from_defaults(
                remote_url="redis://localhost:6379",
                remote_ttl=0,
            )

    def test_ttl_negative_is_rejected(self):
        """Negative remote_ttl should be rejected."""
        with pytest.raises(ValueError, match="remote_ttl must be a positive integer"):
            LMCacheEngineConfig.from_defaults(
                remote_url="redis://localhost:6379",
                remote_ttl=-10,
            )

    def test_ttl_from_env(self):
        """remote_ttl should be settable via LMCACHE_REMOTE_TTL env var."""
        os.environ["LMCACHE_REMOTE_TTL"] = "1800"
        try:
            config = LMCacheEngineConfig.from_defaults(
                remote_url="redis://localhost:6379",
            )
            config.update_config_from_env()
            assert config.remote_ttl == 1800
        finally:
            del os.environ["LMCACHE_REMOTE_TTL"]

    def test_ttl_none_from_env(self):
        """LMCACHE_REMOTE_TTL=none should result in None."""
        os.environ["LMCACHE_REMOTE_TTL"] = "none"
        try:
            config = LMCacheEngineConfig.from_defaults(
                remote_url="redis://localhost:6379",
            )
            config.update_config_from_env()
            assert config.remote_ttl is None
        finally:
            del os.environ["LMCACHE_REMOTE_TTL"]

# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    MPServerConfig,
    P2PConfig,
)
from lmcache.v1.multiprocess.server import run_cache_server


def test_run_cache_server_p2p_without_coordinator():
    # If P2P is enabled but coordinator URL is empty,
    # run_cache_server should raise ValueError.
    mp_config = MPServerConfig(p2p_config=P2PConfig(advertise_url="127.0.0.1:8555"))
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(size_in_bytes=1024, use_lazy=True),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    obs_config = ObservabilityConfig()
    coordinator_config = CoordinatorConfig(url="")

    with pytest.raises(ValueError, match="P2P requires a coordinator"):
        run_cache_server(
            mp_config=mp_config,
            storage_manager_config=storage_manager_config,
            obs_config=obs_config,
            coordinator_config=coordinator_config,
        )


def test_run_cache_server_p2p_with_incompatible_l1():
    # If P2P is enabled and coordinator URL is configured, but L1 is
    # incompatible (e.g. GDS L1), run_cache_server should raise ValueError.
    mp_config = MPServerConfig(p2p_config=P2PConfig(advertise_url="127.0.0.1:8555"))
    # L1 with GdsL1Config is incompatible with single memory region
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(size_in_bytes=1024, use_lazy=True),
            gds_l1_config=GdsL1Config(
                file_location="/dev/dax0.0",
                size_in_bytes=1024,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    obs_config = ObservabilityConfig()
    coordinator_config = CoordinatorConfig(url="http://localhost:9300")

    with pytest.raises(ValueError, match="P2P requires a single L1 memory region"):
        run_cache_server(
            mp_config=mp_config,
            storage_manager_config=storage_manager_config,
            obs_config=obs_config,
            coordinator_config=coordinator_config,
        )


@patch("lmcache.v1.multiprocess.server.MPCacheServerContext")
@patch("lmcache.v1.multiprocess.server._build_modules")
@patch("lmcache.v1.multiprocess.server.MPCacheServer")
@patch("lmcache.v1.multiprocess.server.MessageQueueServer")
@patch("lmcache.v1.multiprocess.server.init_observability")
@patch("lmcache.v1.multiprocess.server.maybe_initialize_trace_recorder")
def test_run_cache_server_p2p_valid_config(
    mock_maybe_trace,
    mock_init_obs,
    mock_mq_server,
    mock_mp_cache_server,
    mock_build_modules,
    mock_ctx_class,
):
    # If P2P is enabled, coordinator URL is configured, and L1 is
    # compatible, run_cache_server should proceed. We mock the internal
    # components so it returns instead of spinning a ZMQ server.
    mp_config = MPServerConfig(p2p_config=P2PConfig(advertise_url="127.0.0.1:8555"))
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(size_in_bytes=1024, use_lazy=True),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    obs_config = ObservabilityConfig()
    coordinator_config = CoordinatorConfig(url="http://localhost:9300")

    mock_build_modules.return_value = []

    # Should run successfully with return_engine=True
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
        coordinator_config=coordinator_config,
        return_engine=True,
    )

    mock_build_modules.assert_called_once()

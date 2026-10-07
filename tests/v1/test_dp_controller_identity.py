# SPDX-License-Identifier: Apache-2.0
"""Each vLLM data-parallel engine gets its own controller instance and port."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import set_dp_rank_controller_identity
import lmcache.integration.vllm.utils as lmcache_vllm_utils

PORTS = [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008]


def _configs(dp_size, dp_rank, engine_id="abc", world_size=1, local_rank=None):
    config = SimpleNamespace(
        lmcache_instance_id="pod-a", lmcache_worker_ports=list(PORTS)
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            data_parallel_rank_local=local_rank,
            data_parallel_size_local=None,
            world_size=world_size,
        ),
        kv_transfer_config=SimpleNamespace(engine_id=engine_id),
    )
    return config, vllm_config


def test_no_data_parallelism_unchanged():
    config, vllm_config = _configs(dp_size=1, dp_rank=0)
    set_dp_rank_controller_identity(config, vllm_config)
    assert config.lmcache_instance_id == "pod-a"
    assert config.lmcache_worker_ports == PORTS


def test_rank_from_engine_id_suffix():
    config, vllm_config = _configs(dp_size=8, dp_rank=0, engine_id="abc_dp3")
    set_dp_rank_controller_identity(config, vllm_config)
    assert config.lmcache_instance_id == "pod-a-dp3"
    assert config.lmcache_worker_ports[0] == 8004
    assert sorted(config.lmcache_worker_ports) == PORTS


def test_rank_from_parallel_config():
    config, vllm_config = _configs(dp_size=8, dp_rank=5)
    set_dp_rank_controller_identity(config, vllm_config)
    assert config.lmcache_instance_id == "pod-a-dp5"
    assert config.lmcache_worker_ports[0] == 8006


def test_applied_once_per_process():
    # Scheduler and worker roles may share one config singleton.
    config, vllm_config = _configs(dp_size=8, dp_rank=3, engine_id="abc_dp3")
    set_dp_rank_controller_identity(config, vllm_config)
    set_dp_rank_controller_identity(config, vllm_config)
    assert config.lmcache_instance_id == "pod-a-dp3"
    assert config.lmcache_worker_ports[0] == 8004


def test_all_local_ranks_distinct():
    seen = set()
    for rank in range(8):
        config, vllm_config = _configs(dp_size=8, dp_rank=rank, engine_id=f"x_dp{rank}")
        set_dp_rank_controller_identity(config, vllm_config)
        seen.add((config.lmcache_instance_id, config.lmcache_worker_ports[0]))
    assert len(seen) == 8
    assert len({port for _, port in seen}) == 8


def test_tp_workers_offset_and_local_rank_used_for_ports():
    # DP rank 3 is local rank 1 on its host; TP=4: ports 8005..8008.
    config, vllm_config = _configs(
        dp_size=4, dp_rank=3, engine_id="x_dp3", world_size=4, local_rank=1
    )
    set_dp_rank_controller_identity(config, vllm_config)
    assert config.lmcache_instance_id == "pod-a-dp3"
    assert config.lmcache_worker_ports[:4] == [8005, 8006, 8007, 8008]


def test_warns_when_ports_are_shared(monkeypatch: pytest.MonkeyPatch):
    # Assert on the logger call: LMCache loggers do not propagate and other tests
    # change logger state, so caplog misses the warning in the full suite.
    warn = MagicMock()
    monkeypatch.setattr(lmcache_vllm_utils.logger, "warning", warn)
    config, vllm_config = _configs(
        dp_size=8, dp_rank=1, engine_id="x_dp1", world_size=2
    )
    set_dp_rank_controller_identity(config, vllm_config)
    assert config.lmcache_worker_ports[0] == 8003
    warn.assert_called_once()
    assert "ports will be shared" in warn.call_args.args[0]

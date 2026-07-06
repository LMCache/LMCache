# SPDX-License-Identifier: Apache-2.0

"""Tests for Maru L1 config parsing and startup guards."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    MaruL1Config,
    StorageManagerConfig,
    parse_args,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig


def _args(*extra: str, l1_size_gb: str = "1") -> list[str]:
    """Minimal required flags (eviction policy + L1 size) plus extras."""
    return [
        "--eviction-policy",
        "LRU",
        "--l1-size-gb",
        l1_size_gb,
        "--no-l1-use-lazy",
        *extra,
    ]


def test_no_maru_flags_leaves_maru_config_none():
    cfg = parse_args(_args())
    assert cfg.l1_manager_config.memory_config.maru_config is None


def test_maru_flags_build_maru_config():
    cfg = parse_args(
        _args(
            "--maru-server-url",
            "maru://localhost:9000",
            "--maru-pool-size-gb",
            "8",
            "--maru-instance-id",
            "node-a",
        )
    )
    maru = cfg.l1_manager_config.memory_config.maru_config
    assert isinstance(maru, MaruL1Config)
    assert maru.server_url == "maru://localhost:9000"
    assert maru.pool_size_bytes == 8 * (1 << 30)
    assert maru.instance_id == "node-a"


def test_maru_without_pool_size_raises():
    with pytest.raises(ValueError):
        parse_args(_args("--maru-server-url", "maru://localhost:9000"))


# ---------------------------------------------------------------------------
# C9: validate_storage_manager_config rejects unsupported maru combinations.
# ---------------------------------------------------------------------------


def _maru_memory(**overrides) -> L1MemoryManagerConfig:
    return L1MemoryManagerConfig(
        size_in_bytes=0,
        use_lazy=False,
        maru_config=MaruL1Config(
            server_url="maru://localhost:5555",
            pool_size_bytes=1 << 20,
            instance_id="t",
        ),
        **overrides,
    )


def _maru_sm_config(
    *, memory=None, gds=None, store_policy="default", adapters=None
) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=memory if memory is not None else _maru_memory(),
            gds_l1_config=gds,
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=adapters or []),
        store_policy=store_policy,
    )


def test_maru_plus_copy_l2_is_accepted():
    # A copy-type L2 (mock) needs no registerable region -> allowed.
    _maru_sm_config(
        adapters=[MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=1.0)]
    )


def test_maru_rejects_gds_l1():
    with pytest.raises(ValueError, match="gds"):
        _maru_sm_config(
            gds=GdsL1Config(file_location="/tmp/gds", size_in_bytes=1 << 20)
        )


def test_maru_rejects_devdax_l1():
    with pytest.raises(ValueError, match="devdax"):
        _maru_sm_config(memory=_maru_memory(devdax_path="/dev/dax0.0", shm_name=""))


def test_maru_rejects_skip_l1_store_policy():
    with pytest.raises(ValueError, match="skip_l1"):
        _maru_sm_config(store_policy="skip_l1")


def test_maru_rejects_registered_l2(monkeypatch):
    # Simulate a registered/RDMA adapter via the shared region classifier.
    monkeypatch.setattr(
        "lmcache.v1.distributed.config._requires_single_l1_memory_region",
        lambda adapter_config: "nixl_store",
    )
    with pytest.raises(ValueError, match="registerable"):
        _maru_sm_config(
            adapters=[MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=1.0)]
        )


# NOTE: the maru transfer-mode guard (maru requires supported_transfer_mode
# == "lmcache_driven") is an inline check in run_http_server, alongside the
# existing p2p startup guards there. Those startup guards are not unit-tested
# (importing run_http_server pulls the full server chain); this one follows
# the same precedent.

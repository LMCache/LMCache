# SPDX-License-Identifier: Apache-2.0

"""Tests for Maru L1 config parsing."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import MaruL1Config, parse_args


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

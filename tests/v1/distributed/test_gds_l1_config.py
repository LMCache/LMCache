# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for the GDS L1 tier."""

# Standard
from pathlib import Path
import argparse

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    GDSL1ManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.gpu_connector import gds_backends
from lmcache.v1.multiprocess.config import add_mp_server_args


@pytest.mark.parametrize("backend", ["ugds", "custom_backend"])
def test_cli_passes_backend_name_to_config(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "custom_backend.py").write_text(
        "raise RuntimeError('must stay lazy')\n"
    )
    monkeypatch.setattr(
        gds_backends, "__path__", [*gds_backends.__path__, str(tmp_path)]
    )
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)

    config = parse_args_to_config(
        parser.parse_args(
            [
                "--l1-size-gb",
                "8",
                "--eviction-policy",
                "LRU",
                "--gds-l1-backend",
                backend,
                "--gds-l1-path",
                "/dev/ugds_drv0",
            ]
        )
    )

    gds_config = config.l1_manager_config.gds_l1_config
    assert gds_config is not None
    assert gds_config.backend == backend
    assert gds_config.file_location == "/dev/ugds_drv0"
    assert gds_config.size_in_bytes == 8 << 30


@pytest.mark.parametrize("concurrency", [-1, 5, True, 1.5])
def test_invalid_concurrency_rejected(concurrency: int | float) -> None:
    with pytest.raises(ValueError, match="io_concurrency"):
        GdsL1Config("/test", 4096, io_concurrency=concurrency)  # type: ignore[arg-type]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_io_budget_rejected(budget: int | float) -> None:
    with pytest.raises(ValueError, match="max_inflight_bytes"):
        GdsL1Config("/test", 4096, max_inflight_bytes=budget)  # type: ignore[arg-type]


def test_async_config_cli_and_json() -> None:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    config = parse_args_to_config(
        parser.parse_args(
            [
                "--l1-size-gb",
                "8",
                "--eviction-policy",
                "LRU",
                "--gds-l1-path",
                "/test",
                "--gds-l1-io-concurrency",
                "4",
                "--gds-l1-max-inflight-bytes",
                "67108864",
            ]
        )
    )
    gds = config.l1_manager_config.gds_l1_config
    assert gds is not None
    assert gds.io_concurrency == 4
    assert gds.max_inflight_bytes == 64 << 20
    parsed = GDSL1ManagerConfig.from_dict(
        {
            "type": "gds",
            "tag": "disk",
            "size_gb": 8,
            "path": "/test",
            "io_concurrency": 1,
            "max_inflight_bytes": 32 << 20,
        },
        EvictionConfig(eviction_policy="LRU"),
    )
    assert parsed.gds_l1_config is not None
    assert parsed.gds_l1_config.io_concurrency == 1
    assert parsed.gds_l1_config.max_inflight_bytes == 32 << 20

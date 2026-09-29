# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for the GDS L1 tier."""

# Standard
from pathlib import Path
import argparse
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
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
                "--eviction-policy",
                "LRU",
                "--l1-manager",
                json.dumps(
                    {
                        "type": "GDS",
                        "tag": "nvme",
                        "size_gb": 8,
                        "backend": backend,
                        "path": "/dev/ugds_drv0",
                    }
                ),
            ]
        )
    )

    assert config.l1_manager_config is not None
    gds_config = config.l1_manager_config.gds_l1_config
    assert gds_config is not None
    assert gds_config.backend == backend
    assert gds_config.file_location == "/dev/ugds_drv0"
    assert gds_config.size_in_bytes == 8 << 30

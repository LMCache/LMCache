# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path

# Third Party
import pytest


@pytest.fixture(autouse=True)
def host_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The env config.toml reads, pointed at the test's tmp dir."""
    monkeypatch.setenv("PRT_ROOT", str(tmp_path))
    monkeypatch.setenv("PRT_HF_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("PRT_RDMA_DEVICES", "mlx5_0,mlx5_1")
